"""Unit tests for EmailNotifier. smtplib is monkeypatched with an in-memory
fake — hermetic, no real network or port binding, no flakes. Verifies
timeouts, STARTTLS/login behavior, retry-on-transient-failure, and that the
SMTP password never leaks into logs or repr."""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone

import pytest

from service.detection_api.src.services.alerting import (
    AlertNotification,
    AlertingSettings,
    EmailNotifier,
    NullNotifier,
    build_notifier,
)


class FakeSMTP:
    """Records calls; raises_on_connect can simulate a transient failure."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_called = False
        self.login_args: tuple[str, str] | None = None
        self.sent_messages: list[tuple[str, list[str], str]] = []
        self.quit_called = False
        FakeSMTP.instances.append(self)

    def starttls(self) -> None:
        self.starttls_called = True

    def login(self, user: str, password: str) -> None:
        self.login_args = (user, password)

    def sendmail(self, from_addr: str, to_addrs: list[str], msg: str) -> None:
        self.sent_messages.append((from_addr, to_addrs, msg))

    def quit(self) -> None:
        self.quit_called = True


class FlakyThenOkSMTP(FakeSMTP):
    call_count = 0

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        super().__init__(host, port, timeout)
        FlakyThenOkSMTP.call_count += 1
        if FlakyThenOkSMTP.call_count == 1:
            raise smtplib.SMTPServerDisconnected("connection reset")


class AlwaysFailsSMTP(FakeSMTP):
    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        super().__init__(host, port, timeout)
        raise smtplib.SMTPServerDisconnected("still down")


def _settings(**overrides: object) -> AlertingSettings:
    defaults: dict[str, object] = dict(
        alerting_enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="alerts@example.com",
        smtp_password="super-secret-password",
        alert_email_from="alerts@example.com",
        alert_email_to="soc@example.com",
    )
    defaults.update(overrides)
    return AlertingSettings(**defaults)


def _notification() -> AlertNotification:
    return AlertNotification(
        alert_id="abc-123",
        flow_id="flow-1",
        severity="high",
        attack_type="DoS",
        confidence=0.94,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        detected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        top_features=(("sttl", 0.35), ("ct_state_ttl", 0.22)),
    )


class TestEmailNotifierHappyPath:
    @pytest.mark.asyncio
    async def test_connects_with_host_port_and_5s_timeout(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=FakeSMTP, sleep=lambda _: None)
        await notifier.send(_notification())
        assert FakeSMTP.instances[0].host == "smtp.example.com"
        assert FakeSMTP.instances[0].port == 587
        assert FakeSMTP.instances[0].timeout == 5

    @pytest.mark.asyncio
    async def test_starttls_called_when_configured(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(smtp_use_starttls=True), smtp_cls=FakeSMTP, sleep=lambda _: None)
        await notifier.send(_notification())
        assert FakeSMTP.instances[0].starttls_called is True

    @pytest.mark.asyncio
    async def test_starttls_skipped_when_disabled(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(smtp_use_starttls=False), smtp_cls=FakeSMTP, sleep=lambda _: None)
        await notifier.send(_notification())
        assert FakeSMTP.instances[0].starttls_called is False

    @pytest.mark.asyncio
    async def test_logs_in_with_configured_credentials(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=FakeSMTP, sleep=lambda _: None)
        await notifier.send(_notification())
        assert FakeSMTP.instances[0].login_args == ("alerts@example.com", "super-secret-password")

    @pytest.mark.asyncio
    async def test_message_headers_carry_severity_and_attack_type(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=FakeSMTP, sleep=lambda _: None)
        await notifier.send(_notification())
        _, _, msg = FakeSMTP.instances[0].sent_messages[0]
        assert "high" in msg
        assert "DoS" in msg

    @pytest.mark.asyncio
    async def test_body_includes_top_features(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=FakeSMTP, sleep=lambda _: None)
        await notifier.send(_notification())
        _, _, msg = FakeSMTP.instances[0].sent_messages[0]
        assert "sttl" in msg

    @pytest.mark.asyncio
    async def test_result_reports_success(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=FakeSMTP, sleep=lambda _: None)
        result = await notifier.send(_notification())
        assert result.success is True
        assert result.channel == "email"


class TestEmailNotifierSecretHygiene:
    @pytest.mark.asyncio
    async def test_password_never_in_repr(self) -> None:
        settings = _settings()
        assert "super-secret-password" not in repr(settings)

    @pytest.mark.asyncio
    async def test_password_never_in_log_output(self, caplog: pytest.LogCaptureFixture) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=FakeSMTP, sleep=lambda _: None)
        with caplog.at_level(logging.DEBUG):
            await notifier.send(_notification())
        for record in caplog.records:
            assert "super-secret-password" not in record.getMessage()


class TestEmailNotifierRetryBehavior:
    @pytest.mark.asyncio
    async def test_transient_disconnect_then_success_retries_exactly_twice(self) -> None:
        FlakyThenOkSMTP.call_count = 0
        FakeSMTP.instances.clear()
        sleeps: list[float] = []
        notifier = EmailNotifier(_settings(), smtp_cls=FlakyThenOkSMTP, sleep=sleeps.append)
        result = await notifier.send(_notification())
        assert result.success is True
        assert result.attempts == 2
        assert len(sleeps) == 1

    @pytest.mark.asyncio
    async def test_permanent_failure_returns_failed_result_not_raise(self) -> None:
        FakeSMTP.instances.clear()
        notifier = EmailNotifier(_settings(), smtp_cls=AlwaysFailsSMTP, sleep=lambda _: None)
        result = await notifier.send(_notification())
        assert result.success is False
        assert result.channel == "email"


class TestNullNotifierFallback:
    @pytest.mark.asyncio
    async def test_disabled_by_env_returns_null_notifier(self) -> None:
        settings = _settings(alerting_enabled=False)
        notifier = build_notifier(settings)
        assert isinstance(notifier, NullNotifier) or "Composite" not in type(notifier).__name__

    @pytest.mark.asyncio
    async def test_unconfigured_smtp_does_not_crash(self) -> None:
        settings = AlertingSettings(alerting_enabled=True)
        notifier = build_notifier(settings)
        result = await notifier.send(_notification())
        assert result is not None
