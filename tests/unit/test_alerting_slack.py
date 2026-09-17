"""Unit tests for SlackNotifier. httpx transport is replaced with
httpx.MockTransport — hermetic, no real network. Verifies the POST target,
payload shape, timeout, retry-on-429 (not on other 4xx), and that the
webhook URL is validated at config time (SSRF guard) and never logged."""
from __future__ import annotations

import logging

import httpx
import pytest

from service.detection_api.src.services.alerting import (
    AlertNotification,
    AlertingSettings,
    SlackNotifier,
)
from datetime import datetime, timezone

VALID_WEBHOOK = "https://hooks.slack.com/services/T000/B000/XXXXXXXXXXXXXXXXXXXXXXXX"


def _notification() -> AlertNotification:
    return AlertNotification(
        alert_id="abc-123",
        flow_id="flow-1",
        severity="critical",
        attack_type="Exploits",
        confidence=0.88,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        detected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        top_features=(("sttl", 0.35),),
    )


class TestSlackNotifierHappyPath:
    @pytest.mark.asyncio
    async def test_posts_to_configured_webhook_url(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["json"] = httpx.Request.read(request) and request.content
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        notifier = SlackNotifier(settings, transport=transport, sleep=lambda _: None)

        result = await notifier.send(_notification())

        assert result.success is True
        assert seen["url"] == VALID_WEBHOOK

    @pytest.mark.asyncio
    async def test_payload_contains_severity_and_attack_type(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        notifier = SlackNotifier(settings, transport=transport, sleep=lambda _: None)

        await notifier.send(_notification())

        body_text = str(captured["body"])
        assert "critical" in body_text
        assert "Exploits" in body_text

    @pytest.mark.asyncio
    async def test_uses_5_second_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        notifier = SlackNotifier(settings, transport=transport, sleep=lambda _: None)
        assert notifier.timeout_seconds == 5.0


class TestSlackNotifierRetryBehavior:
    @pytest.mark.asyncio
    async def test_429_is_retried_with_injected_sleep(self) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(429, json={"error": "rate_limited"})
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        sleeps: list[float] = []
        notifier = SlackNotifier(settings, transport=transport, sleep=sleeps.append)

        result = await notifier.send(_notification())

        assert result.success is True
        assert calls["count"] == 2
        assert len(sleeps) == 1

    @pytest.mark.asyncio
    async def test_400_is_not_retried(self) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(400, json={"error": "bad_request"})

        transport = httpx.MockTransport(handler)
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        notifier = SlackNotifier(settings, transport=transport, sleep=lambda _: None)

        result = await notifier.send(_notification())

        assert result.success is False
        assert calls["count"] == 1


class TestSlackWebhookSsrfGuard:
    def test_non_slack_url_rejected_at_config_time(self) -> None:
        with pytest.raises(Exception):
            AlertingSettings(alerting_enabled=True, slack_webhook_url="https://evil.example.com/steal")

    def test_valid_slack_url_accepted(self) -> None:
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        assert settings.slack_webhook_url is not None


class TestSlackWebhookSecretHygiene:
    @pytest.mark.asyncio
    async def test_webhook_url_never_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        settings = AlertingSettings(alerting_enabled=True, slack_webhook_url=VALID_WEBHOOK)
        notifier = SlackNotifier(settings, transport=transport, sleep=lambda _: None)

        with caplog.at_level(logging.DEBUG):
            await notifier.send(_notification())

        for record in caplog.records:
            assert VALID_WEBHOOK not in record.getMessage()
