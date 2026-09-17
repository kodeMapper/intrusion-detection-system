"""Email/Slack alert notifications.

Config is env-only (IDPS_ prefix) with secrets typed as SecretStr so they
can't leak through a repr, a log line, or a validation error. Notifiers are
invoked via FastAPI BackgroundTasks from detection.py so SMTP/Slack latency
never blocks the HTTP response — "fires within 5s" means dispatch-initiated,
the only reading compatible with a background task.
"""
from __future__ import annotations

import logging
import smtplib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import anyio
import httpx
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SLACK_WEBHOOK_PREFIX = "https://hooks.slack.com/services/"
_SMTP_RETRY_DELAYS_S: tuple[float, ...] = (0.5, 1.5)
_SLACK_TIMEOUT_S = 5.0
_SLACK_RETRY_DELAY_S = 1.0
_SLACK_MAX_ATTEMPTS = 2

# httpx logs "HTTP Request: POST <url> ..." at INFO by default, which would
# otherwise leak the webhook URL's embedded secret path into application
# logs on every send. Suppress httpx's own request/response logging here.
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass(frozen=True, slots=True)
class AlertNotification:
    alert_id: str | None
    flow_id: str
    severity: str
    attack_type: str
    confidence: float
    src_ip: str
    dst_ip: str
    detected_at: datetime
    top_features: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class NotificationResult:
    success: bool
    channel: str
    detail: str | None = None
    attempts: int = 1


class Notifier(Protocol):
    async def send(self, notification: AlertNotification) -> NotificationResult: ...


class AlertingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IDPS_", extra="ignore")

    alerting_enabled: bool = True
    min_notify_severity: str = "high"
    suppression_window_s: float = 60.0

    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    smtp_use_starttls: bool = True
    alert_email_from: str | None = None
    alert_email_to: str | None = None

    slack_webhook_url: SecretStr | None = None

    @field_validator("slack_webhook_url")
    @classmethod
    def _validate_slack_webhook(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return value
        if not value.get_secret_value().startswith(_SLACK_WEBHOOK_PREFIX):
            raise ValueError(f"slack_webhook_url must start with {_SLACK_WEBHOOK_PREFIX}")
        return value


def _format_features(top_features: Sequence[tuple[str, float]]) -> list[str]:
    return [f"{name}: {value:+.3f}" for name, value in top_features]


class EmailNotifier:
    def __init__(
        self,
        settings: AlertingSettings,
        *,
        smtp_cls: type = smtplib.SMTP,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._smtp_cls = smtp_cls
        self._sleep = sleep
        self._logger = logger or logging.getLogger(__name__)

    async def send(self, notification: AlertNotification) -> NotificationResult:
        return await anyio.to_thread.run_sync(self._send_sync, notification)

    def _build_message(self, notification: AlertNotification) -> str:
        lines = [
            f"From: {self._settings.alert_email_from}",
            f"To: {self._settings.alert_email_to}",
            f"Subject: [IDPS] {notification.severity} alert: {notification.attack_type}",
            "",
            f"Attack type: {notification.attack_type}",
            f"Severity: {notification.severity}",
            f"Confidence: {notification.confidence:.2f}",
            f"Src IP: {notification.src_ip} -> Dst IP: {notification.dst_ip}",
            f"Detected at: {notification.detected_at.isoformat()}",
            "",
            "Top contributing features:",
            *(f"  - {line}" for line in _format_features(notification.top_features)),
        ]
        return "\r\n".join(lines)

    def _send_sync(self, notification: AlertNotification) -> NotificationResult:
        last_error: Exception | None = None
        max_attempts = len(_SMTP_RETRY_DELAYS_S) + 1
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._sleep(_SMTP_RETRY_DELAYS_S[attempt - 2])
            try:
                server = self._smtp_cls(self._settings.smtp_host, self._settings.smtp_port, timeout=5)
                try:
                    if self._settings.smtp_use_starttls:
                        server.starttls()
                    if self._settings.smtp_user and self._settings.smtp_password:
                        server.login(self._settings.smtp_user, self._settings.smtp_password.get_secret_value())
                    server.sendmail(
                        self._settings.alert_email_from,
                        [self._settings.alert_email_to],
                        self._build_message(notification),
                    )
                finally:
                    try:
                        server.quit()
                    except Exception as exc:
                        self._logger.debug("SMTP quit failed (ignored): %s", type(exc).__name__)
                return NotificationResult(success=True, channel="email", attempts=attempt)
            except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, ConnectionError) as exc:
                last_error = exc
                self._logger.warning("transient SMTP failure on attempt %d: %s", attempt, type(exc).__name__)
                continue
            except Exception as exc:
                self._logger.error("permanent SMTP failure: %s", type(exc).__name__)
                return NotificationResult(success=False, channel="email", detail=str(exc), attempts=attempt)
        return NotificationResult(success=False, channel="email", detail=str(last_error), attempts=max_attempts)


class SlackNotifier:
    def __init__(
        self,
        settings: AlertingSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep
        self._logger = logger or logging.getLogger(__name__)
        self.timeout_seconds = _SLACK_TIMEOUT_S

    def _build_payload(self, notification: AlertNotification) -> dict:
        lines = [
            f"[{notification.severity}] {notification.attack_type} detected",
            f"src: {notification.src_ip} -> dst: {notification.dst_ip}",
            f"confidence: {notification.confidence:.2f}",
            *_format_features(notification.top_features),
        ]
        return {"text": "\n".join(lines)}

    async def send(self, notification: AlertNotification) -> NotificationResult:
        if self._settings.slack_webhook_url is None:
            return NotificationResult(success=False, channel="slack", detail="slack not configured")

        url = self._settings.slack_webhook_url.get_secret_value()
        payload = self._build_payload(notification)

        async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout_seconds) as client:
            for attempt in range(1, _SLACK_MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(url, json=payload)
                except httpx.HTTPError as exc:
                    self._logger.error("slack request failed: %s", type(exc).__name__)
                    return NotificationResult(success=False, channel="slack", detail=str(exc), attempts=attempt)

                if response.status_code < 300:
                    return NotificationResult(success=True, channel="slack", attempts=attempt)
                if response.status_code == 429 and attempt < _SLACK_MAX_ATTEMPTS:
                    self._sleep(_SLACK_RETRY_DELAY_S)
                    continue
                self._logger.warning("slack webhook returned HTTP %d", response.status_code)
                return NotificationResult(
                    success=False, channel="slack", detail=f"HTTP {response.status_code}", attempts=attempt
                )
        return NotificationResult(success=False, channel="slack", detail="exhausted retries", attempts=_SLACK_MAX_ATTEMPTS)


class NullNotifier:
    async def send(self, notification: AlertNotification) -> NotificationResult:
        return NotificationResult(success=True, channel="null", detail="alerting disabled or unconfigured")


class CompositeNotifier:
    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        self._notifiers = list(notifiers)

    async def send(self, notification: AlertNotification) -> NotificationResult:
        results = [await notifier.send(notification) for notifier in self._notifiers]
        detail = "; ".join(f"{r.channel}={r.success}" for r in results)
        return NotificationResult(success=any(r.success for r in results), channel="composite", detail=detail)


def build_notifier(settings: AlertingSettings) -> Notifier:
    if not settings.alerting_enabled:
        return NullNotifier()

    notifiers: list[Notifier] = []
    if settings.smtp_host and settings.alert_email_from and settings.alert_email_to:
        notifiers.append(EmailNotifier(settings))
    if settings.slack_webhook_url is not None:
        notifiers.append(SlackNotifier(settings))

    if not notifiers:
        return NullNotifier()
    if len(notifiers) == 1:
        return notifiers[0]
    return CompositeNotifier(notifiers)
