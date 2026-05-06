"""
Alert module — console logging + optional email notifications.

Email config (optional, set in .env):
  ALERT_EMAIL_FROM=you@gmail.com
  ALERT_EMAIL_TO=you@gmail.com
  ALERT_EMAIL_PASSWORD=app-password
  ALERT_EMAIL_SMTP=smtp.gmail.com
  ALERT_EMAIL_PORT=587
"""

from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Optional

from loguru import logger

from config import settings
from strategy import Signal, SignalType


_EMOJI = {SignalType.BUY: "[BUY]", SignalType.SELL: "[SELL]", SignalType.HOLD: "[HOLD]"}


class Notifier:

    def __init__(self):
        self._email_from = getattr(settings, "alert_email_from", "")
        self._email_to   = getattr(settings, "alert_email_to", "")
        self._email_pw   = getattr(settings, "alert_email_password", "")
        self._smtp_host  = getattr(settings, "alert_email_smtp", "smtp.gmail.com")
        self._smtp_port  = getattr(settings, "alert_email_port", 587)
        self._email_enabled = bool(self._email_from and self._email_to and self._email_pw)

    def send_signal(self, signal: Signal) -> bool:
        tag = _EMOJI[signal.signal]
        title = f"{tag} [{signal.strategy}] {signal.symbol} @ {signal.price:.2f}"
        content = self._format_signal(signal)
        logger.info(f"Signal alert: {title}")
        return self._send(title, content)

    def send_risk_alert(self, msg: str) -> bool:
        logger.warning(f"Risk alert: {msg}")
        return self._send("Risk Alert", msg)

    def send_daily_report(self, report: dict) -> bool:
        title = f"Daily Report {datetime.now().strftime('%Y-%m-%d')}"
        lines = [f"# {title}", ""]
        for k, v in report.items():
            lines.append(f"- **{k}**: {v}")
        content = "\n".join(lines)
        logger.info("Sending daily report")
        return self._send(title, content)

    @staticmethod
    def _format_signal(sig: Signal) -> str:
        lines = [
            f"{_EMOJI[sig.signal]} {sig.symbol} {sig.signal.value}",
            f"Strategy : {sig.strategy}",
            f"Price    : {sig.price:.2f}",
            f"Confidence: {sig.confidence:.0%}",
            f"Time     : {sig.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Reason   : {sig.reason}",
        ]
        if sig.metadata.get("key_factors"):
            lines.append("Key factors:")
            for f_ in sig.metadata["key_factors"]:
                lines.append(f"  - {f_}")
        return "\n".join(lines)

    def _send(self, title: str, content: str) -> bool:
        if not self._email_enabled:
            return False
        try:
            msg = MIMEText(content)
            msg["Subject"] = title
            msg["From"] = self._email_from
            msg["To"] = self._email_to
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as s:
                s.starttls()
                s.login(self._email_from, self._email_pw)
                s.send_message(msg)
            logger.debug(f"Email sent: {title}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
