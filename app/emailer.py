from __future__ import annotations
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_trigger_email(
    to_emails: list[str],
    subject: str,
    content: str,
) -> tuple[bool, str]:
    api_key = os.getenv("SENDGRID_API_KEY", "")
    from_email = os.getenv("FROM_EMAIL", "")
    if not api_key or not from_email:
        return False, "SendGrid not configured (SENDGRID_API_KEY / FROM_EMAIL)."

    try:
        msg = Mail(
            from_email=from_email,
            to_emails=to_emails,
            subject=subject,
            plain_text_content=content,
        )
        sg = SendGridAPIClient(api_key)
        resp = sg.send(msg)
        return True, f"Sent (status {resp.status_code})."
    except Exception as e:
        return False, f"Send failed: {e}"
