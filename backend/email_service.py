import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


def send_email(to_email: str, subject: str, html_body: str):
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER", os.getenv("GMAIL_USER"))
    smtp_password = os.getenv("SMTP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD"))

    if not smtp_user or not smtp_password:
        print(f"[EMAIL STUB] To: {to_email} | Subject: {subject}")
        print(f"[EMAIL STUB] Body: {html_body[:200]}...")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email to {to_email}: {e}")


def send_interview_invite(to_email: str, name: str, token: str, job_title: str):
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    link = f"{frontend_url}/interview/{token}"
    subject = f"HappyHR - Interview Invitation for {job_title}"
    body = f"""
    <h2>Hi {name},</h2>
    <p>Congratulations! Your application for <strong>{job_title}</strong> has passed our initial screening.</p>
    <p>We'd like to invite you to a short AI-powered voice interview.</p>
    <p><a href="{link}" style="background:#2563eb;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold;">
        Start Your Interview
    </a></p>
    <p>The interview takes about 8 minutes. You'll need a microphone and a quiet environment.</p>
    <p>Good luck!<br>The HappyHR Team</p>
    """
    send_email(to_email, subject, body)


def send_rejection_email(
    to_email: str,
    name: str,
    job_title: str,
    screening_feedback: str | None = None,
):
    feedback_block = (
        f"<p><strong>Screening feedback:</strong> {screening_feedback}</p>"
        if screening_feedback
        else ""
    )
    subject = f"HappyHR - Application Update for {job_title}"
    body = f"""
    <h2>Hi {name},</h2>
    <p>Thank you for your interest in the <strong>{job_title}</strong> position.</p>
    {feedback_block}
    <p>After reviewing your application, we've decided to move forward with other candidates whose experience more closely matches our current requirements.</p>
    <p>We encourage you to apply again in the future.</p>
    <p>Best regards,<br>The HappyHR Team</p>
    """
    send_email(to_email, subject, body)


def send_decision_email(
    to_email: str,
    name: str,
    job_title: str,
    decision: str,
    summary_candidate: str | None = None,
):
    feedback_block = (
        f"<p><strong>Interview feedback:</strong> {summary_candidate}</p>"
        if summary_candidate
        else ""
    )
    if decision == "accept":
        subject = f"HappyHR - Great News About Your {job_title} Application!"
        body = f"""
        <h2>Hi {name},</h2>
        <p>We're pleased to inform you that you've been <strong>selected</strong> for the <strong>{job_title}</strong> position!</p>
        <p>Our team will reach out shortly with next steps.</p>
        <p>Congratulations!<br>The HappyHR Team</p>
        """
    else:
        subject = f"HappyHR - Update on Your {job_title} Application"
        body = f"""
        <h2>Hi {name},</h2>
        <p>Thank you for completing the interview for the <strong>{job_title}</strong> position.</p>
        {feedback_block}
        <p>After careful consideration, we've decided to proceed with other candidates.</p>
        <p>We wish you the best in your job search.</p>
        <p>Best regards,<br>The HappyHR Team</p>
        """
    send_email(to_email, subject, body)
