import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


def send_email(to_email: str, subject: str, html_body: str):
    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        print(f"[EMAIL STUB] To: {to_email} | Subject: {subject}")
        print(f"[EMAIL STUB] Body: {html_body[:200]}...")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, to_email, msg.as_string())


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


def send_rejection_email(to_email: str, name: str, job_title: str):
    subject = f"HappyHR - Application Update for {job_title}"
    body = f"""
    <h2>Hi {name},</h2>
    <p>Thank you for your interest in the <strong>{job_title}</strong> position.</p>
    <p>After reviewing your application, we've decided to move forward with other candidates whose experience more closely matches our current requirements.</p>
    <p>We encourage you to apply again in the future.</p>
    <p>Best regards,<br>The HappyHR Team</p>
    """
    send_email(to_email, subject, body)


def send_decision_email(to_email: str, name: str, job_title: str, decision: str):
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
        <p>After careful consideration, we've decided to proceed with other candidates.</p>
        <p>We wish you the best in your job search.</p>
        <p>Best regards,<br>The HappyHR Team</p>
        """
    send_email(to_email, subject, body)
