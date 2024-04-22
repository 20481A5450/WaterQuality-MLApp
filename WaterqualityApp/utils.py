from django.core.mail import EmailMessage
from django.conf import settings
from datetime import datetime

def send_email(subject, message, to_email, file_path):
    subject = f"Water Quality Report on {datetime.now()}"
    message = " "
    from_email = settings.EMAIL_HOST_USER
    recipient_list = ["shaikzohaibpardeep@gmail.com"]
    file_path = f"{settings.BASE_DIR}/WaterqualityApp/static/reports/waterquality_report.xlsx"

    msg = EmailMessage(subject=subject, body=message, from_email=from_email, to=recipient_list)
    msg.content_subtype = "html"
    msg.attach_file(file_path)
    msg.send()