"""
Quick test to verify email service works with .env configuration
"""
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.email_service import EmailService
from app.core.config import settings

print("=" * 70)
print("Email Service Configuration Test")
print("=" * 70)

# Check configuration
print(f"\nAPI Key configured: {bool(settings.BREVO_API_KEY)}")
if settings.BREVO_API_KEY:
    print(f"API Key: {settings.BREVO_API_KEY[:20]}...")
else:
    print("ERROR: BREVO_API_KEY not set in .env file!")
    exit(1)

print(f"Sender Email: {settings.BREVO_SENDER_EMAIL}")
print(f"Sender Name: {settings.BREVO_SENDER_NAME}")

# Test email service
print("\n" + "=" * 70)
print("Sending test email...")
print("=" * 70)

email_service = EmailService()
result = email_service.send_verification_otp(
    email="mrsachinchaurasiya@gmail.com",
    name="Sachin",
    otp="999999"
)

print("\n" + "=" * 70)
if result:
    print("SUCCESS! Email sent successfully!")
    print("\nThe email service is working correctly.")
    print("You can now restart your server to use the updated configuration.")
else:
    print("FAILED! Email could not be sent.")
    print("\nPlease check:")
    print("1. BREVO_API_KEY is correct in .env file")
    print("2. Sender email is verified in Brevo dashboard")
print("=" * 70)
