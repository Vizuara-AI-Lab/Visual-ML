"""
Email service using SMTP (Brevo).
Handles OTP verification, welcome emails, and marketing campaigns.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
from app.core.logging import logger
from typing import Optional


class EmailService:
    """Email service using SMTP."""

    def __init__(self):
        """Initialize SMTP configuration."""
        self.smtp_server = "smtp-relay.brevo.com"
        self.smtp_port = 587
        self.smtp_user = "a0917001@smtp-brevo.com"
        self.smtp_password = "UNZvCfIAEwXk94Q"
        self.sender_email = settings.BREVO_SENDER_EMAIL
        self.sender_name = settings.BREVO_SENDER_NAME

    def _send_email(
        self,
        to_email: str,
        to_name: str,
        subject: str,
        html_content: str,
    ) -> bool:
        """
        Send email using SMTP.
        
        Args:
            to_email: Recipient email
            to_name: Recipient name
            subject: Email subject
            html_content: HTML email content
            
        Returns:
            bool: True if sent successfully
        """
        try:
            logger.info(f"Preparing to send email to {to_email}")
            logger.info(f"SMTP Server: {self.smtp_server}:{self.smtp_port}")
            logger.info(f"SMTP User: {self.smtp_user}")
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.sender_name} <{self.sender_email}>"
            message["To"] = f"{to_name} <{to_email}>"
            
            # Attach HTML content
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            logger.info(f"Connecting to SMTP server...")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                logger.info("Connected to SMTP server")
                server.set_debuglevel(1)  # Enable debug output
                
                logger.info("Starting TLS...")
                server.starttls()
                logger.info("TLS started")
                
                logger.info(f"Logging in as {self.smtp_user}...")
                server.login(self.smtp_user, self.smtp_password)
                logger.info("Login successful")
                
                logger.info(f"Sending email to {to_email}...")
                server.send_message(message)
                logger.info(f"✅ Email sent successfully to {to_email}")
            
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ SMTP Authentication failed: {e}")
            logger.error(f"Check SMTP credentials in email_service.py")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP error sending email to {to_email}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error sending email to {to_email}: {e}")
            logger.exception("Full traceback:")
            return False

    def send_verification_otp(self, email: str, name: str, otp: str) -> bool:
        """
        Send OTP verification email.
        
        Args:
            email: User email
            name: User name
            otp: 6-digit OTP code
            
        Returns:
            bool: True if sent successfully
        """
        logger.info(f"📧 Sending OTP verification email to {email}")
        logger.info(f"OTP Code: {otp}")
        
        subject = "Verify your Visual ML account"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .otp-box {{ background: white; border: 2px dashed #667eea; padding: 20px; 
                           text-align: center; margin: 20px 0; border-radius: 8px; }}
                .otp-code {{ font-size: 32px; font-weight: bold; color: #667eea; 
                            letter-spacing: 8px; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Visual ML!</h1>
                </div>
                <div class="content">
                    <p>Hi {name},</p>
                    <p>Thank you for signing up! Please verify your email address using the code below:</p>
                    
                    <div class="otp-box">
                        <p style="margin: 0; font-size: 14px; color: #666;">Your verification code</p>
                        <p class="otp-code">{otp}</p>
                    </div>
                    
                    <p><strong>This code will expire in 10 minutes.</strong></p>
                    <p>If you didn't create an account with Visual ML, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Visual ML. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        result = self._send_email(email, name, subject, html_content)
        if result:
            logger.info(f"✅ OTP email sent successfully to {email}")
        else:
            logger.error(f"❌ Failed to send OTP email to {email}")
        return result

    def send_welcome_email(self, email: str, name: str) -> bool:
        """
        Send welcome email after successful verification.
        
        Args:
            email: User email
            name: User name
            
        Returns:
            bool: True if sent successfully
        """
        logger.info(f"📧 Sending welcome email to {email}")
        
        subject = "Welcome to Visual ML! 🎉"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 40px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .feature {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; 
                           border-left: 4px solid #667eea; }}
                .cta-button {{ display: inline-block; background: #667eea; color: white; 
                              padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                              margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Welcome to Visual ML!</h1>
                </div>
                <div class="content">
                    <p>Hi {name},</p>
                    <p>Your account has been successfully verified! You're now ready to start building amazing ML pipelines.</p>
                    
                    <h3>What you can do with Visual ML:</h3>
                    
                    <div class="feature">
                        <strong>📊 Visual Pipeline Builder</strong><br>
                        Create ML pipelines with an intuitive drag-and-drop interface
                    </div>
                    
                    <div class="feature">
                        <strong>🤖 Multiple Algorithms</strong><br>
                        Choose from Linear Regression, Logistic Regression, and more
                    </div>
                    
                    <div class="feature">
                        <strong>💾 Save & Resume</strong><br>
                        Your projects are automatically saved and synced
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="http://localhost:5173/dashboard" class="cta-button">
                            Get Started Now
                        </a>
                    </div>
                    
                    <p>If you have any questions, feel free to reach out to our support team.</p>
                    <p>Happy building!</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Visual ML. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        result = self._send_email(email, name, subject, html_content)
        if result:
            logger.info(f"✅ Welcome email sent successfully to {email}")
        else:
            logger.error(f"❌ Failed to send welcome email to {email}")
        return result

    def send_experience_survey(self, email: str, name: str) -> bool:
        """
        Send experience survey email (10 days after signup).
        
        Args:
            email: User email
            name: User name
            
        Returns:
            bool: True if sent successfully
        """
        logger.info(f"📧 Sending experience survey to {email}")
        
        subject = "How's your Visual ML experience? 📝"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .cta-button {{ display: inline-block; background: #667eea; color: white; 
                              padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                              margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>We'd love your feedback!</h1>
                </div>
                <div class="content">
                    <p>Hi {name},</p>
                    <p>You've been using Visual ML for 10 days now! We hope you're enjoying the experience.</p>
                    
                    <p>We'd really appreciate it if you could take 2 minutes to share your thoughts with us. 
                    Your feedback helps us make Visual ML better for everyone.</p>
                    
                    <div style="text-align: center;">
                        <a href="https://forms.google.com/visualml-survey" class="cta-button">
                            Share Your Feedback
                        </a>
                    </div>
                    
                    <p>As a thank you, we'll give you <strong>1 month of Premium access</strong> for free! 🎁</p>
                    
                    <p>Thank you for being part of our community!</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Visual ML. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        result = self._send_email(email, name, subject, html_content)
        if result:
            logger.info(f"✅ Survey email sent successfully to {email}")
        else:
            logger.error(f"❌ Failed to send survey email to {email}")
        return result

    def send_marketing_email(
        self,
        email: str,
        name: str,
        subject: str,
        content: str
    ) -> bool:
        """
        Send marketing/promotional email.
        
        Args:
            email: User email
            name: User name
            subject: Email subject
            content: HTML email content
            
        Returns:
            bool: True if sent successfully
        """
        logger.info(f"📧 Sending marketing email to {email}")
        result = self._send_email(email, name, subject, content)
        if result:
            logger.info(f"✅ Marketing email sent successfully to {email}")
        else:
            logger.error(f"❌ Failed to send marketing email to {email}")
        return result


# Singleton instance
email_service = EmailService()
            to_name: Recipient name
            subject: Email subject
            html_content: HTML email content
            template_id: Optional Brevo template ID
            params: Optional template parameters
            
        Returns:
            bool: True if sent successfully
        """
        try:
            send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
                to=[{"email": to_email, "name": to_name}],
                sender={
                    "email": settings.BREVO_SENDER_EMAIL,
                    "name": settings.BREVO_SENDER_NAME
                },
                subject=subject,
                html_content=html_content,
            )

            if template_id:
                send_smtp_email.template_id = template_id
                send_smtp_email.params = params or {}

            response = self.api_instance.send_transac_email(send_smtp_email)
            logger.info(f"Email sent to {to_email}: {response.message_id}")
            return True

        except ApiException as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def send_verification_otp(self, email: str, name: str, otp: str) -> bool:
        """
        Send OTP verification email.
        
        Args:
            email: User email
            name: User name
            otp: 6-digit OTP code
            
        Returns:
            bool: True if sent successfully
        """
        subject = "Verify your Visual ML account"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .otp-box {{ background: white; border: 2px dashed #667eea; padding: 20px; 
                           text-align: center; margin: 20px 0; border-radius: 8px; }}
                .otp-code {{ font-size: 32px; font-weight: bold; color: #667eea; 
                            letter-spacing: 8px; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Visual ML!</h1>
                </div>
                <div class="content">
                    <p>Hi {name},</p>
                    <p>Thank you for signing up! Please verify your email address using the code below:</p>
                    
                    <div class="otp-box">
                        <p style="margin: 0; font-size: 14px; color: #666;">Your verification code</p>
                        <p class="otp-code">{otp}</p>
                    </div>
                    
                    <p><strong>This code will expire in 10 minutes.</strong></p>
                    <p>If you didn't create an account with Visual ML, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Visual ML. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self._send_email(email, name, subject, html_content)

    def send_welcome_email(self, email: str, name: str) -> bool:
        """
        Send welcome email after successful verification.
        
        Args:
            email: User email
            name: User name
            
        Returns:
            bool: True if sent successfully
        """
        subject = "Welcome to Visual ML! 🎉"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 40px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .feature {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; 
                           border-left: 4px solid #667eea; }}
                .cta-button {{ display: inline-block; background: #667eea; color: white; 
                              padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                              margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Welcome to Visual ML!</h1>
                </div>
                <div class="content">
                    <p>Hi {name},</p>
                    <p>Your account has been successfully verified! You're now ready to start building amazing ML pipelines.</p>
                    
                    <h3>What you can do with Visual ML:</h3>
                    
                    <div class="feature">
                        <strong>📊 Visual Pipeline Builder</strong><br>
                        Create ML pipelines with an intuitive drag-and-drop interface
                    </div>
                    
                    <div class="feature">
                        <strong>🤖 Multiple Algorithms</strong><br>
                        Choose from Linear Regression, Logistic Regression, and more
                    </div>
                    
                    <div class="feature">
                        <strong>💾 Save & Resume</strong><br>
                        Your projects are automatically saved and synced
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="http://localhost:5173/dashboard" class="cta-button">
                            Get Started Now
                        </a>
                    </div>
                    
                    <p>If you have any questions, feel free to reach out to our support team.</p>
                    <p>Happy building!</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Visual ML. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self._send_email(email, name, subject, html_content)

    def send_experience_survey(self, email: str, name: str) -> bool:
        """
        Send experience survey email (10 days after signup).
        
        Args:
            email: User email
            name: User name
            
        Returns:
            bool: True if sent successfully
        """
        subject = "How's your Visual ML experience? 📝"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .cta-button {{ display: inline-block; background: #667eea; color: white; 
                              padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                              margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>We'd love your feedback!</h1>
                </div>
                <div class="content">
                    <p>Hi {name},</p>
                    <p>You've been using Visual ML for 10 days now! We hope you're enjoying the experience.</p>
                    
                    <p>We'd really appreciate it if you could take 2 minutes to share your thoughts with us. 
                    Your feedback helps us make Visual ML better for everyone.</p>
                    
                    <div style="text-align: center;">
                        <a href="https://forms.google.com/visualml-survey" class="cta-button">
                            Share Your Feedback
                        </a>
                    </div>
                    
                    <p>As a thank you, we'll give you <strong>1 month of Premium access</strong> for free! 🎁</p>
                    
                    <p>Thank you for being part of our community!</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 Visual ML. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self._send_email(email, name, subject, html_content)

    def send_marketing_email(
        self,
        email: str,
        name: str,
        subject: str,
        content: str
    ) -> bool:
        """
        Send marketing/promotional email.
        
        Args:
            email: User email
            name: User name
            subject: Email subject
            content: HTML email content
            
        Returns:
            bool: True if sent successfully
        """
        return self._send_email(email, name, subject, content)


# Singleton instance
email_service = EmailService()
