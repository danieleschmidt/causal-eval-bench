"""
Notification services for evaluation events and alerts.
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


@dataclass
class NotificationEvent:
    """Notification event data."""
    
    event_type: str  # evaluation_completed, session_started, error_occurred, etc.
    title: str
    message: str
    severity: str = "info"  # info, warning, error, critical
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class NotificationService(ABC):
    """Abstract base class for notification services."""
    
    @abstractmethod
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send a notification."""
        pass
    
    @abstractmethod
    async def send_batch_notifications(self, events: List[NotificationEvent]) -> List[bool]:
        """Send multiple notifications."""
        pass


class EmailNotifier(NotificationService):
    """Email notification service using SMTP."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        use_tls: bool = True
    ):
        """Initialize email notifier."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("EMAIL_FROM", self.username)
        self.use_tls = use_tls
        
        # Default recipients by severity
        self.default_recipients = {
            "info": os.getenv("NOTIFICATION_EMAIL_INFO", "").split(","),
            "warning": os.getenv("NOTIFICATION_EMAIL_WARNING", "").split(","),
            "error": os.getenv("NOTIFICATION_EMAIL_ERROR", "").split(","),
            "critical": os.getenv("NOTIFICATION_EMAIL_CRITICAL", "").split(",")
        }
        
        # Clean empty emails
        for severity in self.default_recipients:
            self.default_recipients[severity] = [
                email.strip() for email in self.default_recipients[severity] 
                if email.strip()
            ]
    
    async def send_notification(
        self, 
        event: NotificationEvent,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Send email notification."""
        if not self.username or not self.password:
            logger.warning("Email credentials not configured")
            return False
        
        recipients = recipients or self.default_recipients.get(event.severity, [])
        if not recipients:
            logger.warning(f"No recipients configured for severity: {event.severity}")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[Causal Eval] {event.title}"
            
            # Create email body
            body = self._create_email_body(event)
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                text = msg.as_string()
                server.sendmail(self.from_email, recipients, text)
            
            logger.info(f"Email notification sent: {event.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    async def send_batch_notifications(self, events: List[NotificationEvent]) -> List[bool]:
        """Send multiple email notifications."""
        results = []
        for event in events:
            result = await self.send_notification(event)
            results.append(result)
            # Add small delay to avoid overwhelming SMTP server
            await asyncio.sleep(0.1)
        return results
    
    def _create_email_body(self, event: NotificationEvent) -> str:
        """Create HTML email body."""
        severity_colors = {
            "info": "#17a2b8",
            "warning": "#ffc107", 
            "error": "#dc3545",
            "critical": "#721c24"
        }
        
        color = severity_colors.get(event.severity, "#6c757d")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">{event.title}</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Severity: {event.severity.upper()}</p>
                </div>
                
                <div style="padding: 30px;">
                    <div style="margin-bottom: 20px;">
                        <h3 style="color: #343a40; margin-bottom: 10px;">Message</h3>
                        <p style="color: #6c757d; line-height: 1.6;">{event.message}</p>
                    </div>
                    
                    <div style="margin-bottom: 20px;">
                        <h3 style="color: #343a40; margin-bottom: 10px;">Event Details</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">Event Type:</td>
                                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{event.event_type}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">Timestamp:</td>
                                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{event.timestamp.isoformat()}</td>
                            </tr>
                        </table>
                    </div>
        """
        
        # Add metadata if present
        if event.metadata:
            html += """
                    <div style="margin-bottom: 20px;">
                        <h3 style="color: #343a40; margin-bottom: 10px;">Additional Information</h3>
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 12px; overflow-x: auto;">
            """
            
            for key, value in event.metadata.items():
                html += f"<strong>{key}:</strong> {str(value)}<br>"
            
            html += """
                        </div>
                    </div>
            """
        
        html += """
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6;">
                    <p style="margin: 0; color: #6c757d; font-size: 14px;">
                        This notification was sent by Causal Evaluation Benchmark
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


class SlackNotifier(NotificationService):
    """Slack notification service using webhooks."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Slack notifier."""
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
    
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send Slack notification."""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            # Create Slack message
            message = self._create_slack_message(event)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=message,
                    timeout=10.0
                )
                response.raise_for_status()
            
            logger.info(f"Slack notification sent: {event.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    async def send_batch_notifications(self, events: List[NotificationEvent]) -> List[bool]:
        """Send multiple Slack notifications."""
        results = []
        for event in events:
            result = await self.send_notification(event)
            results.append(result)
            # Add small delay to respect rate limits
            await asyncio.sleep(0.5)
        return results
    
    def _create_slack_message(self, event: NotificationEvent) -> Dict[str, Any]:
        """Create Slack message format."""
        severity_colors = {
            "info": "#36a64f",
            "warning": "#ff9000",
            "error": "#ff0000", 
            "critical": "#800000"
        }
        
        severity_emojis = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:"
        }
        
        color = severity_colors.get(event.severity, "#808080")
        emoji = severity_emojis.get(event.severity, ":bell:")
        
        # Create fields for metadata
        fields = [
            {
                "title": "Event Type",
                "value": event.event_type,
                "short": True
            },
            {
                "title": "Severity",
                "value": event.severity.upper(),
                "short": True
            },
            {
                "title": "Timestamp",
                "value": event.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "short": False
            }
        ]
        
        # Add metadata fields
        if event.metadata:
            for key, value in event.metadata.items():
                if len(fields) < 10:  # Slack limit
                    fields.append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True
                    })
        
        return {
            "text": f"{emoji} Causal Eval Notification",
            "attachments": [
                {
                    "color": color,
                    "title": event.title,
                    "text": event.message,
                    "fields": fields,
                    "footer": "Causal Evaluation Benchmark",
                    "ts": int(event.timestamp.timestamp())
                }
            ]
        }


class WebhookNotifier(NotificationService):
    """Generic webhook notification service."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """Initialize webhook notifier."""
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
    
    async def send_notification(self, event: NotificationEvent) -> bool:
        """Send webhook notification."""
        try:
            payload = {
                "event_type": event.event_type,
                "title": event.title,
                "message": event.message,
                "severity": event.severity,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=10.0
                )
                response.raise_for_status()
            
            logger.info(f"Webhook notification sent: {event.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    async def send_batch_notifications(self, events: List[NotificationEvent]) -> List[bool]:
        """Send multiple webhook notifications."""
        results = []
        for event in events:
            result = await self.send_notification(event)
            results.append(result)
        return results


class NotificationManager:
    """Manager for multiple notification services."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.services: List[NotificationService] = []
    
    def add_service(self, service: NotificationService):
        """Add a notification service."""
        self.services.append(service)
    
    async def send_notification(self, event: NotificationEvent) -> List[bool]:
        """Send notification to all configured services."""
        if not self.services:
            logger.warning("No notification services configured")
            return []
        
        tasks = [service.send_notification(event) for service in self.services]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to False
        return [result if isinstance(result, bool) else False for result in results]
    
    async def send_evaluation_completed(
        self,
        model_name: str,
        session_id: str,
        overall_score: float,
        total_evaluations: int,
        duration_minutes: float
    ):
        """Send evaluation completed notification."""
        event = NotificationEvent(
            event_type="evaluation_completed",
            title=f"Evaluation Completed: {model_name}",
            message=f"Model {model_name} completed evaluation with overall score {overall_score:.2%}",
            severity="info",
            metadata={
                "model_name": model_name,
                "session_id": session_id,
                "overall_score": overall_score,
                "total_evaluations": total_evaluations,
                "duration_minutes": duration_minutes
            }
        )
        
        return await self.send_notification(event)
    
    async def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Send error alert notification."""
        event = NotificationEvent(
            event_type="error_occurred",
            title=f"Error: {error_type}",
            message=error_message,
            severity="error",
            metadata=context or {}
        )
        
        return await self.send_notification(event)
    
    async def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Send performance alert notification."""
        severity = "critical" if current_value > threshold * 1.5 else "warning"
        
        event = NotificationEvent(
            event_type="performance_alert",
            title=f"Performance Alert: {metric_name}",
            message=f"{metric_name} is {current_value:.2f}, above threshold of {threshold:.2f}",
            severity=severity,
            metadata={
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold,
                **(context or {})
            }
        )
        
        return await self.send_notification(event)


# Factory functions
def create_email_notifier() -> Optional[EmailNotifier]:
    """Create email notifier from environment variables."""
    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        return None
    
    return EmailNotifier(
        smtp_host=smtp_host,
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    )


def create_slack_notifier() -> Optional[SlackNotifier]:
    """Create Slack notifier from environment variables."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return None
    
    return SlackNotifier(webhook_url)


def create_notification_manager() -> NotificationManager:
    """Create notification manager with all configured services."""
    manager = NotificationManager()
    
    # Add email notifier if configured
    email_notifier = create_email_notifier()
    if email_notifier:
        manager.add_service(email_notifier)
    
    # Add Slack notifier if configured
    slack_notifier = create_slack_notifier()
    if slack_notifier:
        manager.add_service(slack_notifier)
    
    # Add webhook notifier if configured
    webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")
    if webhook_url:
        webhook_notifier = WebhookNotifier(webhook_url)
        manager.add_service(webhook_notifier)
    
    return manager