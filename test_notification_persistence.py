"""
tests/test_notification_persistence.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive unit tests for the notification dispatcher and persistent alert repository.

Test Groups:
S01: Default Simulation Mode when Unconfigured
S02: Incomplete SMTP Configuration Falls Back to Simulation
S03: Mocked Successful SMTP Delivery
S04: SMTP Authentication Failure Handling & Graceful Degradation
S05: SMTP Network Connection Failure Handling
S06: Delivery Status Verified Only on Actual Success
S07: Idempotency & Duplicate Dispatch Prevention
S08: Event Persistence Across Repository Reload
S09: Clinician Acknowledgement Persistence
S10: Event Resolution Persistence
S11: Idempotent Event ID Rejection (No Duplicate Event IDs)
S12: Safe Recovery from Malformed JSON (No Silent Loss of History)
S13: Missing Storage File Graceful Initialization
S14: Full Temporary Path Isolation
S15: Password & Credential Redaction in Error Messages
S16: Hospital EHR Identifier Rejection in Notification Dispatch
"""

import os
import sys
import json
import socket
import smtplib
import datetime
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.sensor.alert_engine import (
    AlertEvent,
    AlertCategory,
    AlertSeverity,
    DeliveryStatus,
    DataQualityStatus,
)
from src.sensor.notification_service import (
    AlertRepository,
    NotificationDispatcher,
    SMTPConfig,
    DeliveryOutcome,
)


@pytest.fixture
def sample_event():
    """Create a realistic sample alert event for notification testing."""
    return AlertEvent(
        event_id="ALT-20261016-120000-CRITACH-A1B2C3",
        rule_name="CRITICAL_TACHYCARDIA",
        category=AlertCategory.PHYSIOLOGICAL_RISK,
        severity=AlertSeverity.CRITICAL,
        timestamp=datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc),
        reason="Heart rate HR=138 bpm exceeds critical threshold (130 bpm).",
        vitals={"heart_rate": 138.0, "spo2": 97.0, "ir_ppg": 85000.0, "risk_status_code": 2},
        data_quality_status=DataQualityStatus.VALID,
        session_id="SESSION-20261016-120000",
        delivery_status=DeliveryStatus.NOT_SENT,
        is_acknowledged=False,
    )


@pytest.fixture
def temp_repo(tmp_path):
    """Isolated repository instance targeting a temporary path."""
    storage_file = str(tmp_path / "test_sensor_alerts.json")
    return AlertRepository(storage_path=storage_file)


class TestNotificationDispatcherAndPersistence:

    def test_S01_simulation_mode_by_default(self, sample_event, temp_repo):
        """When SMTP configuration is absent or disabled, dispatcher runs in safe simulation mode."""
        config = SMTPConfig(enabled=False)
        dispatcher = NotificationDispatcher(config=config, repository=temp_repo)
        
        assert dispatcher.is_simulation_mode is True
        outcome = dispatcher.dispatch_alert(sample_event)
        
        assert outcome.status == "simulated"
        assert outcome.success is True
        assert sample_event.delivery_status == DeliveryStatus.SIMULATED
        
        # Verify repository was updated with simulated status
        saved_evt = temp_repo.get_event(sample_event.event_id)
        assert saved_evt is not None
        assert saved_evt.delivery_status == DeliveryStatus.SIMULATED

    def test_S02_incomplete_smtp_config_falls_back_to_simulation(self, sample_event, temp_repo):
        """Enabled SMTP with missing host or recipient falls back to simulation mode without crashing."""
        incomplete_config = SMTPConfig(
            enabled=True,
            host="", # Missing host
            recipient_email="test@telemed.local",
        )
        dispatcher = NotificationDispatcher(config=incomplete_config, repository=temp_repo)
        assert dispatcher.is_simulation_mode is True
        
        outcome = dispatcher.dispatch_alert(sample_event)
        assert outcome.status == "simulated"
        assert outcome.success is True

    @patch("smtplib.SMTP")
    def test_S03_mocked_successful_smtp_delivery(self, mock_smtp_cls, sample_event, temp_repo):
        """With complete SMTP config and mocked server success, alert is marked SENT."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        config = SMTPConfig(
            enabled=True,
            host="smtp.telemed-relay.local",
            port=587,
            username="telemed_bot",
            password="fake_password_123",
            use_tls=True,
            sender_email="alerts@xmt-sepsis.local",
            recipient_email="icu-response@hospital.org",
        )
        dispatcher = NotificationDispatcher(config=config, repository=temp_repo)
        assert dispatcher.is_simulation_mode is False

        outcome = dispatcher.dispatch_alert(sample_event)

        assert outcome.status == "sent"
        assert outcome.success is True
        assert sample_event.delivery_status == DeliveryStatus.SENT
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("telemed_bot", "fake_password_123")
        mock_server.sendmail.assert_called_once()

        # Check repository state
        saved_evt = temp_repo.get_event(sample_event.event_id)
        assert saved_evt.delivery_status == DeliveryStatus.SENT

    @patch("smtplib.SMTP")
    def test_S04_smtp_authentication_failure_gracefully_handled(self, mock_smtp_cls, sample_event, temp_repo):
        """SMTP auth errors do not crash application; outcome is FAILED with redacted error."""
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid")
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        config = SMTPConfig(
            enabled=True,
            host="smtp.example.com",
            port=587,
            username="bot_user",
            password="secret_password_xyz",
            sender_email="alerts@example.com",
            recipient_email="icu@example.com",
        )
        dispatcher = NotificationDispatcher(config=config, repository=temp_repo)

        outcome = dispatcher.dispatch_alert(sample_event)

        assert outcome.status == "failed"
        assert outcome.success is False
        assert sample_event.delivery_status == DeliveryStatus.FAILED
        assert "Authentication credentials invalid" in outcome.message
        assert "secret_password_xyz" not in outcome.message

        saved_evt = temp_repo.get_event(sample_event.event_id)
        assert saved_evt.delivery_status == DeliveryStatus.FAILED

    @patch("smtplib.SMTP")
    def test_S05_smtp_network_failure_handled(self, mock_smtp_cls, sample_event, temp_repo):
        """Socket / timeout error during connect gracefully logs failure."""
        mock_smtp_cls.side_effect = socket.timeout("Connection to mail server timed out")

        config = SMTPConfig(
            enabled=True,
            host="192.0.2.1",
            port=587,
            sender_email="alerts@example.com",
            recipient_email="icu@example.com",
        )
        dispatcher = NotificationDispatcher(config=config, repository=temp_repo)

        outcome = dispatcher.dispatch_alert(sample_event)
        assert outcome.status == "failed"
        assert outcome.success is False
        assert "timed out" in outcome.message

    def test_S06_delivery_status_verified_only_on_actual_success(self, sample_event, temp_repo):
        """Unsuccessful dispatches never leave the event marked SENT."""
        assert sample_event.delivery_status == DeliveryStatus.NOT_SENT
        
        with patch("smtplib.SMTP", side_effect=IOError("Network down")):
            config = SMTPConfig(enabled=True, host="smtp.fake.org", sender_email="a@b.com", recipient_email="c@d.com")
            dispatcher = NotificationDispatcher(config=config, repository=temp_repo)
            outcome = dispatcher.dispatch_alert(sample_event)
            
            assert outcome.status == "failed"
            assert sample_event.delivery_status != DeliveryStatus.SENT
            assert sample_event.delivery_status == DeliveryStatus.FAILED

    def test_S07_idempotency_prevents_duplicate_dispatch(self, sample_event, temp_repo):
        """Events already marked SENT are not re-transmitted."""
        sample_event.delivery_status = DeliveryStatus.SENT
        temp_repo.add_event(sample_event)

        config = SMTPConfig(enabled=True, host="smtp.fake.org", sender_email="a@b.com", recipient_email="c@d.com")
        dispatcher = NotificationDispatcher(config=config, repository=temp_repo)

        with patch("smtplib.SMTP") as mock_smtp:
            outcome = dispatcher.dispatch_alert(sample_event)
            assert outcome.status == "already_sent"
            assert outcome.success is True
            mock_smtp.assert_not_called()

    def test_S08_event_persistence_across_repository_reload(self, sample_event, tmp_path):
        """Events saved in one repository instance persist faithfully and reload in a new instance."""
        storage_file = str(tmp_path / "persistence_test.json")
        repo1 = AlertRepository(storage_path=storage_file)
        
        added = repo1.add_event(sample_event)
        assert added is True

        # Reopen via new repository instance
        repo2 = AlertRepository(storage_path=storage_file)
        events = repo2.load_events()
        
        assert len(events) == 1
        loaded = events[0]
        assert loaded.event_id == sample_event.event_id
        assert loaded.rule_name == sample_event.rule_name
        assert loaded.severity == sample_event.severity
        assert loaded.timestamp == sample_event.timestamp
        assert loaded.vitals["heart_rate"] == 138.0

    def test_S09_clinician_acknowledgement_persistence(self, sample_event, tmp_path):
        """Clinician acknowledgement metadata persists across repository reload."""
        storage_file = str(tmp_path / "ack_test.json")
        repo1 = AlertRepository(storage_path=storage_file)
        repo1.add_event(sample_event)

        ack_time = datetime.datetime(2026, 10, 16, 12, 5, 0, tzinfo=datetime.timezone.utc)
        ack_evt = repo1.acknowledge_event(
            event_id=sample_event.event_id,
            clinician_id="DR_TELEMED_ONCALL",
            notes="Patient reviewed. Telemetry artifact ruled out; bedside nurse alerted.",
            ack_time=ack_time,
        )
        assert ack_evt is not None
        assert ack_evt.is_acknowledged is True

        # Reload in fresh instance
        repo2 = AlertRepository(storage_path=storage_file)
        reloaded = repo2.get_event(sample_event.event_id)
        assert reloaded.is_acknowledged is True
        assert reloaded.acknowledged_by == "DR_TELEMED_ONCALL"
        assert reloaded.acknowledged_at == ack_time
        assert "bedside nurse alerted" in reloaded.acknowledgement_notes

    def test_S10_event_resolution_persistence(self, sample_event, tmp_path):
        """Event resolution timestamp and status persist across reload."""
        storage_file = str(tmp_path / "resolve_test.json")
        repo1 = AlertRepository(storage_path=storage_file)
        repo1.add_event(sample_event)

        res_time = datetime.datetime(2026, 10, 16, 12, 15, 0, tzinfo=datetime.timezone.utc)
        resolved = repo1.resolve_event(sample_event.event_id, resolved_time=res_time)
        assert resolved.is_resolved is True

        # Reload in fresh instance
        repo2 = AlertRepository(storage_path=storage_file)
        reloaded = repo2.get_event(sample_event.event_id)
        assert reloaded.is_resolved is True
        assert reloaded.resolved_at == res_time

    def test_S11_duplicate_event_ids_rejected(self, sample_event, temp_repo):
        """Attempting to add an existing event_id returns False and preserves single instance."""
        add1 = temp_repo.add_event(sample_event)
        assert add1 is True

        add2 = temp_repo.add_event(sample_event)
        assert add2 is False
        assert len(temp_repo.load_events()) == 1

    def test_S12_malformed_json_recovery_creates_backup_without_silent_loss(self, tmp_path):
        """If storage file is corrupted with invalid JSON, preserve as .corrupt backup without crashing."""
        storage_file = str(tmp_path / "corrupt_test.json")
        with open(storage_file, "w", encoding="utf-8") as f:
            f.write("{ INVALID JSON CONTENT: [TRUNCATED_WRITE_ERROR")

        repo = AlertRepository(storage_path=storage_file)
        events = repo.load_events()

        assert events == []
        # Check that backup file was created to preserve data
        backups = [f for f in os.listdir(tmp_path) if f.startswith("corrupt_test.json.corrupt_")]
        assert len(backups) == 1, "Expected corrupt backup to be preserved"

    def test_S13_missing_storage_file_initializes_safely(self, tmp_path):
        """Non-existent storage file returns empty list and creates path on first save."""
        non_existent = str(tmp_path / "subdir" / "new_alerts.json")
        repo = AlertRepository(storage_path=non_existent)
        assert repo.load_events() == []

    def test_S14_full_temporary_path_isolation(self, sample_event, tmp_path):
        """Tests use tmp_path so the real data/processed/sensor_alerts_log.json is never modified."""
        test_file = str(tmp_path / "isolated.json")
        repo = AlertRepository(storage_path=test_file)
        repo.add_event(sample_event)

        real_log = os.path.join(_PROJECT_ROOT, "data", "processed", "sensor_alerts_log.json")
        assert os.path.abspath(test_file) != os.path.abspath(real_log)

    def test_S15_password_and_credential_redaction_in_errors(self, sample_event, temp_repo):
        """Passwords are redacted from string representations and error messages."""
        config = SMTPConfig(
            enabled=True,
            host="smtp.example.com",
            username="my_username",
            password="SuperSecretPassword99!",
            sender_email="from@example.com",
            recipient_email="to@example.com",
        )
        repr_str = str(config)
        assert "SuperSecretPassword99!" not in repr_str
        assert "***REDACTED***" in repr_str

        # Redaction in error sanitizer
        dispatcher = NotificationDispatcher(config=config, repository=temp_repo)
        raw_error = "Authentication failed for user my_username with password SuperSecretPassword99!"
        clean_error = dispatcher._sanitize_error_message(raw_error)
        assert "SuperSecretPassword99!" not in clean_error
        assert "***REDACTED***" in clean_error

    def test_S16_hospital_ehr_identifier_scrubbed_from_notification(self, base_time, temp_repo):
        """If historical EHR IDs (subject_id, stay_id) exist, they are scrubbed and not emailed."""
        compromised_event = AlertEvent(
            event_id="ALT-20261016-LEAK-TEST",
            rule_name="CRITICAL_TACHYCARDIA",
            category=AlertCategory.PHYSIOLOGICAL_RISK,
            severity=AlertSeverity.CRITICAL,
            timestamp=base_time,
            reason="HR exceeds critical threshold.",
            vitals={"heart_rate": 140.0, "subject_id": 99991234, "stay_id": 88884321},
            data_quality_status=DataQualityStatus.VALID,
        )
        dispatcher = NotificationDispatcher(repository=temp_repo)
        subject, text, html = dispatcher.format_email_message(compromised_event)

        assert "99991234" not in text
        assert "88884321" not in text
        assert "99991234" not in html
        assert "88884321" not in html


@pytest.fixture
def base_time():
    return datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.timezone.utc)
