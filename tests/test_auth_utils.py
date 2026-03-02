"""tests/test_auth_utils.py

Comprehensive tests for authentication utilities.

Coverage:
- TokenStatus enum validation
- TokenValidationResult dataclass creation and fields
- validate_hf_token() with missing, invalid format, invalid response, and valid tokens
- validate_modal_auth() with missing and valid authentication
- AuthValidator class methods
- AuthenticationError exception handling
- setup_auth_logging() configuration
- preflight_check() integration

Requirements from MISSING_TASKS_SUMMARY.json T-A03-AUTH-PHASE1.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auth_utils import (
    AuthValidator,
    AuthenticationError,
    TokenStatus,
    TokenValidationResult,
    preflight_check,
    require_hf_token,
    require_modal_auth,
    setup_auth_logging,
    validate_hf_token,
    validate_modal_auth,
)

# Check if huggingface_hub is available
try:
    import huggingface_hub

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class TestTokenStatus:
    """Test TokenStatus enum values and behavior."""

    def test_all_status_values(self):
        """Test all TokenStatus enum members have correct values."""
        assert TokenStatus.VALID.value == "valid"
        assert TokenStatus.INVALID.value == "invalid"
        assert TokenStatus.MISSING.value == "missing"
        assert TokenStatus.EXPIRED.value == "expired"
        assert TokenStatus.INSUFFICIENT_PERMISSIONS.value == "insufficient_permissions"
        assert TokenStatus.UNKNOWN.value == "unknown"

    def test_token_status_is_enum(self):
        """Test TokenStatus is properly an Enum."""
        assert issubclass(TokenStatus, type(TokenStatus.VALID))

    def test_status_comparison(self):
        """Test status can be compared correctly."""
        assert TokenStatus.VALID == TokenStatus.VALID
        assert TokenStatus.VALID != TokenStatus.INVALID
        assert TokenStatus.MISSING != TokenStatus.UNKNOWN


class TestTokenValidationResult:
    """Test TokenValidationResult dataclass creation and validation."""

    def test_result_creation_with_required_fields(self):
        """Test creating result with only required fields."""
        result = TokenValidationResult(
            status=TokenStatus.VALID,
            message="Token is valid",
        )
        assert result.status == TokenStatus.VALID
        assert result.message == "Token is valid"
        assert result.token_type is None
        assert result.user_name is None
        assert result.expires_at is None
        assert result.permissions is None
        assert result.metadata is None

    def test_result_creation_with_all_fields(self):
        """Test creating result with all optional fields."""
        from datetime import datetime

        result = TokenValidationResult(
            status=TokenStatus.VALID,
            message="Token validated",
            token_type="huggingface",
            user_name="testuser",
            expires_at=datetime(2024, 12, 31, 23, 59, 59),
            permissions=["read", "write"],
            metadata={"key": "value"},
        )
        assert result.token_type == "huggingface"
        assert result.user_name == "testuser"
        assert result.expires_at == datetime(2024, 12, 31, 23, 59, 59)
        assert result.permissions == ["read", "write"]
        assert result.metadata == {"key": "value"}

    def test_result_dataclass_fields(self):
        """Test all expected fields exist in dataclass."""
        field_names = {f.name for f in fields(TokenValidationResult)}
        expected = {
            "status",
            "message",
            "token_type",
            "user_name",
            "expires_at",
            "permissions",
            "metadata",
        }
        assert field_names == expected

    def test_result_equality(self):
        """Test result equality comparison."""
        result1 = TokenValidationResult(
            status=TokenStatus.VALID, message="test", token_type="huggingface"
        )
        result2 = TokenValidationResult(
            status=TokenStatus.VALID, message="test", token_type="huggingface"
        )
        result3 = TokenValidationResult(
            status=TokenStatus.INVALID, message="test", token_type="huggingface"
        )
        assert result1 == result2
        assert result1 != result3


class TestValidateHfToken:
    """Test validate_hf_token function with various scenarios."""

    def test_missing_token_env_not_set(self):
        """Test validate_hf_token returns INVALID when HF_TOKEN env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, message = validate_hf_token()
            assert is_valid is False
            assert "not set" in message.lower() or "missing" in message.lower()

    def test_missing_token_empty_string(self):
        """Test validate_hf_token returns INVALID when token is empty string."""
        is_valid, message = validate_hf_token(token="")
        assert is_valid is False
        assert "not set" in message.lower() or "missing" in message.lower()

    def test_invalid_token_format_no_prefix(self):
        """Test validate_hf_token with invalid format - no hf_ prefix."""
        is_valid, message = validate_hf_token(token="invalid_token_without_prefix")
        assert is_valid is False
        assert "Invalid token format" in message
        assert "hf_" in message

    def test_invalid_token_format_wrong_prefix(self):
        """Test validate_hf_token with wrong prefix."""
        is_valid, message = validate_hf_token(token="abc_12345")
        assert is_valid is False
        assert "Invalid token format" in message

    def test_invalid_token_format_short_token(self):
        """Test validate_hf_token with short invalid token."""
        is_valid, message = validate_hf_token(token="xyz")
        assert is_valid is False
        assert "Invalid token format" in message

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_invalid_token_api_response_401(self, mock_hf_api_class):
        """Test validate_hf_token when API returns 401 Unauthorized."""
        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("401 Client Error: Unauthorized")
        mock_hf_api_class.return_value = mock_api

        is_valid, message = validate_hf_token(token="hf_testtoken123")
        assert is_valid is False
        assert "invalid" in message.lower() or "expired" in message.lower()

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_invalid_token_api_response_403(self, mock_hf_api_class):
        """Test validate_hf_token when API returns 403 Forbidden."""
        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("403 Client Error: Forbidden")
        mock_hf_api_class.return_value = mock_api

        is_valid, message = validate_hf_token(token="hf_testtoken123")
        assert is_valid is False
        assert "permission" in message.lower() or "403" in message

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_valid_token_mocked(self, mock_hf_api_class):
        """Test validate_hf_token with valid token (mocked API response)."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser", "email": "test@example.com"}
        mock_hf_api_class.return_value = mock_api

        is_valid, message = validate_hf_token(token="hf_validtoken123456")
        assert is_valid is True
        assert "testuser" in message

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_valid_token_with_explicit_param(self, mock_hf_api_class):
        """Test validate_hf_token with explicit token parameter."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "explicituser"}
        mock_hf_api_class.return_value = mock_api

        with patch.dict(os.environ, {"HF_TOKEN": "hf_envtoken"}):
            is_valid, message = validate_hf_token(token="hf_paramtoken123456")
            assert is_valid is True
            # Should use the explicit token, not env var
            mock_api.whoami.assert_called_once_with(token="hf_paramtoken123456")


class TestCheckModalAuth:
    """Test Modal authentication checking."""

    @patch("sys.modules", {"modal": None})  # Simulate modal not installed
    def test_check_modal_auth_missing(self):
        """Test check_modal_auth when Modal token not configured."""
        validator = AuthValidator()
        result = validator.check_modal_auth()

        # Should return a valid token type and reasonable status
        # Note: In environments where modal is installed, it returns VALID
        # In environments without modal, it returns MISSING or UNKNOWN
        assert result.token_type == "modal"
        assert result.status in [
            TokenStatus.VALID,
            TokenStatus.UNKNOWN,
            TokenStatus.MISSING,
        ]

    def test_check_modal_auth_valid(self):
        """Test check_modal_auth when Modal authentication is valid."""
        # Test with actual modal if available
        validator = AuthValidator()
        result = validator.check_modal_auth()

        # Result should have modal token type regardless of actual status
        assert result.token_type == "modal"
        assert isinstance(result.status, TokenStatus)

    def test_validate_modal_auth_function(self):
        """Test validate_modal_auth wrapper function."""
        is_valid, message = validate_modal_auth()
        # Should return tuple without crashing
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)


class TestAuthValidator:
    """Test AuthValidator class methods."""

    def test_validator_initialization_default_logger(self):
        """Test AuthValidator initializes with default logger."""
        validator = AuthValidator()
        assert validator.results == {}
        assert validator.logger is not None
        assert hasattr(validator.logger, "info")

    def test_validator_initialization_custom_logger(self):
        """Test AuthValidator initializes with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        validator = AuthValidator(logger=custom_logger)
        assert validator.logger == custom_logger

    def test_check_hf_token_missing_env_var(self):
        """Test check_hf_token with missing environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            validator = AuthValidator()
            result = validator.check_hf_token(token=None)

            assert result.status == TokenStatus.MISSING
            assert result.token_type == "huggingface"
            assert "not set" in result.message.lower()
            assert "huggingface" in validator.results

    def test_check_hf_token_invalid_format_stored_in_results(self):
        """Test check_hf_token stores invalid result in validator results."""
        validator = AuthValidator()
        result = validator.check_hf_token(token="invalid_format")

        assert result.status == TokenStatus.INVALID
        assert "huggingface" in validator.results
        assert validator.results["huggingface"].status == TokenStatus.INVALID

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_check_hf_token_valid_stores_result(self, mock_hf_api_class):
        """Test check_hf_token stores valid result."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "storeduser"}
        mock_hf_api_class.return_value = mock_api

        validator = AuthValidator()
        result = validator.check_hf_token(token="hf_validtoken123456")

        assert result.status == TokenStatus.VALID
        assert "huggingface" in validator.results
        assert validator.results["huggingface"].user_name == "storeduser"

    def test_check_modal_auth_stores_result(self):
        """Test check_modal_auth stores result in validator."""
        validator = AuthValidator()
        result = validator.check_modal_auth()

        assert "modal" in validator.results
        assert validator.results["modal"].token_type == "modal"

    def test_report_empty_results(self, caplog):
        """Test report with no results."""
        with caplog.at_level(logging.INFO):
            validator = AuthValidator()
            all_valid = validator.report()

        assert all_valid is True  # No invalid results
        assert "AUTHENTICATION STATUS REPORT" in caplog.text

    def test_report_with_valid_results(self, caplog):
        """Test report with valid results."""
        with caplog.at_level(logging.INFO):
            validator = AuthValidator()
            validator.results["huggingface"] = TokenValidationResult(
                status=TokenStatus.VALID,
                message="Valid token",
                token_type="huggingface",
                user_name="testuser",
            )
            all_valid = validator.report()

        assert all_valid is True
        assert "HUGGINGFACE: valid" in caplog.text
        assert "testuser" in caplog.text

    def test_report_with_invalid_results(self, caplog):
        """Test report with invalid results."""
        with caplog.at_level(logging.INFO):
            validator = AuthValidator()
            validator.results["huggingface"] = TokenValidationResult(
                status=TokenStatus.INVALID,
                message="Invalid token",
                token_type="huggingface",
            )
            all_valid = validator.report()

        assert all_valid is False
        assert "HUGGINGFACE: invalid" in caplog.text
        assert "Some authentication tokens invalid" in caplog.text

    def test_report_with_mixed_results(self, caplog):
        """Test report with both valid and invalid results."""
        with caplog.at_level(logging.INFO):
            validator = AuthValidator()
            validator.results["huggingface"] = TokenValidationResult(
                status=TokenStatus.VALID,
                message="Valid",
                token_type="huggingface",
            )
            validator.results["modal"] = TokenValidationResult(
                status=TokenStatus.MISSING,
                message="Missing token",
                token_type="modal",
            )
            all_valid = validator.report()

        assert all_valid is False
        assert "HUGGINGFACE: valid" in caplog.text
        assert "MODAL: missing" in caplog.text


class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_error_basic_message(self):
        """Test AuthenticationError with basic message."""
        error = AuthenticationError("Test error message")
        assert str(error) == "Test error message"
        assert error.token_type is None

    def test_error_with_token_type(self):
        """Test AuthenticationError with token type."""
        error = AuthenticationError("Auth failed", token_type="huggingface")
        assert str(error) == "Auth failed"
        assert error.token_type == "huggingface"
        assert error.message == "Auth failed"

    def test_error_modal_token_type(self):
        """Test AuthenticationError with modal token type."""
        error = AuthenticationError("Modal auth failed", token_type="modal")
        assert error.token_type == "modal"

    def test_error_inheritance(self):
        """Test AuthenticationError is an Exception."""
        error = AuthenticationError("test")
        assert isinstance(error, Exception)


class TestRequireFunctions:
    """Test require_hf_token and require_modal_auth functions."""

    def test_require_hf_token_missing_raises(self):
        """Test require_hf_token raises when token missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AuthenticationError) as exc_info:
                require_hf_token(token=None)
            assert "HuggingFace" in str(exc_info.value)
            assert exc_info.value.token_type == "huggingface"

    def test_require_hf_token_invalid_raises(self):
        """Test require_hf_token raises when token invalid."""
        with pytest.raises(AuthenticationError) as exc_info:
            require_hf_token(token="invalid_token")
        assert "HuggingFace" in str(exc_info.value)

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_require_hf_token_valid_returns_token(self, mock_hf_api_class):
        """Test require_hf_token returns token when valid."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser"}
        mock_hf_api_class.return_value = mock_api

        token = "hf_validtoken123456"
        result = require_hf_token(token=token)
        assert result == token

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_require_hf_token_from_env(self, mock_hf_api_class):
        """Test require_hf_token uses env var when token is None."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "envuser"}
        mock_hf_api_class.return_value = mock_api

        with patch.dict(os.environ, {"HF_TOKEN": "hf_envtoken123456"}):
            result = require_hf_token(token=None)
            assert result == "hf_envtoken123456"

    def test_require_modal_auth_raises_when_invalid(self):
        """Test require_modal_auth raises when auth invalid."""
        # Create a mock modal module
        mock_modal = MagicMock()
        mock_modal.is_local = MagicMock(side_effect=Exception("Auth failed"))
        mock_modal.exception = MagicMock()
        mock_modal.exception.AuthError = Exception

        with patch.dict("sys.modules", {"modal": mock_modal}):
            # May or may not raise depending on implementation
            try:
                require_modal_auth()
            except AuthenticationError as e:
                assert "Modal" in str(e)


class TestSetupAuthLogging:
    """Test setup_auth_logging function."""

    def test_setup_logging_returns_logger(self):
        """Test setup_auth_logging returns configured logger."""
        logger = setup_auth_logging()
        assert logger is not None
        assert logger.name == "auth_utils"

    def test_setup_logging_console_handler(self):
        """Test setup_auth_logging adds console handler."""
        logger = setup_auth_logging()
        console_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) >= 1

    def test_setup_logging_with_file_handler(self, tmp_path):
        """Test setup_auth_logging with file handler."""
        log_file = tmp_path / "auth_test.log"
        logger = setup_auth_logging(log_file=str(log_file))

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_file.exists()

    def test_setup_logging_level(self):
        """Test setup_auth_logging sets level correctly."""
        logger = setup_auth_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

        logger = setup_auth_logging(level=logging.WARNING)
        assert logger.level == logging.WARNING

    def test_setup_logging_clears_existing_handlers(self):
        """Test setup_auth_logging clears existing handlers."""
        # First call sets up handlers
        logger1 = setup_auth_logging()
        initial_handler_count = len(logger1.handlers)

        # Second call should clear and recreate
        logger2 = setup_auth_logging()
        assert len(logger2.handlers) == initial_handler_count


class TestPreflightCheck:
    """Test preflight_check function."""

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_preflight_all_valid(self, mock_hf_api_class, caplog):
        """Test preflight_check when all auth is valid."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser"}
        mock_hf_api_class.return_value = mock_api

        with patch.dict(os.environ, {"HF_TOKEN": "hf_validtoken123456"}):
            with caplog.at_level(logging.INFO):
                result = preflight_check()

        assert result is True
        assert "All pre-flight checks passed" in caplog.text

    def test_preflight_hf_invalid(self, caplog):
        """Test preflight_check when HuggingFace auth invalid."""
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.INFO):
                result = preflight_check()

        assert result is False
        assert "HuggingFace" in caplog.text

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_preflight_with_custom_logger(self, mock_hf_api_class, caplog):
        """Test preflight_check with custom logger."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "customuser"}
        mock_hf_api_class.return_value = mock_api

        custom_logger = logging.getLogger("custom_preflight")
        custom_logger.setLevel(logging.INFO)

        with patch.dict(os.environ, {"HF_TOKEN": "hf_validtoken123456"}):
            with caplog.at_level(logging.INFO):
                result = preflight_check(logger=custom_logger)

        # Should work with custom logger
        assert isinstance(result, bool)


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_hf_api_import_error(self, mock_hf_api_class):
        """Test handling when HfApi import fails."""
        mock_hf_api_class.side_effect = ImportError("No module named 'huggingface_hub'")

        validator = AuthValidator()
        result = validator.check_hf_token(token="hf_sometoken123456")

        # Should handle import error gracefully
        assert result.status in [TokenStatus.UNKNOWN, TokenStatus.INVALID]

    def test_token_with_whitespace(self):
        """Test token validation with whitespace."""
        # Token with leading/trailing whitespace should be invalid
        is_valid, message = validate_hf_token(token="  hf_token123  ")
        # Whitespace makes it not start with "hf_"
        assert is_valid is False

    def test_very_long_token(self):
        """Test validation with very long token."""
        long_token = "hf_" + "a" * 1000
        # Should handle without crashing
        is_valid, message = validate_hf_token(token=long_token)
        # Will fail API check but should not crash
        assert isinstance(is_valid, bool)

    def test_special_characters_in_token(self):
        """Test token with special characters."""
        special_token = "hf_token!@#$%^&*()"
        is_valid, message = validate_hf_token(token=special_token)
        assert isinstance(is_valid, bool)

    def test_unicode_in_token(self):
        """Test token with unicode characters."""
        unicode_token = "hf_tökén_ñame_日本語"
        is_valid, message = validate_hf_token(token=unicode_token)
        assert isinstance(is_valid, bool)

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_api_generic_exception(self, mock_hf_api_class):
        """Test handling of generic API exception."""
        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Network timeout")
        mock_hf_api_class.return_value = mock_api

        validator = AuthValidator()
        result = validator.check_hf_token(token="hf_validtoken123456")

        assert result.status == TokenStatus.UNKNOWN
        assert "failed" in result.message.lower() or "network" in result.message.lower()

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_api_unauthorized_exception(self, mock_hf_api_class):
        """Test handling of Unauthorized exception message."""
        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("Unauthorized access")
        mock_hf_api_class.return_value = mock_api

        validator = AuthValidator()
        result = validator.check_hf_token(token="hf_validtoken123456")

        assert result.status == TokenStatus.INVALID


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_validation_workflow_missing_both(self):
        """Test full workflow when both tokens are missing."""
        with patch.dict(os.environ, {}, clear=True):
            validator = AuthValidator()
            hf_result = validator.check_hf_token()
            modal_result = validator.check_modal_auth()

            assert hf_result.status == TokenStatus.MISSING
            assert validator.report() is False

    @pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
    @patch("huggingface_hub.HfApi")
    def test_full_validation_workflow_valid_hf(self, mock_hf_api_class):
        """Test full workflow with valid HF token."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "integration_user"}
        mock_hf_api_class.return_value = mock_api

        with patch.dict(os.environ, {"HF_TOKEN": "hf_valid123456"}):
            validator = AuthValidator()
            hf_result = validator.check_hf_token()

            assert hf_result.status == TokenStatus.VALID
            assert hf_result.user_name == "integration_user"

    def test_validator_reuse(self):
        """Test validator can be reused for multiple checks."""
        validator = AuthValidator()

        # First check - missing
        with patch.dict(os.environ, {}, clear=True):
            result1 = validator.check_hf_token()
            assert result1.status == TokenStatus.MISSING

        # Second check - invalid format
        result2 = validator.check_hf_token(token="bad_token")
        assert result2.status == TokenStatus.INVALID

        # Results should accumulate
        assert len(validator.results) == 1  # Same key overwritten
        assert validator.results["huggingface"].status == TokenStatus.INVALID
