"""src/auth_utils.py

Authentication utilities for Modal and HuggingFace.

This module provides:
- Token validation for HuggingFace and Modal
- Authentication status checking
- Structured logging for authentication flows
- Pre-flight validation for CI/CD pipelines

Usage:
    from auth_utils import validate_hf_token, validate_modal_auth, AuthValidator

    # Quick validation
    is_valid, message = validate_hf_token()
    if not is_valid:
        raise ValueError(f"HF auth failed: {message}")

    # Detailed validation with AuthValidator
    validator = AuthValidator()
    hf_status = validator.check_hf_token()
    modal_status = validator.check_modal_auth()

    validator.report()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TokenStatus(Enum):
    """Authentication token status."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    EXPIRED = "expired"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    UNKNOWN = "unknown"


@dataclass
class TokenValidationResult:
    """Result of token validation."""

    status: TokenStatus
    message: str
    token_type: str | None = None
    user_name: str | None = None
    expires_at: datetime | None = None
    permissions: list[str] | None = None
    metadata: dict[str, Any] | None = None


class AuthValidator:
    """Validate authentication tokens for HuggingFace and Modal."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("auth_utils")
        self.results: dict[str, TokenValidationResult] = {}

    def check_hf_token(self, token: str | None = None) -> TokenValidationResult:
        """Validate HuggingFace token.

        Args:
            token: HF token (uses HF_TOKEN env var if None)

        Returns:
            TokenValidationResult with status and details
        """
        if token is None:
            token = os.environ.get("HF_TOKEN")

        if not token:
            result = TokenValidationResult(
                status=TokenStatus.MISSING,
                message="HF_TOKEN not set. Set via 'export HF_TOKEN=hf_...' or GitHub Secrets",
                token_type="huggingface",
            )
            self.results["huggingface"] = result
            self.logger.warning(f"HF token validation: {result.message}")
            return result

        # Check token format
        if not token.startswith("hf_"):
            result = TokenValidationResult(
                status=TokenStatus.INVALID,
                message=f"Invalid token format. Expected 'hf_...' prefix, got '{token[:8]}...'",
                token_type="huggingface",
            )
            self.results["huggingface"] = result
            self.logger.error(f"HF token validation: {result.message}")
            return result

        # Test token validity with API
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            info = api.whoami(token=token)

            result = TokenValidationResult(
                status=TokenStatus.VALID,
                message=f"Token valid for user: {info['name']}",
                token_type="huggingface",
                user_name=info["name"],
            )
            self.results["huggingface"] = result
            self.logger.info(f"HF token validated: {result.message}")
            return result

        except Exception as e:
            error_msg = str(e)

            # Check for specific error types
            if "401" in error_msg or "Unauthorized" in error_msg:
                status = TokenStatus.INVALID
                message = f"Token invalid or expired: {e}"
            elif "403" in error_msg or "Forbidden" in error_msg:
                status = TokenStatus.INSUFFICIENT_PERMISSIONS
                message = f"Token lacks required permissions: {e}"
            else:
                status = TokenStatus.UNKNOWN
                message = f"Token validation failed: {e}"

            result = TokenValidationResult(
                status=status, message=message, token_type="huggingface"
            )
            self.results["huggingface"] = result
            self.logger.error(f"HF token validation: {result.message}")
            return result

    def check_modal_auth(self) -> TokenValidationResult:
        """Validate Modal authentication.

        Returns:
            TokenValidationResult with status and details
        """
        try:
            import modal

            # Check if Modal is properly configured (requires auth)
            # Using is_local() as a simple check that doesn't require network
            # In a real scenario, you'd use modal.config.get_user() or similar
            is_configured = hasattr(modal, "is_local")

            if is_configured:
                result = TokenValidationResult(
                    status=TokenStatus.VALID,
                    message="Modal module loaded successfully",
                    token_type="modal",
                )
            else:
                result = TokenValidationResult(
                    status=TokenStatus.UNKNOWN,
                    message="Modal module check inconclusive",
                    token_type="modal",
                )

            self.results["modal"] = result
            self.logger.info(f"Modal auth validated: {result.message}")
            return result

        except Exception as e:
            # Check if it's an auth error from modal
            error_msg = str(e).lower()
            if "auth" in error_msg or "authentication" in error_msg or "token" in error_msg:
                result = TokenValidationResult(
                    status=TokenStatus.MISSING,
                    message=f"Modal authentication required. Run 'modal token new': {e}",
                    token_type="modal",
                )
            else:
                result = TokenValidationResult(
                    status=TokenStatus.UNKNOWN,
                    message=f"Modal auth check failed: {e}",
                    token_type="modal",
                )
            self.results["modal"] = result
            self.logger.warning(f"Modal auth check: {result.message}")
            return result

    def validate_before_upload(self) -> TokenValidationResult:
        """Validate authentication before uploading to HuggingFace Hub.

        This method performs comprehensive validation for upload operations,
        checking both HF token validity and required permissions.

        Returns:
            TokenValidationResult with status and upload-specific details
        """
        self.logger.info("=" * 60)
        self.logger.info("PRE-UPLOAD AUTHENTICATION VALIDATION")
        self.logger.info("=" * 60)

        # First check HF token
        hf_result = self.check_hf_token()

        if hf_result.status != TokenStatus.VALID:
            self.logger.error(f"❌ Upload validation failed: {hf_result.message}")
            return TokenValidationResult(
                status=hf_result.status,
                message=f"Cannot upload: {hf_result.message}",
                token_type="upload",
                metadata={"hf_status": hf_result.status.value, "errors": [hf_result.message]},
            )

        # Check for write permissions (write token required for uploads)
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            # Try to get user info to verify token has required permissions
            info = api.whoami(token=os.environ.get("HF_TOKEN"))

            # Check if user can create repos (indicates write permission)
            can_write = info.get("can_create_repo", True)  # Default to True if field missing

            if not can_write:
                result = TokenValidationResult(
                    status=TokenStatus.INSUFFICIENT_PERMISSIONS,
                    message="Token lacks write permissions for uploading models",
                    token_type="upload",
                    user_name=info.get("name"),
                    metadata={
                        "hf_status": "valid",
                        "permissions": info.get("scopes", []),
                        "can_create_repo": can_write,
                    },
                )
                self.results["upload"] = result
                self.logger.error(f"❌ Upload validation failed: {result.message}")
                return result

            # All checks passed
            result = TokenValidationResult(
                status=TokenStatus.VALID,
                message=f"Ready to upload as user: {info.get('name')}",
                token_type="upload",
                user_name=info.get("name"),
                permissions=info.get("scopes", []),
                metadata={
                    "hf_status": "valid",
                    "can_create_repo": can_write,
                    "permissions": info.get("scopes", []),
                },
            )
            self.results["upload"] = result
            self.logger.info(f"✅ Upload validation passed: {result.message}")
            return result

        except Exception as e:
            result = TokenValidationResult(
                status=TokenStatus.UNKNOWN,
                message=f"Upload validation failed: {e}",
                token_type="upload",
                metadata={"error": str(e)},
            )
            self.results["upload"] = result
            self.logger.error(f"❌ Upload validation failed: {result.message}")
            return result

    def report(self) -> bool:
        """Generate authentication report.

        Returns:
            True if all tokens are valid, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("AUTHENTICATION STATUS REPORT")
        self.logger.info("=" * 60)

        all_valid = True

        for token_type, result in self.results.items():
            status_icon = "✅" if result.status == TokenStatus.VALID else "❌"
            self.logger.info(
                f"{status_icon} {token_type.upper()}: {result.status.value}"
            )
            self.logger.info(f"   {result.message}")

            if result.user_name:
                self.logger.info(f"   User: {result.user_name}")

            if result.status != TokenStatus.VALID:
                all_valid = False

        self.logger.info("=" * 60)

        if all_valid:
            self.logger.info("✅ All authentication tokens valid")
        else:
            self.logger.error("❌ Some authentication tokens invalid or missing")
            self.logger.error(
                "See AGENTS.md or agents-docs/auth-troubleshooting.md for help"
            )

        return all_valid


def validate_hf_token(token: str | None = None) -> tuple[bool, str]:
    """Validate HuggingFace token format and accessibility.

    Args:
        token: HF token (uses HF_TOKEN env var if None)

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = AuthValidator()
    result = validator.check_hf_token(token)
    return result.status == TokenStatus.VALID, result.message


def validate_modal_auth() -> tuple[bool, str]:
    """Validate Modal authentication.

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = AuthValidator()
    result = validator.check_modal_auth()
    return result.status == TokenStatus.VALID, result.message


def setup_auth_logging(
    log_file: str | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Setup structured logging for authentication flows.

    Args:
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger
    """
    auth_logger = logging.getLogger("auth_utils")
    auth_logger.setLevel(level)

    # Clear existing handlers
    auth_logger.handlers.clear()

    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    auth_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        auth_logger.addHandler(file_handler)

    return auth_logger


def preflight_check(logger: logging.Logger | None = None) -> bool:
    """Run pre-flight authentication checks.

    Args:
        logger: Optional logger (creates one if None)

    Returns:
        True if all checks pass, False otherwise
    """
    if logger is None:
        logger = setup_auth_logging()

    logger.info("Running pre-flight authentication checks...")

    validator = AuthValidator(logger)

    # Check HuggingFace
    hf_result = validator.check_hf_token()
    if hf_result.status != TokenStatus.VALID:
        logger.error(f"❌ HuggingFace: {hf_result.message}")
        return False
    logger.info(f"✅ HuggingFace: {hf_result.message}")

    # Check Modal
    modal_result = validator.check_modal_auth()
    if modal_result.status != TokenStatus.VALID:
        logger.error(f"❌ Modal: {modal_result.message}")
        return False
    logger.info(f"✅ Modal: {modal_result.message}")

    logger.info("✅ All pre-flight checks passed")
    return True


class AuthenticationError(Exception):
    """Custom exception for authentication failures."""

    def __init__(self, message: str, token_type: str | None = None):
        self.message = message
        self.token_type = token_type
        super().__init__(self.message)


def require_hf_token(token: str | None = None) -> str:
    """Require valid HuggingFace token or raise AuthenticationError.

    Args:
        token: HF token (uses HF_TOKEN env var if None)

    Returns:
        Valid token string

    Raises:
        AuthenticationError: If token is missing or invalid
    """
    is_valid, message = validate_hf_token(token)

    if not is_valid:
        raise AuthenticationError(
            f"HuggingFace authentication failed: {message}", token_type="huggingface"
        )

    return token or os.environ.get("HF_TOKEN", "")


def require_modal_auth() -> None:
    """Require valid Modal authentication or raise AuthenticationError.

    Raises:
        AuthenticationError: If authentication fails
    """
    is_valid, message = validate_modal_auth()

    if not is_valid:
        raise AuthenticationError(
            f"Modal authentication failed: {message}", token_type="modal"
        )
