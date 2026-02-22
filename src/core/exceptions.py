"""Custom exception types used across the project."""


class SPOfflineRLError(Exception):
    """Base exception for the project."""


class RegistryError(SPOfflineRLError):
    """Raised when a registry lookup fails."""


class DataValidationError(SPOfflineRLError):
    """Raised when offline dataset schema validation fails."""


class ConfigurationError(SPOfflineRLError):
    """Raised when configuration is invalid or incomplete."""
