class MediaKitException(Exception):
    """Base exception class for MediaKit AI application."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ProviderError(MediaKitException):
    """Raised when an AI provider fails to generate an image."""
    def __init__(self, provider_name: str, message: str):
        self.provider_name = provider_name
        self.message = f"Provider '{provider_name}' error: {message}"
        super().__init__(self.message)


class ConfigurationError(MediaKitException):
    """Raised when application configuration is invalid or missing."""
    pass
