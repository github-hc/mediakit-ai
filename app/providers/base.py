from abc import ABC, abstractmethod

class ImageProvider(ABC):
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> bytes:
        """
        Generate an image from a prompt.

        Args:
            prompt: Text description of the image to generate.
            **kwargs: Provider-specific configuration overrides (e.g., width, height, quality, seed).

        Returns:
            bytes: Raw binary image data (e.g., PNG or JPEG).
        """
        pass
