from typing import Dict, Type
from app.providers.base import ImageProvider

class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Type[ImageProvider]] = {}

    def register(self, name: str, provider_cls: Type[ImageProvider]):
        """
        Register a new provider class.
        """
        self._providers[name.lower()] = provider_cls

    def get(self, name: str) -> Type[ImageProvider]:
        """
        Retrieve a registered provider class.
        """
        name_lower = name.lower()
        if name_lower not in self._providers:
            raise KeyError(
                f"Provider '{name}' is not registered. "
                f"Available providers: {list(self._providers.keys())}"
            )
        return self._providers[name_lower]

# Singleton registry instance
provider_registry = ProviderRegistry()

# Import and register providers to populate the registry upon module loading
from app.providers.ollama.provider import OllamaProvider
from app.providers.diffusers.provider import DiffusersProvider

provider_registry.register("ollama", OllamaProvider)
provider_registry.register("diffusers", DiffusersProvider)
