import base64
import httpx
from app.providers.base import ImageProvider
from app.core.exceptions import ProviderError

class OllamaProvider(ImageProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_image(self, prompt: str, **kwargs) -> bytes:
        """
        Generate image by calling the local Ollama API.
        This assumes Ollama is configured with an image generation model or adapter.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "width": int(kwargs.get("width") or 512),
                "height": int(kwargs.get("height") or 512),
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                
                if response.status_code != 200:
                    raise ProviderError(
                        "ollama",
                        f"Ollama server returned status code {response.status_code}: {response.text}"
                    )
                
                try:
                    result = response.json()
                except ValueError:
                    raise ProviderError("ollama", "Failed to parse JSON response from Ollama server.")
                
                # Check for standard response structures. Custom image generation setups
                # often return the image encoded in base64 under key 'image' or 'response'.
                image_data = result.get("image") or result.get("response")
                
                if not image_data:
                    raise ProviderError(
                        "ollama",
                        f"Ollama response did not contain 'image' or 'response' field. Response received: {result}"
                    )
                
                try:
                    # Clean up base64 URI prefixes if present
                    if isinstance(image_data, str):
                        if "," in image_data:
                            image_data = image_data.split(",", 1)[1]
                        return base64.b64decode(image_data.strip())
                    elif isinstance(image_data, bytes):
                        return image_data
                    else:
                        raise ProviderError("ollama", "Unsupported image data format returned by Ollama.")
                except Exception as e:
                    raise ProviderError("ollama", f"Failed to decode base64 image data: {str(e)}")
                    
        except httpx.RequestError as e:
            raise ProviderError("ollama", f"Connection to Ollama server failed: {str(e)}")
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
            raise ProviderError("ollama", f"Unexpected error occurred: {str(e)}")
