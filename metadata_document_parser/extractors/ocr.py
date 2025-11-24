from abc import ABC, abstractmethod
from typing import Optional
import requests
import json
import base64

class ExternalOCR(ABC):
    """
    Abstract base class for external OCR services.
    """
    @abstractmethod
    def image_to_latex(self, image_bytes: bytes) -> Optional[str]:
        """
        Convert image bytes to LaTeX string.
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            LaTeX string if successful, None otherwise
        """
        pass

class MathpixOCR(ExternalOCR):
    """
    Mathpix OCR implementation.
    """
    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
        self.api_url = "https://api.mathpix.com/v3/text"

    def image_to_latex(self, image_bytes: bytes) -> Optional[str]:
        """
        Convert image to LaTeX using Mathpix API.
        """
        try:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-type": "application/json"
            }
            
            payload = {
                "src": f"data:image/png;base64,{image_base64}",
                "formats": ["latex_simplified"],
                "data_options": {
                    "include_latex": True
                }
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "latex_simplified" in result:
                    return result["latex_simplified"]
                elif "text" in result:
                    # Fallback if latex_simplified is not present but text is (sometimes happens)
                    return result["text"]
            else:
                print(f"Mathpix API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error calling Mathpix API: {e}")
            
        return None
