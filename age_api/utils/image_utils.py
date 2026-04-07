import cv2
import numpy as np

def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes uploaded image bytes into a numpy array (BGR format for OpenCV).
    Returns None if decoding fails.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
