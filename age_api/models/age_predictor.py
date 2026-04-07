import os
import cv2
import numpy as np

class AgePredictor:
    """
    A lightweight age predictor model class for MLOps pipelines.
    This class loads its dependencies heavily on instantiation to ensure
    predict calls are fast and stateless.
    """
    def __init__(self):
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load a built-in OpenCV Haar Cascade for fast, lightweight face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Define age classes common in age prediction datasets
        self.age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # NOTE: For a real deep learning model, you would load your ONNX or Caffe model here.
        # e.g., self.age_net = cv2.dnn.readNetFromONNX(os.path.join(self.model_dir, "age_model.onnx"))
        self.age_net = None 

    def predict_age(self, image: np.ndarray) -> dict:
        """
        Detects a face in the image and predicts their age.
        """
        if image is None:
            return {"error": "Invalid image provided."}
            
        # 1. Detect Face
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {"error": "No face detected in the image."}
            
        # Process the largest face found
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        face_crop = image[y:y+h, x:x+w]
        
        # 2. Predict Age
        # Since we want a functional "mock" that is lightweight without requiring users
        # to manually download weights, we provide a placeholder prediction logic.
        # IF you have a real loaded model:
        # blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
        # self.age_net.setInput(blob)
        # age_preds = self.age_net.forward()
        # predicted_age = self.age_classes[age_preds[0].argmax()]
        # confidence = float(age_preds[0].max())
        
        # --- Mock logic ---
        # We simulate a "prediction" based on simple face heuristics to ensure the API works end-to-end
        idx = (x + y + w + h) % len(self.age_classes)
        predicted_age = self.age_classes[idx]
        confidence = 0.85 + (idx % 10) / 100.0
        # ------------------
        
        return {
            "age_range": predicted_age,
            "confidence": round(confidence, 4),
            "face_box": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        }
