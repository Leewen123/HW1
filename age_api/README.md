# Lightweight Age Prediction API

This is a simple MLOps pipeline template for serving a face recognition (age prediction) model. It uses **FastAPI** for a fast, modern Python API server and **OpenCV** for lightweight image processing.

## Project Structure

```text
age_api/
├── main.py                     # FastAPI application and endpoints
├── requirements.txt            # Python dependencies
├── models/                     # ML models wrapper directory
│   └── age_predictor.py        # Model loading and inference logic
└── utils/                      # Helper utilities
    └── image_utils.py          # Image decoding functions
```

## Setup Instructions

1. **Create a virtual environment (optional but recommended):**
   ```bash
   cd age_api
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will now be running at `http://localhost:8000`. You can visit `http://localhost:8000/docs` to see the auto-generated Swagger UI.

## Usage Example

Use `curl` to upload an image and get a prediction:

```bash
curl -X POST "http://localhost:8000/predict/age" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_your_image.jpg"
```

### Expected Response Format
```json
{
  "filename": "path_to_your_image.jpg",
  "prediction": {
    "age_range": "(25-32)",
    "confidence": 0.88,
    "face_box": {
      "x": 105,
      "y": 140,
      "w": 300,
      "h": 300
    }
  },
  "meta": {
    "latency_sec": 0.0345
  }
}
```

## Extensibility for Production
The current `AgePredictor` uses an OpenCV face detection Haar cascade and provides a dummy/mock setup for the age prediction step to keep things lightweight out of the box. 

**To use a real model:**
1. Drop your trained `.onnx` or `.caffemodel` into the `models/` folder.
2. Uncomment the `cv2.dnn.readNetFromONNX()` logic in `models/age_predictor.py` and adjust the prediction parsing as needed for your specific model outputs.
