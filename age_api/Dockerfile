# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables for optimizations:
# 1. PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# 2. PYTHONUNBUFFERED: Ensures console output is not buffered by Docker (useful for logs)
# 3. DEBIAN_FRONTEND: Suppress apt-get interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for certain C/C++ libraries that OpenCV headless relies on)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to explicitly leverage Docker layer caching
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies without saving cache
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Security best practice: Create a non-root user and set permissions
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Specify the default command to start the FastAPI server via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
