# Use Python 3.8 as base image to match DeepFace's recommended version
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY deepface_service.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=deepface_service.py

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application using gunicorn
CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "deepface_service:app"]