# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables to avoid python generating .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Disable OneDNN opts to prevent TensorFlow/PyTorch crashing on some generic CPUs
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies (ffmpeg is required by Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We install PyTorch CPU first to keep the Docker image lightweight. 
# (If deploying to a GPU cluster, use the CUDA wheel instead)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download the required spaCy model for Auto-NER and preprocessing
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_trf

# Pre-download NLTK tokenizers used by newspaper3k
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the entire project directory into the container
COPY . .

# Expose port 8000 for the FastAPI web server
EXPOSE 8000

# Set the default command to run the web dashboard
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000"]
