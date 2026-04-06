# Use lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing pyc files & enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (important for some ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories (safe for runtime)
RUN mkdir -p logs artifacts

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "frontend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]