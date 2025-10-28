# Dockerfile - build a tiny container image for the Streamlit app
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Create app directory
WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirements.txt /app/requirements.txt

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]