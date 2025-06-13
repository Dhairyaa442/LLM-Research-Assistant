FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directory for FAISS index
RUN mkdir -p data/faiss_index

# Expose ports
EXPOSE 8000 8501

# Default: run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
