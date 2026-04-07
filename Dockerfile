FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]