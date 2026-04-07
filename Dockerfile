# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Prevent buffering
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Run inference script
CMD ["uvicorn", "customer_support_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]