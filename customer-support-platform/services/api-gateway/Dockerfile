# API Gateway Service Dockerfile

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy shared modules
COPY shared/ /app/shared/

# Copy service code
COPY services/api-gateway/ /app/services/api-gateway/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
CMD ["python", "-m", "uvicorn", "services.api-gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]