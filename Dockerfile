FROM python:3.13.9-slim

WORKDIR /app

# ENV LOAD_SEED_DATA=true

# Create a directory for persistent data
RUN mkdir -p /app/data

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application (assuming main.py is the entry point)
# We use uvicorn for serving the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
