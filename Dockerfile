# Use an official Python runtime as a parent image
# syntax=docker/dockerfile:1.4
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed for some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install Python dependencies with a persistent pip cache to speed rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Create a non-root user for security purposes
RUN useradd --create-home appuser

# Copy the application source (owned by appuser) once
COPY --chown=appuser:appuser ./src /home/appuser/app/src

# Switch to non-root user and working directory
USER appuser
WORKDIR /home/appuser/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application
# We run uvicorn and point it to the app instance in our main.py file
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
