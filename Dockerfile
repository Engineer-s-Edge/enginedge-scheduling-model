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

# Make port 3009 available to the world outside this container (default, can be overridden via PORT env var)
EXPOSE 3009

# Define the command to run the application
# We run uvicorn and point it to the app instance in our main.py file
# --host 0.0.0.0 makes the server accessible from outside the container
# Use PORT env var if set, otherwise default to 3009
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-3009}"]
