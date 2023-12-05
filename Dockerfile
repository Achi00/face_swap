# Use an official Python runtime as a base image
FROM python:3.9 as builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Multi-stage build: Use a smaller base image for the final image
FROM python:3.9-slim

WORKDIR /app

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy the rest of your application's code
COPY . /app

# Define environment variable for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google.json

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["gunicorn", "-b", ":5000", "-t", "120", "swap_faces:flask_app"]
