# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy our requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our application code into the container
COPY src/ src/
COPY data/ data/

# Define the default command to run when the container starts
CMD ["python", "src/inference.py"]