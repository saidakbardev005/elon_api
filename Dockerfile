FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any required)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
# && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Default command to run the app
CMD ["python", "app.py"]
