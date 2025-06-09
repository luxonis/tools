FROM python:3.10-slim

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic system tools
    curl gnupg ca-certificates \
    # OpenCV dependencies
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    # Node
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
    gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /yolov7

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Runtime configuration
ENV RUNTIME=prod

# Entry point
CMD ["python3", "/yolov7/main.py"]
