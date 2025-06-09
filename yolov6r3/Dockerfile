FROM python:3.10-slim

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
    gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /yolov6r3

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Environment variable
ENV RUNTIME=prod

# Application entry point
CMD ["python3", "/yolov6r3/main.py"]
