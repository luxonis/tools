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
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up frontend (React/Vite/etc.)
WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm install
COPY client/public ./public
COPY client/src ./src
RUN npm run build

# Copy backend code (after npm build to leverage Docker layer cache)
WORKDIR /app
COPY . .

# Runtime environment
ENV RUNTIME=prod

# Run the Python app
CMD ["python3", "/app/main.py"]
