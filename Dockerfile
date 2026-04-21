# Use Python 3.11 slim image
FROM python:3.11-slim

# Install Node.js for building the frontend
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package.json and install frontend dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy frontend source and build
COPY src ./src
COPY index.html vite.config.js ./
RUN npm run build

# Copy backend requirements and install
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY backend ./backend

# Expose the port that Cloud Run will use
EXPOSE 8080

# Run the application
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]