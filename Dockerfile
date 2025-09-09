# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs chroma_db

# Set permissions
RUN chmod +x scripts/run.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]

# docker-compose.yml
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    container_name: topic_clustering_mysql
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
      MYSQL_USER: ${MYSQL_USER}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      timeout: 20s
      retries: 10

  app:
    build: .
    container_name: topic_clustering_app
    depends_on:
      mysql:
        condition: service_healthy
    ports:
      - "8000:8000"
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_USER=${MYSQL_USER}
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_DATABASE=${MYSQL_DATABASE}
      - CHROMA_PERSIST_DIR=/app/chroma_db
      - LOG_FILE=/app/logs/app.log
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: topic_clustering_nginx
    depends_on:
      - app
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    restart: unless-stopped

volumes:
  mysql_data:

# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name localhost;
        
        client_max_body_size 10M;
        
        # Rate limiting for API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS headers
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
            add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        }
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # Static files caching
        location /static/ {
            proxy_pass http://app;
            expires 1d;
            add_header Cache-Control "public, immutable";
        }
        
        # Health check endpoint
        location /health {
            proxy_pass http://app;
            access_log off;
        }
    }
}

# scripts/run.sh
#!/bin/bash

# Production deployment script for Multilingual Topic Clustering System

set -e  # Exit on any error

echo "🚀 Starting Multilingual Topic Clustering System deployment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    print_warning "Environment file not found. Creating from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your configuration:"
    echo "  - Set MYSQL_PASSWORD to a secure password"
    echo "  - Adjust other settings as needed"
    echo ""
    read -p "Do you want to edit .env now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} .env
    else
        print_warning "Remember to edit .env before running again."
        exit 1
    fi
fi

print_success "Environment configuration found"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p chroma_db logs ssl

# Set proper permissions
chmod 755 chroma_db logs
if [ -d ssl ]; then
    chmod 700 ssl
fi

print_success "Directories created"

# Check if services are already running
if docker-compose ps | grep -q "Up"; then
    print_warning "Some services are already running"
    read -p "Do you want to restart them? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping existing services..."
        docker-compose down
    else
        print_status "Using existing services"
    fi
fi

# Build and start services
print_status "Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 10

# Check MySQL health
print_status "Checking MySQL connection..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker-compose exec -T mysql mysqladmin ping -h localhost --silent; then
        print_success "MySQL is ready"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "MySQL failed to start after $max_attempts attempts"
        print_status "Checking logs..."
        docker-compose logs mysql
        exit 1
    fi
    
    print_status "Waiting for MySQL... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

# Check application health
print_status "Checking application health..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -s -f http://localhost:8000/health > /dev/null; then
        print_success "Application is ready"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Application failed to start after $max_attempts attempts"
        print_status "Checking logs..."
        docker-compose logs app
        exit 1
    fi
    
    print_status "Waiting for application... (attempt $attempt/$max_attempts)"
    sleep 3
    ((attempt++))
done

# Final status check
print_status "Checking all services..."
if docker-compose ps | grep -q "Up"; then
    print_success "All services are running!"
    echo ""
    echo "🌐 Application URLs:"
    echo "  • Main Interface:    http://localhost:8000"
    echo "  • Interactive Demo:  http://localhost:8000/demo"
    echo "  • Analytics:         http://localhost:8000/analytics"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo ""
    echo "📊 System Status:"
    echo "  • Health Check:      http://localhost:8000/health"
    echo "  • System Stats:      http://localhost:8000/api/stats"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs:         docker-compose logs -f"
    echo "  • Stop services:     docker-compose down"
    echo "  • Restart services:  docker-compose restart"
    echo ""
    print_success "Deployment completed successfully!"
else
    print_error "Some services failed to start properly"
    print_status "Service status:"
    docker-compose ps
    echo