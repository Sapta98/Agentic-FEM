#!/bin/bash
# Quick deployment script for Agentic FEM
# Usage: ./deploy.sh [production|staging]

set -e

ENVIRONMENT=${1:-production}
echo "🚀 Deploying Agentic FEM in $ENVIRONMENT mode..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Run: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed."
    exit 1
fi

# Check for .env file
if [ ! -f "nlp_parser/src/.env" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: No .env file found and OPENAI_API_KEY not set."
    echo "   Please create nlp_parser/src/.env with your OPENAI_API_KEY"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stop existing containers
echo "📦 Stopping existing containers..."
docker compose down 2>/dev/null || true

# Build the image
echo "🔨 Building Docker image..."
docker compose build

# Start the application
echo "🚀 Starting application..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker compose up -d
else
    docker compose up
fi

# Wait for health check
echo "⏳ Waiting for application to start..."
sleep 5

# Check if application is running
if curl -f http://localhost:8080/health &> /dev/null; then
    echo "✅ Application is running!"
    echo ""
    echo "📍 Access your application at:"
    echo "   http://localhost:8080"
    echo "   http://localhost:8080/docs (API documentation)"
    echo ""
    echo "📊 Check status with: docker compose ps"
    echo "📋 View logs with: docker compose logs -f"
else
    echo "⚠️  Application may still be starting. Check logs with:"
    echo "   docker compose logs -f agentic-fem"
fi

