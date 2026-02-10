#!/bin/bash

# Visual-ML Docker Deployment Script
# Usage: ./deploy.sh [dev|prod]

set -e

ENVIRONMENT=${1:-dev}

echo "🚀 Deploying Visual-ML in $ENVIRONMENT mode..."

if [ "$ENVIRONMENT" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    
    # Check if .env exists
    if [ ! -f .env ]; then
        echo "❌ Error: .env file not found!"
        echo "📝 Please create .env file from .env.docker.example"
        echo "   cp .env.docker.example .env"
        echo "   nano .env"
        exit 1
    fi
    
    echo "📦 Building production images..."
    docker-compose -f $COMPOSE_FILE build
    
    echo "🔄 Starting production services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
    
    echo "🗄️  Running database migrations..."
    docker-compose -f $COMPOSE_FILE exec -T backend alembic upgrade head
    
else
    COMPOSE_FILE="docker-compose.yml"
    
    echo "🔄 Starting development services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Service Status:"
docker-compose -f $COMPOSE_FILE ps
echo ""

if [ "$ENVIRONMENT" = "prod" ]; then
    echo "🌐 Access your application:"
    echo "   Frontend: http://localhost"
    echo "   Backend API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
else
    echo "🌐 Access your application:"
    echo "   Frontend: http://localhost:5173"
    echo "   Backend API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
fi

echo ""
echo "📝 View logs:"
echo "   docker-compose -f $COMPOSE_FILE logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose -f $COMPOSE_FILE down"
