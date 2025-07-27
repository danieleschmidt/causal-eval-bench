#!/bin/bash

# Post-start script for dev container
set -e

echo "🚀 Starting development services..."

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 5

# Check if PostgreSQL is ready
if command -v pg_isready &> /dev/null; then
    echo "🗄️  Checking PostgreSQL connection..."
    until pg_isready -h postgres -p 5432 -U causal_eval; do
        echo "Waiting for PostgreSQL..."
        sleep 2
    done
    echo "✅ PostgreSQL is ready!"
fi

# Check if Redis is ready
if command -v redis-cli &> /dev/null; then
    echo "🟥 Checking Redis connection..."
    until redis-cli -h redis ping | grep PONG; do
        echo "Waiting for Redis..."
        sleep 2
    done
    echo "✅ Redis is ready!"
fi

# Start development monitoring (optional)
if [ "$ENABLE_DEV_MONITORING" = "true" ]; then
    echo "📊 Starting development monitoring..."
    # Add any monitoring setup here
fi

# Display service status
echo ""
echo "🌍 Development environment status:"
echo "  • FastAPI server: http://localhost:8000"
echo "  • Documentation: http://localhost:8080"
echo "  • Grafana: http://localhost:3000 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • PostgreSQL: localhost:5432"
echo "  • Redis: localhost:6379"
echo ""
echo "✅ Development environment is ready!"
