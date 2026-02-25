#!/bin/bash
set -e

# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "🚀 Starting DashCord via Docker Compose..."

# This builds the image if needed and starts the container in the background
docker compose up -d --build

echo "✅ DashCord is running!"
echo "📜 Type './logs.sh' to see the bot activity."