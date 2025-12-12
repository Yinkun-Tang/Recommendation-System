#!/bin/bash
set -e

WIN_CHROME="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

echo "Cleaning old processes..."
fuser -k 8000/tcp || true
fuser -k 5173/tcp || true

BACKEND_URL="http://localhost:8000/docs"

echo "[1/6] Checking Python venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Activating venv..."
source venv/bin/activate

echo "[2/6] Installing Python dependencies..."
pip install -r requirements.txt

echo "[3/6] Installing frontend dependencies..."
if [ ! -d "frontend/node_modules" ]; then
    cd frontend && npm install && cd ..
else
    echo "node_modules exists, skip npm install."
fi

echo "[4/6] Starting backend server..."
uvicorn backend.fast_api.main:app --reload &
BACKEND_PID=$!

echo "Waiting for backend to be ready..."
until curl -s $BACKEND_URL > /dev/null; do
    sleep 1
done
echo "Backend is ready."

echo "[5/6] Starting frontend server..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

echo "[6/6] Opening browser..."
"$WIN_CHROME" "http://localhost:5173" &

echo "All services are running."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

wait $BACKEND_PID
wait $FRONTEND_PID

cleanup() {
    echo "Cleaning up..."
    fuser -k 8000/tcp || true
    fuser -k 5173/tcp || true
    echo "Done."
}
trap cleanup EXIT