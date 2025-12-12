#!/bin/bash
set -e

WIN_CHROME="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

STREAMLIT_URL="http://localhost:8501"

echo "Cleaning old processes..."
fuser -k 8501/tcp || true

echo "[1/4] Checking Python venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Activating venv..."
source venv/bin/activate

echo "[2/4] Installing Python dependencies..."
pip install -r requirements.txt

echo "[3/4] Launching Streamlit Interface..."
cd backend && streamlit run streamlit_presentation.py &
STREAMLIT_PID=$!

echo "Waiting for Streamlit to be ready..."
until curl -s $STREAMLIT_URL > /dev/null; do
    sleep 1
done
echo "Streamlit is ready."

echo "[4/4] Opening browser..."
"$WIN_CHROME" "$STREAMLIT_URL"

echo "All services are running."
echo "Streamlit PID: $STREAMLIT_PID"

wait $STREAMLIT_PID

cleanup() {
    echo "Cleaning up..."
    fuser -k 8501/tcp || true
    echo "Done."
}
trap cleanup EXIT