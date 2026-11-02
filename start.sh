#!/usr/bin/env bash
# ==============================================================================
# PlangoGraph Service Launcher & Colab GPU Bridge
# ==============================================================================
# - Local runtime: 100% CPU-only (Zero NVIDIA/CUDA bloatware)
# - GPU tasks: Offloaded to Google Colab via `colab` CLI
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/.venv"
API_PORT=8000
UI_PORT=8501
MODE="both"
COLAB_SESSION="plangograph-gpu"
COLAB_GPU_TYPE="L4"

print_usage() {
    cat <<EOF
PlangoGraph Control Launcher

Local CPU Execution (Zero NVIDIA Bloat):
  ./start.sh               Launch both FastAPI backend and Streamlit UI (default)
  ./start.sh --both        Same as above
  ./start.sh --api         Launch only FastAPI backend (http://localhost:${API_PORT})
  ./start.sh --ui          Launch only Streamlit dashboard (http://localhost:${UI_PORT})
  ./start.sh --test        Run the local unit test suite (pytest/unittest)
  ./start.sh --benchmark   Run 500-scene Monte Carlo ablation benchmark
  ./start.sh --setup       Create .venv and install clean CPU dependencies

Google Colab GPU Offloading (via colab CLI):
  ./start.sh --colab-new   Provision a new Colab GPU session (${COLAB_GPU_TYPE})
  ./start.sh --colab-train Run YOLOv8m SKU-110K training notebook on Colab GPU
  ./start.sh --colab-exec  Run DINOv2 graph experiments notebook on Colab GPU
  ./start.sh --colab-stop  Tear down the remote Colab GPU session

Options:
  -h, --help               Show this help message
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --both)
            MODE="both"
            shift
            ;;
        --api)
            MODE="api"
            shift
            ;;
        --ui)
            MODE="ui"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --benchmark)
            MODE="benchmark"
            shift
            ;;
        --setup)
            MODE="setup"
            shift
            ;;
        --colab-new)
            MODE="colab-new"
            shift
            ;;
        --colab-train)
            MODE="colab-train"
            shift
            ;;
        --colab-exec)
            MODE="colab-exec"
            shift
            ;;
        --colab-stop)
            MODE="colab-stop"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Handle remote Colab tasks directly
case "${MODE}" in
    colab-new)
        echo "[*] Provisioning Google Colab GPU session '${COLAB_SESSION}' (${COLAB_GPU_TYPE})..."
        colab new -s "${COLAB_SESSION}" --gpu "${COLAB_GPU_TYPE}"
        echo "[*] Installing GPU requirements on remote Colab VM..."
        colab install -s "${COLAB_SESSION}" -r requirements-gpu.txt
        echo "[✓] Colab session '${COLAB_SESSION}' ready. Check status with 'colab status -s ${COLAB_SESSION}'"
        exit 0
        ;;
    colab-train)
        echo "[*] Launching YOLOv8 training on Google Colab (${COLAB_GPU_TYPE})..."
        colab run --gpu "${COLAB_GPU_TYPE}" notebooks/01_yolo_training.ipynb
        exit 0
        ;;
    colab-exec)
        echo "[*] Running DINOv2 graph experiments on Google Colab (${COLAB_GPU_TYPE})..."
        colab run --gpu "${COLAB_GPU_TYPE}" notebooks/02_graph_experiments.ipynb
        exit 0
        ;;
    colab-stop)
        echo "[*] Stopping Google Colab GPU session '${COLAB_SESSION}'..."
        colab stop -s "${COLAB_SESSION}"
        echo "[✓] Session stopped. Compute units preserved."
        exit 0
        ;;
esac

# Ensure local virtual environment is active
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    python3 -c "import onnxruntime" 2>/dev/null || pip install -q --retries 0 --timeout 2 -r requirements.txt 2>/dev/null || true
elif [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
    python3 -c "import onnxruntime" 2>/dev/null || pip install -q --retries 0 --timeout 2 -r requirements.txt 2>/dev/null || true
else
    echo "[+] Creating clean CPU Python virtual environment in .venv..."
    python3 -m venv "${VENV_DIR}"
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    echo "[+] Installing lightweight CPU requirements (Zero NVIDIA libraries)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "[✓] Setup complete."
    if [[ "${MODE}" == "setup" ]]; then
        exit 0
    fi
fi

# Trap to gracefully shut down background processes on exit
cleanup() {
    if [[ "${MODE}" == "both" ]]; then
        echo ""
        echo "[!] Stopping PlangoGraph services..."
        if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
            kill -TERM "${API_PID}" 2>/dev/null || true
        fi
        if [[ -n "${UI_PID:-}" ]] && kill -0 "${UI_PID}" 2>/dev/null; then
            kill -TERM "${UI_PID}" 2>/dev/null || true
        fi
        wait 2>/dev/null || true
        echo "[!] Services stopped."
    fi
}
trap cleanup SIGINT SIGTERM EXIT

case "${MODE}" in
    setup)
        echo "[✓] Environment is already set up and active."
        ;;
    test)
        echo "[*] Running PlangoGraph unit test suite..."
        python3 -m unittest discover -s tests -v
        ;;
    benchmark)
        echo "[*] Running PlangoGraph 500-scene Monte Carlo benchmark..."
        python3 scripts/run_benchmarks.py
        ;;
    api)
        echo "[*] Launching FastAPI backend on http://0.0.0.0:${API_PORT}..."
        uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" --reload
        ;;
    ui)
        echo "[*] Launching Streamlit dashboard on http://localhost:${UI_PORT}..."
        streamlit run app.py --server.port "${UI_PORT}" --server.headless true
        ;;
    both)
        echo "=========================================================="
        echo " PlangoGraph: Retail Shelf Compliance Service (CPU Mode)"
        echo "=========================================================="
        echo "[*] Starting FastAPI backend on http://0.0.0.0:${API_PORT}..."
        uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" &
        API_PID=$!

        echo "[*] Starting Streamlit UI on http://localhost:${UI_PORT}..."
        streamlit run app.py --server.port "${UI_PORT}" --server.headless true &
        UI_PID=$!

        echo "[✓] Services started."
        echo "    - FastAPI docs: http://localhost:${API_PORT}/docs"
        echo "    - Streamlit UI: http://localhost:${UI_PORT}"
        echo "    (Press Ctrl+C to stop both)"
        wait
        ;;
esac
