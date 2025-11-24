#!/usr/bin/env bash
# Preflight checklist for running pytest tests
# This script verifies the environment is correctly configured before running tests

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
VENV_PYTEST="${ROOT_DIR}/.venv/bin/pytest"
LLM_SERVER_URL="http://localhost:8000"

echo "=========================================="
echo "Preflight Checklist for EcoRenoAdvisor"
echo "=========================================="
echo ""

# 1. Check working directory
echo "[1/5] Checking working directory..."
if [[ "$(pwd)" != "$ROOT_DIR" ]]; then
    echo "  [WARN] Not in project root"
    echo "  Current: $(pwd)"
    echo "  Expected: $ROOT_DIR"
    echo "  Fix: cd $ROOT_DIR"
    exit 1
else
    echo "  [PASS] Working directory: $ROOT_DIR"
fi
echo ""

# 2. Check venv exists and Python interpreter
echo "[2/5] Checking Python interpreter..."
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "  [FAIL] .venv/bin/python not found at $VENV_PYTHON"
    echo "  Fix: Run 'uv venv' to create .venv"
    exit 1
else
    PYTHON_VERSION=$("$VENV_PYTHON" --version)
    PYTHON_PATH=$("$VENV_PYTHON" -c "import sys; print(sys.executable)")
    echo "  [PASS] Python found: $PYTHON_VERSION"
    echo "  [PASS] Python path: $PYTHON_PATH"
    
    # Verify it's the venv Python, not system Python
    if [[ "$PYTHON_PATH" == *"/usr/bin/python"* ]]; then
        echo "  [WARN] Python path looks like system Python, not .venv"
        echo "  Expected path to contain: .venv/bin/python"
    fi
fi
echo ""

# 3. Check pytest is available in venv
echo "[3/5] Checking pytest in venv..."
if [[ ! -f "$VENV_PYTEST" ]]; then
    echo "  [FAIL] .venv/bin/pytest not found"
    echo "  Fix: Run 'uv pip install pytest'"
    exit 1
else
    PYTEST_VERSION=$("$VENV_PYTEST" --version)
    echo "  [PASS] Pytest found: $PYTEST_VERSION"
fi
echo ""

# 4. Check LLM availability (direct or server mode)
echo "[4/5] Checking LLM availability..."
MODEL_FOUND=0
if [[ -f "${ROOT_DIR}/models/qwen2.5-3b-instruct-q4_k_m.gguf" ]]; then
    echo "  [PASS] LLM model file found (direct mode available)"
    MODEL_FOUND=1
elif [[ -f "${ROOT_DIR}/models/llama-3.2-3b-instruct-q4_k_m.gguf" ]]; then
    echo "  [PASS] LLM model file found (direct mode available)"
    MODEL_FOUND=1
else
    echo "  [WARN] No LLM model file found in models/ directory"
    echo "  Direct mode not available"
fi

# Check server mode
if command -v curl &> /dev/null; then
    SERVER_AVAILABLE=0
    if curl -s -f --max-time 2 "${LLM_SERVER_URL}/health" > /dev/null 2>&1; then
        echo "  [PASS] LLM server reachable (server mode available)"
        SERVER_AVAILABLE=1
    elif curl -s -f --max-time 2 -X POST "${LLM_SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local-llm","messages":[{"role":"user","content":"test"}],"max_tokens":5}' > /dev/null 2>&1; then
        echo "  [PASS] LLM server reachable (server mode available)"
        SERVER_AVAILABLE=1
    fi
    
    if [[ $MODEL_FOUND -eq 0 && $SERVER_AVAILABLE -eq 0 ]]; then
        echo "  [WARN] Neither direct mode nor server mode available"
        echo "  LLM tests will be SKIPPED"
        echo "  To enable:"
        echo "    - Download model to models/qwen2.5-3b-instruct-q4_k_m.gguf (direct mode)"
        echo "    - OR start server: .venv/bin/python -m llama_cpp.server --model models/... --port 8000"
    fi
else
    echo "  [WARN] curl not available, cannot check server mode"
fi
echo ""

# 5. Check PYTHONPATH
echo "[5/5] Checking PYTHONPATH..."
if [[ -z "${PYTHONPATH:-}" ]]; then
    echo "  [WARN] PYTHONPATH not set"
    echo "  Recommended: export PYTHONPATH=\"${ROOT_DIR}:\${PYTHONPATH:-}\""
else
    if [[ "$PYTHONPATH" == *"$ROOT_DIR"* ]]; then
        echo "  [PASS] PYTHONPATH includes project root: $PYTHONPATH"
    else
        echo "  [WARN] PYTHONPATH does not include project root"
        echo "  Current: $PYTHONPATH"
        echo "  Recommended: export PYTHONPATH=\"${ROOT_DIR}:\${PYTHONPATH:-}\""
    fi
fi
echo ""

echo "=========================================="
echo "Preflight check complete!"
echo "=========================================="
echo ""
echo "To run tests with venv Python:"
echo "  cd $ROOT_DIR"
echo "  export PYTHONPATH=\"${ROOT_DIR}:\${PYTHONPATH:-}\""
echo "  .venv/bin/pytest tests/test_llm_simple_rag.py -v -s"
echo ""



