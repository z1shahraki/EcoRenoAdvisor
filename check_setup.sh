#!/usr/bin/env bash
# Comprehensive setup verification script for EcoRenoAdvisor
# This script checks all components and provides a completion status

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
VENV_PYTEST="${ROOT_DIR}/.venv/bin/pytest"
LLM_SERVER_URL="http://localhost:8000"
QDRANT_URL="http://localhost:6333"

echo "=========================================="
echo "EcoRenoAdvisor - Setup Verification"
echo "=========================================="
echo ""

# Initialize counters
PASSED=0
FAILED=0
WARNINGS=0
SKIPPED=0

check_pass() {
    echo "  [PASS] $1"
    ((PASSED++))
}

check_fail() {
    echo "  [FAIL] $1"
    ((FAILED++))
}

check_warn() {
    echo "  [WARN] $1"
    ((WARNINGS++))
}

check_skip() {
    echo "  [SKIP] $1"
    ((SKIPPED++))
}

# 1. Environment Checks
echo "[1/10] Environment Setup"
echo "----------------------"
cd "$ROOT_DIR" || exit 1
if [[ "$(pwd)" == "$ROOT_DIR" ]]; then
    check_pass "Working directory: $ROOT_DIR"
else
    check_fail "Working directory mismatch"
fi

if [[ -f "$VENV_PYTHON" ]]; then
    PYTHON_VERSION=$("$VENV_PYTHON" --version 2>&1)
    check_pass "Python: $PYTHON_VERSION"
else
    check_fail "venv/bin/python not found"
fi

if [[ -f "$VENV_PYTEST" ]]; then
    PYTEST_VERSION=$("$VENV_PYTEST" --version 2>&1)
    check_pass "Pytest: $PYTEST_VERSION"
else
    check_fail "venv/bin/pytest not found"
fi
echo ""

# 2. Python Packages
echo "[2/10] Python Dependencies"
echo "------------------------"
if "$VENV_PYTHON" -c "import llama_cpp; print('llama_cpp')" 2>/dev/null; then
    check_pass "llama_cpp installed"
else
    check_fail "llama_cpp not installed"
fi

if "$VENV_PYTHON" -c "import sentence_transformers; print('sentence_transformers')" 2>/dev/null; then
    check_pass "sentence_transformers installed"
else
    check_fail "sentence_transformers not installed"
fi

if "$VENV_PYTHON" -c "import gradio; print('gradio')" 2>/dev/null; then
    check_pass "gradio installed"
else
    check_fail "gradio not installed"
fi

if "$VENV_PYTHON" -c "import qdrant_client; print('qdrant_client')" 2>/dev/null; then
    check_pass "qdrant_client installed"
else
    check_warn "qdrant_client not installed (optional)"
fi
echo ""

# 3. Data Files
echo "[3/10] Data Files"
echo "----------------"
if [[ -f "${ROOT_DIR}/data/clean/materials.parquet" ]]; then
    check_pass "materials.parquet exists"
else
    check_warn "materials.parquet not found (run ingestion/clean_materials.py)"
fi

if [[ -f "${ROOT_DIR}/models/qwen2.5-3b-instruct-q4_k_m.gguf" ]]; then
    check_pass "LLM model file exists"
else
    check_warn "LLM model file not found (optional - download if needed)"
fi
echo ""

# 4. LLM Server
echo "[4/10] LLM Server"
echo "----------------"
if command -v curl &> /dev/null; then
    if curl -s -f --max-time 2 "${LLM_SERVER_URL}/health" > /dev/null 2>&1; then
        check_pass "LLM server health endpoint reachable"
    elif curl -s -f --max-time 2 -X POST "${LLM_SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local-llm","messages":[{"role":"user","content":"test"}],"max_tokens":5}' > /dev/null 2>&1; then
        check_pass "LLM server chat endpoint reachable (health returns 404)"
    else
        check_skip "LLM server not running (optional - start with: venv/bin/python -m llama_cpp.server)"
    fi
else
    check_skip "curl not available, cannot check LLM server"
fi
echo ""

# 5. Qdrant Server
echo "[5/10] Qdrant Server"
echo "------------------"
if command -v curl &> /dev/null; then
    if curl -s -f --max-time 2 "${QDRANT_URL}/health" > /dev/null 2>&1; then
        check_pass "Qdrant server reachable"
    else
        check_skip "Qdrant not running (optional - start with: docker-compose up -d)"
    fi
else
    check_skip "curl not available, cannot check Qdrant"
fi
echo ""

# 6. Configuration Files
echo "[6/10] Configuration Files"
echo "------------------------"
if [[ -f "${ROOT_DIR}/pytest.ini" ]]; then
    check_pass "pytest.ini exists"
else
    check_warn "pytest.ini not found"
fi

if [[ -f "${ROOT_DIR}/.vscode/settings.json" ]]; then
    check_pass "VS Code settings.json exists"
else
    check_warn "VS Code settings.json not found"
fi

if [[ -f "${ROOT_DIR}/.vscode/launch.json" ]]; then
    check_pass "VS Code launch.json exists"
else
    check_warn "VS Code launch.json not found"
fi
echo ""

# 7. Test Files
echo "[7/10] Test Files"
echo "----------------"
TEST_FILES=(
    "tests/test_chunking.py"
    "tests/test_filter_materials.py"
    "tests/test_simple_embeddings.py"
    "tests/test_agent_mocked.py"
    "tests/test_ui_smoke.py"
    "tests/test_llm_simple_rag.py"
)

for test_file in "${TEST_FILES[@]}"; do
    if [[ -f "${ROOT_DIR}/${test_file}" ]]; then
        check_pass "$(basename "$test_file") exists"
    else
        check_warn "$(basename "$test_file") not found"
    fi
done
echo ""

# 8. Code Structure
echo "[8/10] Code Structure"
echo "-------------------"
if [[ -d "${ROOT_DIR}/agent" ]]; then
    check_pass "agent/ directory exists"
else
    check_fail "agent/ directory missing"
fi

if [[ -d "${ROOT_DIR}/ui" ]]; then
    check_pass "ui/ directory exists"
else
    check_fail "ui/ directory missing"
fi

if [[ -d "${ROOT_DIR}/ingestion" ]]; then
    check_pass "ingestion/ directory exists"
else
    check_fail "ingestion/ directory missing"
fi

if [[ -d "${ROOT_DIR}/rag" ]]; then
    check_pass "rag/ directory exists"
else
    check_warn "rag/ directory missing"
fi
echo ""

# 9. Python Path Configuration
echo "[9/10] Python Path"
echo "-----------------"
if [[ -n "${PYTHONPATH:-}" ]]; then
    if [[ "$PYTHONPATH" == *"$ROOT_DIR"* ]]; then
        check_pass "PYTHONPATH includes project root"
    else
        check_warn "PYTHONPATH does not include project root"
    fi
else
    check_warn "PYTHONPATH not set (export PYTHONPATH=\"${ROOT_DIR}:\${PYTHONPATH:-}\")"
fi
echo ""

# 10. Quick Import Test
echo "[10/10] Quick Import Test"
echo "-----------------------"
if "$VENV_PYTHON" -c "
import sys
sys.path.insert(0, '${ROOT_DIR}')
from agent.agent import RenovationAgent
from agent.tools import filter_materials
print('Core imports successful')
" 2>/dev/null; then
    check_pass "Core modules importable"
else
    check_fail "Core modules cannot be imported"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "[PASS] Passed:  $PASSED"
echo "[FAIL] Failed:  $FAILED"
echo "[WARN] Warnings: $WARNINGS"
echo "[SKIP] Skipped:  $SKIPPED"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo "All critical checks passed!"
    if [[ $WARNINGS -gt 0 ]]; then
        echo "[WARN] Some optional components are missing, but core functionality is ready."
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: .venv/bin/pytest tests/ -v"
    echo "  2. Start LLM server (if model available): .venv/bin/python -m llama_cpp.server --model models/... --port 8000"
    echo "  3. Launch UI: .venv/bin/python ui/app.py"
else
    echo "[FAIL] Some critical checks failed. Please review the errors above."
    exit 1
fi

