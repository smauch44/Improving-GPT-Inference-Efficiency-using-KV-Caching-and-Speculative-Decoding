#!/usr/bin/env bash
# =============================================================================
#  run.sh  —  Complete pipeline for the KV-Cache / Speculative Decoding project
#  EN.705.743.8VL.SP26 — Stefan Mauch
# =============================================================================
#
#  Usage:
#    ./run.sh              Full pipeline (train → benchmark → demo)
#    ./run.sh --quick      Fast smoke-test (~5 min on CPU, ~1 min on GPU)
#    ./run.sh --no-train   Skip training; use existing ./checkpoints/
#    ./run.sh --demo-only  Generation demo only (no benchmarks)
#    ./run.sh --plots-only Regenerate figures from saved benchmark_data.json
#    ./run.sh --help       Show this message
#
#  Output:
#    checkpoints/          Trained model weights (.pt)
#    results/              Benchmark data, plots (PNG + PDF), pgfplots .tex files,
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
section() { echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n  $*\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

QUICK=0; NO_TRAIN=0; DEMO_ONLY=0; PLOTS_ONLY=0
for arg in "$@"; do
  case $arg in
    --quick) QUICK=1 ;;
    --no-train) NO_TRAIN=1 ;;
    --demo-only) DEMO_ONLY=1 ;;
    --plots-only) PLOTS_ONLY=1 ;;
    --help|-h)
      cat <<'HELP'
Usage:
  ./run.sh              Full pipeline (train → benchmark → demo)
  ./run.sh --quick      Fast smoke-test
  ./run.sh --no-train   Skip training; use existing ./checkpoints/
  ./run.sh --demo-only  Generation demo only
  ./run.sh --plots-only Regenerate figures from saved benchmark_data.json
HELP
      exit 0 ;;
    *) warn "Unknown flag: $arg (ignored)" ;;
  esac
done

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║   GPT Inference Efficiency: KV-Cache & Speculative Decoding  ║"
echo "  ║   EN.705.743.8VL.SP26  —  Stefan Mauch                       ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

START_TIME=$(date +%s)

section "Step 0 — Environment"

BASE_PYTHON=$(command -v python3 || command -v python || true)
[ -n "$BASE_PYTHON" ] || error "Python not found"

info "System Python: $($BASE_PYTHON --version)"
info "Working directory: $(pwd)"

# Create and use a local virtual environment to avoid macOS/Homebrew PEP 668 issues.
if [ ! -d ".venv" ]; then
  info "Creating local virtual environment in .venv ..."
  "$BASE_PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
PYTHON="$(pwd)/.venv/bin/python"

info "Virtualenv Python: $($PYTHON --version)"

info "Upgrading pip inside virtualenv ..."
"$PYTHON" -m pip install --upgrade pip setuptools wheel >/dev/null

info "Checking required packages in virtualenv ..."
"$PYTHON" - <<'PYCHECK'
import pkgutil
import subprocess
import sys

required = ("torch", "numpy", "matplotlib", "scipy")
missing = [pkg for pkg in required if pkgutil.find_loader(pkg) is None]

if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("All required packages already installed.")
PYCHECK

mkdir -p checkpoints results data
success "Environment ready."

section "Step 1 — Architecture Sanity Check"
"$PYTHON" gpt_model.py
success "gpt_model.py: forward pass OK"

section "Step 2 — Dataset"
"$PYTHON" data_utils.py
success "Tiny Shakespeare loaded."

if [ $PLOTS_ONLY -eq 1 ]; then
  section "Plots-only mode — regenerating figures"
  [ -f results/benchmark_data.json ] || error "results/benchmark_data.json not found. Run full pipeline first."
  "$PYTHON" plots.py --data results/benchmark_data.json
  success "Figures regenerated."
  exit 0
fi

if [ $NO_TRAIN -eq 0 ] && [ $DEMO_ONLY -eq 0 ]; then
  section "Step 3 — Training"
  if [ -f checkpoints/main_model.pt ] && [ -f checkpoints/draft_model.pt ] && [ $QUICK -eq 0 ]; then
    info "Checkpoints found — skipping training."
  else
    TRAIN_FLAGS=""
    [ $QUICK -eq 1 ] && TRAIN_FLAGS="--quick"
    "$PYTHON" train.py $TRAIN_FLAGS
    success "Training complete. Checkpoints in ./checkpoints/"
  fi
fi

section "Step 4 — Inference Smoke Test"
"$PYTHON" inference.py
success "Inference smoke test complete."

section "Step 5 — Full Pipeline (Benchmarks + Demo)"
MAIN_FLAGS="--no-train"
[ $QUICK -eq 1 ] && MAIN_FLAGS="$MAIN_FLAGS --quick"
[ $DEMO_ONLY -eq 1 ] && MAIN_FLAGS="$MAIN_FLAGS --demo-only"

if [ $NO_TRAIN -eq 0 ] && [ $DEMO_ONLY -eq 0 ] && [ $QUICK -eq 0 ]; then
  MAIN_FLAGS=""
fi

"$PYTHON" main.py $MAIN_FLAGS
success "main.py complete."

section "Step 6 — Output Summary"
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo -e "  ${BOLD}Results saved to ./results/${RESET}"
if [ -d results ]; then
  echo ""
  echo -e "  ${CYAN}Artifacts:${RESET}"
  find results -maxdepth 1 -type f | sort | sed 's/^/    /'
fi

echo ""
echo -e "  ${GREEN}${BOLD}Pipeline complete in ${ELAPSED}s.${RESET}"
echo -e "  ${CYAN}Virtual environment:${RESET} $(pwd)/.venv"
