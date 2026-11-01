#!/usr/bin/env bash
# Run upstream run_qa.py on the LLM Farm. Usage: experiments/01_farm_small_run/farm_run.sh <Naive|Sequential> <Oracle|Ignoramus|SparseRAG|DenseRAG>
# Credentials: taken from the environment, else from .env, else from $FARM_ENV_FILE.
# Only the three variable names below are read, at run time; values are never written anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
[ $# -eq 2 ] || { echo "usage: $0 <Naive|Sequential> <Oracle|Ignoramus|SparseRAG|DenseRAG>" >&2; exit 2; }

load_vars() {  # $1 = file
  [ -f "$1" ] || return 0
  set -a
  # shellcheck disable=SC1090
  source <(grep -E '^(LLM_FARM_ENDPOINT|LLM_FARM_KEY|SMALL_MODEL_NAME)=' "$1")
  set +a
}
if [ -z "${LLM_FARM_KEY:-}" ]; then load_vars .env; fi
if [ -z "${LLM_FARM_KEY:-}" ] && [ -n "${FARM_ENV_FILE:-}" ]; then load_vars "$FARM_ENV_FILE"; fi
for v in LLM_FARM_ENDPOINT LLM_FARM_KEY SMALL_MODEL_NAME; do
  [ -n "${!v:-}" ] || { echo "$v is not set (see env.example)" >&2; exit 1; }
done

export MPR_DATA_DIR="${MPR_DATA_DIR:-$PWD/data/subsets}"
export MPR_DATA_TITLE="${MPR_DATA_TITLE:-user0_1perhop}"
[ -f "$MPR_DATA_DIR/$MPR_DATA_TITLE.json" ] || { echo "missing $MPR_DATA_DIR/$MPR_DATA_TITLE.json (run setup.sh)" >&2; exit 1; }
mkdir -p results/logs
cd upstream/MPR
mkdir -p results
log="../../results/logs/$1-$2.log"
"$OLDPWD/.venv/bin/python" -u run_qa.py "$1" "$2" > "$log" 2>&1
echo "$1 $2: exit 0, $(grep -c '^\[Usage\]' "$log") calls, in/out tokens: $(grep '^\[Usage\]' "$log" | sed -E 's/.*in=([0-9]+) out=([0-9]+)/\1 \2/' | awk '{i+=$1;o+=$2} END{print i+0 " / " o+0}')"
