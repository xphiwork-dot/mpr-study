#!/usr/bin/env bash
# Local driver for the Colab T4 experiments. Re-runnable; every step is resumable.
# Needs the `colab` CLI (google-colab-cli) already authorised; never prompts.
#
#   t4.sh up                      create the T4 session (retries), upload code + dataset + earlier results, install vLLM
#   t4.sh serve                   start the vLLM server on the VM (Qwen2.5-7B-Instruct-AWQ, fp16) and wait until healthy
#   t4.sh unserve                 stop the vLLM server (frees the GPU for retrieval / LoRA)
#   t4.sh job NAME SCRIPT [ARGS]  run SCRIPT ARGS on the VM in the background, poll, then pull results (SCRIPT is a bare file
#                                 name such as grid.py; `up` ships every experiment script flat into /content/t4/scripts/t4/)
#   t4.sh pull                    download results/t4 from the VM into the repo
#   t4.sh status                  session, server and job state
#   t4.sh down                    pull, then stop the session (always do this when finished)
#
# Env: T4_SESSION (default mpr-t4). Paths on the VM: /content/t4/{scripts/t4,upstream,data,cache,results,logs}.
# Lives in experiments/_shared/; the repo root is two levels up.
set -uo pipefail
cd "$(dirname "$0")/../.."
S="${T4_SESSION:-mpr-t4}"
UPSTREAM_COMMIT=a24953c
MODEL_REV=b25037543e9394b818fdfca67ab2a00ecc7dd641   # Qwen/Qwen2.5-7B-Instruct-AWQ
LOG=results/t4/timing_log.md

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { echo "- $(stamp) $*" | tee -a "$LOG"; }
vm() { timeout 180 colab exec -s "$S" 2>&1; }   # python on stdin; time-limited because a hung exec call blocked the driver for 30 min once

cmd_up() {
  mkdir -p results/t4 data
  [ -f "$LOG" ] || printf '# Colab T4 timing log\n\nTimestamps are UTC; appended by experiments/_shared/t4.sh.\n\n' > "$LOG"
  if colab sessions 2>&1 | grep -q "$S"; then
    note "session $S already exists, reusing"
  else
    local n=0
    until colab new -s "$S" --gpu T4; do
      n=$((n+1)); note "colab new failed (attempt $n)"
      [ $n -ge 4 ] && { echo "no T4 available after $n attempts" >&2; return 1; }
      sleep $((60*n))
    done
    note "session $S created (T4)"
  fi
  vm <<'PY' | tail -1
import os; os.makedirs('/content/t4/logs', exist_ok=True); print('dirs ok')
PY
  # code, dataset zip, earlier results (so a fresh VM resumes where the last one stopped)
  # The VM keeps the original flat layout scripts/t4/<name>.py, so scripts import each other by bare name there.
  local stage=/tmp/t4_stage
  rm -rf "$stage"; mkdir -p "$stage/scripts/t4" "$stage/results"
  cp experiments/_shared/*.py experiments/s*/*.py experiments/00_probes_retrieval_leak/leak_check.py "$stage/scripts/t4/"
  cp -r results/t4/slices "$stage/results/slices"
  tar czf /tmp/t4_code.tgz -C "$stage" scripts results
  colab upload -s "$S" /tmp/t4_code.tgz /content/t4_code.tgz
  [ -f data/mpr_dataset.zip ] && colab upload -s "$S" data/mpr_dataset.zip /content/t4_dataset.zip
  if [ -f results/t4/vm_results.tgz ]; then colab upload -s "$S" results/t4/vm_results.tgz /content/t4_prev.tgz; fi
  # The setup runs as a detached script on the VM and is polled with short calls: one long `colab exec` cell
  # (the vLLM install takes minutes) hung or lost its connection in earlier sessions.
  vm <<PY | tail -1
open('/content/setup.sh', 'w').write("""set -x
cd /content
mkdir -p t4 && tar xzf t4_code.tgz -C t4
[ -f /content/t4_prev.tgz ] && tar xzf t4_prev.tgz -C t4
[ -d /content/t4/data/mpr_dataset/question_answer ] || { mkdir -p t4/data && python -I -c "import zipfile; zipfile.ZipFile('/content/t4_dataset.zip').extractall('/content/t4/data')"; }
[ -d /content/t4/upstream/MPR/.git ] || { mkdir -p t4/upstream && git clone -q https://github.com/nuster1128/MPR t4/upstream/MPR; }
git -C t4/upstream/MPR checkout -q $UPSTREAM_COMMIT
python -c "import importlib.util as u; assert u.find_spec('vllm')" || uv pip install vllm bm25s==0.2.9 sentence-transformers 2>&1 | tail -3
uv pip uninstall torchao 2>&1 | tail -1
uv pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers|bm25s|sentence-transformers|numpy) '
python --version
echo SETUPDONE
""")
import subprocess, os
if os.path.exists('/content/setup.log'): os.remove('/content/setup.log')
subprocess.Popen('bash /content/setup.sh > /content/setup.log 2>&1', shell=True, start_new_session=True)
print('setup launched')
PY
  local i out
  for i in $(seq 1 90); do
    sleep 20
    out=$(vm <<'PY' | tail -14
import os
print(open('/content/setup.log').read()[-900:] if os.path.exists('/content/setup.log') else 'nolog')
PY
)
    echo "$out" | grep -q SETUPDONE && { echo "$out"; return 0; }
  done
  note "VM setup did not finish in 30 min"; return 1
}

cmd_serve() {
  vm <<PY | tail -3
import subprocess, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); print('server already healthy')
except Exception:
    cmd = ("vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --revision $MODEL_REV --served-model-name qwen --dtype float16 "
           "--quantization awq --max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 64 "
           "--generation-config vllm --port 8000 --host 127.0.0.1")
    subprocess.Popen(cmd + ' > /content/t4/logs/vllm.log 2>&1', shell=True, start_new_session=True)
    print('server starting')
PY
  local t0=$SECONDS
  for i in $(seq 1 60); do
    if vm <<'PY' | grep -q healthy; then
import urllib.request
try: urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); print('healthy')
except Exception: print('no')
PY
      note "vLLM healthy after $((SECONDS-t0)) s of waiting"; return 0
    fi
    sleep 15
  done
  note "vLLM did not become healthy in 15 min"; return 1
}

cmd_unserve() {
  vm <<'PY' | tail -2
import subprocess; subprocess.run("pkill -f 'vllm serve'; pkill -f EngineCore; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True)
PY
}

cmd_pull() {
  vm <<'PY' | tail -1
import subprocess; subprocess.run("cd /content/t4 && tar czf /content/vm_results.tgz results logs/*.log logs/*.exit 2>/dev/null; ls -la /content/vm_results.tgz", shell=True)
PY
  colab download -s "$S" /content/vm_results.tgz results/t4/vm_results.tgz && \
  mkdir -p results/t4/_x && tar xzf results/t4/vm_results.tgz -C results/t4/_x && \
  cp -r results/t4/_x/results/. results/t4/ && { mkdir -p results/t4/logs; cp -r results/t4/_x/logs/. results/t4/logs/ 2>/dev/null; } ; rm -rf results/t4/_x
  note "pulled results from $S"
}

cmd_job() {
  local name="$1" script="$2"; shift 2
  local args="$*"
  note "job $name start: $script $args"
  local t0=$SECONDS
  vm <<PY | tail -2
import subprocess, os
os.chdir('/content/t4')
p = '/content/t4/logs/$name.exit'
if os.path.exists(p): os.remove(p)
cmd = "cd /content/t4 && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false T4_ROOT=/content/t4 python scripts/t4/$script $args > logs/$name.log 2>&1; echo \$? > logs/$name.exit"
subprocess.Popen(cmd, shell=True, start_new_session=True)
print('launched $name')
PY
  local out fails=0 polls=0
  while true; do
    sleep 60
    polls=$((polls+1))
    out=$(vm <<PY | tail -4
import os
e = '/content/t4/logs/$name.exit'
print('EXIT=' + open(e).read().strip() if os.path.exists(e) else 'EXIT=running')
print(''.join(open('/content/t4/logs/$name.log').readlines()[-2:])[-400:])
PY
)
    echo "$out" | tail -3
    if echo "$out" | grep -q 'EXIT='; then
      fails=0
      echo "$out" | grep -q 'EXIT=running' || break
      # pull partial results every 10 polls so a cut session loses little (the grid resumes from the jsonl files)
      [ $((polls % 10)) -eq 0 ] && cmd_pull
    else
      fails=$((fails+1)); [ $fails -ge 5 ] && break    # 5 failed polls in a row: session lost
    fi
  done
  echo "$out" | grep -q 'EXIT=' || { note "poll failed (session lost?) - rerun: t4.sh up; t4.sh job $name $script $args"; return 2; }
  local rc; rc=$(echo "$out" | grep -o 'EXIT=[0-9]*' | cut -d= -f2)
  note "job $name finished rc=$rc wall=$((SECONDS-t0)) s"
  cmd_pull
  return "${rc:-1}"
}

cmd_status() {
  colab sessions 2>&1; colab status -s "$S" 2>&1 | tail -3
  vm <<'PY' | tail -5
import subprocess; print(subprocess.run("curl -s -m 3 localhost:8000/health -o /dev/null -w 'vllm http %{http_code}\n'; ls /content/t4/logs; nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)
PY
}

cmd_down() { cmd_pull; colab stop -s "$S" 2>&1 | tail -2; note "session $S stopped"; colab sessions 2>&1; }

c="${1:-}"; shift || true
case "$c" in
  up) cmd_up;; serve) cmd_serve;; unserve) cmd_unserve;; pull) cmd_pull;; status) cmd_status;; down) cmd_down;;
  job) cmd_job "$@";;
  *) sed -n 2,16p "$0"; exit 1;;
esac
