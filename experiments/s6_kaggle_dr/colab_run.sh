#!/usr/bin/env bash
# Colab CLI T4 driver for the ONE S6 DR (DecompositionReason) session.
#
#   experiments/s6_kaggle_dr/colab_run.sh [output-dir]
#
# Why this exists: the shared Kaggle account accepts accelerator requests but runs them on CPU
# (results/kaggle/dr/gpu_availability/, report.md sections 1-2). The `colab` CLI is authenticated and
# gives a live Tesla T4, and the MR worker already produced its 612 rows this way
# (project/mpr-study-mr, `experiments/s5_kaggle_mr/colab_mr.sh`). This driver runs the SAME committed
# kernel.py on that VM; only the input staging path changes, because Colab mounts /kaggle/input read-only.
#
# What it does, in order:
#   1. hold the shared GPU lock (~/.local/state/mpr-study/kaggle-gpu.lock) for the whole job;
#   2. build the private dataset from a CLEAN tree (manifest records the commit), tar it;
#   3. start a fresh Colab session `mpr-dr` (Tesla T4), upload + unpack the dataset at /content/dr_inputs;
#   4. pre-install the vLLM pin that actually served the MR rows (0.11.2, CUDA-12 wheels on driver 580)
#      so the DR engine matches MR and the kernel's 0.32.0 -> 0.11.2 fallback is not needed
#      (set DR_COLAB_SKIP_VLLM_PREINSTALL=1 to let kernel.py pick the pin itself);
#   5. launch kernel.py detached with MPR_INPUT_ROOT=/content/dr_inputs MPR_WORK=/kaggle/working;
#   6. poll every 60 s, pushing milestones to `hc send <parent>` on gate pass, every 20 min and at exit;
#   7. tar /kaggle/working, download and unpack it into results/kaggle/dr/raw, then stop the session.
#
# Environment:
#   DR_COLAB_SESSION            session name (default mpr-dr)
#   DR_COLAB_KEEP=1             leave the session running on exit (debugging)
#   DR_COLAB_PARENT             hc target for milestones (default t19)
#   DR_COLAB_MAX_POLLS          polls at 60 s (default 270 = 4.5 h)
#   DR_COLAB_SKIP_VLLM_PREINSTALL=1   do not pre-install vllm==0.11.2
#   MPR_SESSION_TIMEOUT_S       kernel session cap (default 9000 = the pre-registered 2.5 h)
#   GPU_LOCK_WAIT_S             seconds to wait for the shared lock (default 21600)
#   KAGGLE_GPU_LOCK             lock path override
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LOCK="${KAGGLE_GPU_LOCK:-/home/xphi/.local/state/mpr-study/kaggle-gpu.lock}"
S="${DR_COLAB_SESSION:-mpr-dr}"
PARENT="${DR_COLAB_PARENT:-t19}"
OUT="${1:-$ROOT/results/kaggle/dr/raw}"
STAGE="${DR_COLAB_STAGE:-/tmp/mpr-dr-colab}"
WAIT_S="${GPU_LOCK_WAIT_S:-21600}"
MAX_POLLS="${DR_COLAB_MAX_POLLS:-270}"
SESSION_S="${MPR_SESSION_TIMEOUT_S:-9000}"
RUNLOG="${DR_COLAB_RUNLOG:-$HOME/.local/state/mpr-study/dr-colab-run.log}"

DS="$STAGE/dataset"
DS_TGZ="$STAGE/dr_inputs.tgz"
RESUME_TGZ="$STAGE/dr_resume.tgz"
DL_TGZ="$STAGE/dr_results.tgz"
STATE="$STAGE/state"
mkdir -p "$STAGE" "$STATE" "$OUT" "$(dirname "$LOCK")" "$(dirname "$RUNLOG")"
: > "$RUNLOG"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$RUNLOG"; }

push() {
  # Milestones are advisory: a missing `hc` must never fail the experiment.
  local msg="$1"
  log "hc -> $PARENT: $msg"
  if command -v hc >/dev/null 2>&1; then
    hc send "$PARENT" "$msg" >>"$RUNLOG" 2>&1 || true
  elif [ -x "$HOME/.local/bin/hc" ]; then
    "$HOME/.local/bin/hc" send "$PARENT" "$msg" >>"$RUNLOG" 2>&1 || true
  else
    log "hc not found; milestone not delivered"
  fi
}

exec 9>"$LOCK"
log "waiting for the GPU lock ($LOCK, up to ${WAIT_S}s)"
if ! flock -w "$WAIT_S" 9; then
  log "lock not acquired within ${WAIT_S}s; not starting"
  push "DR Colab BLOCKED: GPU lock not acquired within ${WAIT_S}s"
  exit 1
fi
log "GPU lock acquired"

cleanup() {
  if [ "${DR_COLAB_KEEP:-0}" != "1" ]; then
    timeout 120 colab stop -s "$S" >/dev/null 2>&1 || true
    log "session $S stopped"
  else
    log "DR_COLAB_KEEP=1: session $S left running"
  fi
  log "releasing the GPU lock"
}
trap cleanup EXIT

# ---------------------------------------------------------------- 1. package from a clean tree
log "building the private dataset from HEAD"
python3 "$HERE/package.py" build >>"$RUNLOG" 2>&1 || { log "package.py build failed"; push "DR Colab BLOCKED: package.py build failed (dirty tree or missing inputs); see $RUNLOG"; exit 2; }
rm -rf "$DS"
cp -a "$ROOT/build/kaggle/dr_dataset" "$DS"
tar czf "$DS_TGZ" -C "$DS" .
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
log "dataset staged ($(du -h "$DS_TGZ" | cut -f1)); commit $COMMIT"

# optional resume: previously downloaded DR rows are re-uploaded and skipped by the runner
rm -f "$RESUME_TGZ"
RESUME_ROWS=0
if compgen -G "$ROOT/results/kaggle/dr/runs/DecompositionReason_*.jsonl" >/dev/null; then
  tar czf "$RESUME_TGZ" -C "$ROOT/results/kaggle/dr/runs" -- $(cd "$ROOT/results/kaggle/dr/runs" && ls DecompositionReason_*.jsonl)
  RESUME_ROWS=$(cat "$ROOT"/results/kaggle/dr/runs/DecompositionReason_*.jsonl | wc -l)
  log "resume rows staged: $RESUME_ROWS"
fi

# ---------------------------------------------------------------- 2. fresh T4 session
timeout 120 colab stop -s "$S" >/dev/null 2>&1 || true
n=0
until timeout 300 colab new -s "$S" --gpu T4 >>"$RUNLOG" 2>&1; do
  n=$((n + 1))
  log "colab new failed (attempt $n)"
  if [ "$n" -ge 3 ]; then
    log "no T4 session after 3 attempts"
    push "DR Colab BLOCKED: could not allocate a Colab T4 session after 3 attempts"
    exit 3
  fi
  sleep $((30 * n))
done
log "session $S created"

colab upload -s "$S" "$DS_TGZ" /content/dr_inputs.tgz >>"$RUNLOG" 2>&1
if [ -f "$RESUME_TGZ" ]; then
  colab upload -s "$S" "$RESUME_TGZ" /content/dr_resume.tgz >>"$RUNLOG" 2>&1
fi

# ---------------------------------------------------------------- 3. prepare the VM
prep=$(timeout 300 colab exec -s "$S" --timeout 120 <<'PY'
import subprocess
script = r'''
set -e
mkdir -p /content/dr_inputs
rm -rf /content/dr_inputs
mkdir -p /content/dr_inputs
tar xzf /content/dr_inputs.tgz -C /content/dr_inputs
# The exact committed entrypoint and every hash-covered helper live in the dataset.
test -f /content/dr_inputs/kernel.py
test -f /content/dr_inputs/manifest.json
# Colab mounts /kaggle/input read-only, so the dataset is handed to kernel.py through MPR_INPUT_ROOT
# (checked before the Kaggle mount candidates). Only /kaggle/working is used for outputs.
mkdir -p /kaggle/working/results/kaggle/dr/runs
if [ -f /content/dr_resume.tgz ]; then tar xzf /content/dr_resume.tgz -C /kaggle/working/results/kaggle/dr/runs; fi
rm -f /content/dr_kernel.exit /content/dr_kernel.log
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import sys; print('python', sys.version.split()[0])"
'''
r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=240)
print((r.stdout or '') + (r.stderr or ''))
raise SystemExit(r.returncode)
PY
) || { log "VM preparation failed: $prep"; push "DR Colab BLOCKED: VM preparation failed (upload/unpack/GPU check); see $RUNLOG"; exit 4; }
log "prep: $(echo "$prep" | tr '\n' ' ' | cut -c1-400)"
if ! echo "$prep" | grep -q 'GPU 0'; then
  log "no GPU on the session; refusing to run"
  push "DR Colab BLOCKED: session $S has no GPU (nvidia-smi -L) "
  exit 5
fi
GPU_LINE="$(echo "$prep" | grep -m1 'GPU 0')"

# ---------------------------------------------------------------- 4. engine pin used by MR (0.11.2)
VLLM_STACK=""
if [ "${DR_COLAB_SKIP_VLLM_PREINSTALL:-0}" != "1" ]; then
  log "pre-installing vllm==0.11.2 (the pin that served the MR rows)"
  pin=$(timeout 1800 colab exec -s "$S" --timeout 0 <<'PY'
import subprocess
script = r'''
set -e
python3 -m pip install -q "vllm==0.11.2" 2>&1 | tail -5
python3 -c "import vllm, torch, transformers; print('VLLM=%s TORCH=%s CUDA=%s TRANSFORMERS=%s' % (vllm.__version__, torch.__version__, torch.version.cuda, transformers.__version__))"
'''
r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=1700)
print((r.stdout or '') + (r.stderr or ''))
raise SystemExit(r.returncode)
PY
  ) || true
  log "vllm pre-install output: $(echo "$pin" | tr '\n' ' ' | cut -c1-300)"
  if echo "$pin" | grep -q '^VLLM=0.11.2'; then
    VLLM_STACK="$(echo "$pin" | grep '^VLLM=' | tail -1)"
    log "engine stack: $VLLM_STACK"
  else
    log "vllm pre-install unusable; kernel.py will pick a pin itself (deps.json)"
    VLLM_STACK="preinstall_failed"
  fi
else
  log "DR_COLAB_SKIP_VLLM_PREINSTALL=1: kernel.py picks the pin"
  VLLM_STACK="kernel_selected"
fi

# ---------------------------------------------------------------- 5. launch kernel.py detached
timeout 180 colab exec -s "$S" --timeout 60 <<PY >/dev/null
import subprocess
cmd = ("cd /kaggle/working && "
       "MPR_INPUT_ROOT=/content/dr_inputs "
       "MPR_WORK=/kaggle/working "
       "MPR_SESSION_TIMEOUT_S=$SESSION_S "
       "python3 -u /content/dr_inputs/kernel.py "
       "> /content/dr_kernel.log 2>&1; echo \$? > /content/dr_kernel.exit")
subprocess.Popen(['bash', '-c', cmd], start_new_session=True)
print('launched')
PY
log "kernel launched ($(date -u +%FT%TZ)) with session cap ${SESSION_S}s"
push "DR Colab milestone: T4 session $S ready ($GPU_LINE); inputs staged, sha256-verified by kernel.py; engine ${VLLM_STACK}; kernel.py launched (cap ${SESSION_S}s)"

# ---------------------------------------------------------------- 6. poll (one loop, 60 s)
poll() {
  timeout 150 colab exec -s "$S" --timeout 90 <<'PY'
import glob, json, os
W = '/kaggle/working'
e = '/content/dr_kernel.exit'
print('EXIT=' + (open(e).read().strip() if os.path.exists(e) else 'running'))
st = {}
try:
    st = json.load(open(os.path.join(W, 'status.json')))
except Exception:
    pass
print('STATUS=%s STAGE=%s ELAPSED=%s' % (st.get('status'), st.get('stage'), st.get('elapsed_s')))
counts = {}
for p in glob.glob(os.path.join(W, 'results/kaggle/dr/runs/*.jsonl')):
    name = os.path.basename(p)
    if name.startswith('failures_'):
        continue
    try:
        counts[name] = sum(1 for line in open(p, encoding='utf-8', errors='replace') if line.strip())
    except OSError:
        counts[name] = -1
print('ROWS %s %s' % (counts.get('DecompositionReason_SparseRAG_k20.jsonl', 0),
                      counts.get('DecompositionReason_DenseRAG_k20.jsonl', 0)))
log = os.path.join(W, 'run.log')
if os.path.exists(log):
    for line in ''.join(open(log, errors='replace').readlines()[-3:]).splitlines()[-3:]:
        print('LOG ' + line[:220])
PY
}

fails=0
GATE_PUSHED=0
LAST_HEARTBEAT=0
STATUS_LAST=""
STAGE_LAST=""
ROWS_S=0
ROWS_D=0
EXIT_V="running"
STARTED=$(date +%s)
for i in $(seq 1 "$MAX_POLLS"); do
  sleep 60
  if ! out=$(poll); then
    fails=$((fails + 1))
    log "poll failed ($fails)"
    if [ "$fails" -ge 5 ]; then
      push "DR Colab BLOCKED: lost the session $S (5 consecutive poll failures, rows sparse=$ROWS_S dense=$ROWS_D)"
      break
    fi
    continue
  fi
  fails=0
  EXIT_V="$(printf '%s\n' "$out" | sed -n 's/^EXIT=//p' | head -1)"
  line_status="$(printf '%s\n' "$out" | grep -m1 '^STATUS=' || true)"
  STATUS_LAST="$(printf '%s' "$line_status" | sed -n 's/^STATUS=\([^ ]*\).*/\1/p')"
  STAGE_LAST="$(printf '%s' "$line_status" | sed -n 's/^.*STAGE=\([^ ]*\).*/\1/p')"
  ELAPSED="$(printf '%s' "$line_status" | sed -n 's/^.*ELAPSED=\([^ ]*\).*/\1/p')"
  if printf '%s\n' "$out" | grep -q '^ROWS '; then
    ROWS_S="$(printf '%s\n' "$out" | awk '/^ROWS /{print $2}')"
    ROWS_D="$(printf '%s\n' "$out" | awk '/^ROWS /{print $3}')"
  fi
  log "poll $i: exit=$EXIT_V status=$STATUS_LAST stage=$STAGE_LAST elapsed=${ELAPSED}s rows=${ROWS_S}/${ROWS_D}"
  printf '%s\n' "$out" | grep '^LOG ' | sed 's/^LOG //' >>"$RUNLOG" || true

  # gate milestone: full_run is only entered after divide/smoke/NR-sanity/runtime gates pass
  if [ "$GATE_PUSHED" -eq 0 ]; then
    case "$STAGE_LAST" in
      full_run|validate|finished)
        GATE_PUSHED=1
        push "DR Colab milestone: vLLM server + divide/smoke/NR-sanity/runtime gates passed (stage=$STAGE_LAST, elapsed=$ELAPSED s); full 612-row run started"
        ;;
    esac
  fi

  now=$(date +%s)
  if [ "$EXIT_V" = "running" ] && [ $((now - LAST_HEARTBEAT)) -ge 1200 ]; then
    LAST_HEARTBEAT=$now
    push "DR Colab milestone: running ${STAGE_LAST}, rows sparse=${ROWS_S}/306 dense=${ROWS_D}/306, elapsed=${ELAPSED}s, $(echo $(( (now - STARTED) / 60 )) )m wall"
  fi
  [ "$EXIT_V" = "running" ] || break
done

if [ "$EXIT_V" = "running" ]; then
  log "poll budget exhausted; downloading whatever exists"
fi
log "kernel exit: $EXIT_V status=$STATUS_LAST stage=$STAGE_LAST rows=${ROWS_S}/${ROWS_D}"

# ---------------------------------------------------------------- 7. download outputs
pull() {
  timeout 300 colab exec -s "$S" --timeout 120 <<'PY' >>"$RUNLOG" 2>&1 || true
import subprocess
script = r'''
set -e
rm -rf /content/dr_out && mkdir -p /content/dr_out
# /kaggle/working holds results/kaggle/dr, status.json, DONE, manifest_out.json, gates and logs
cp -a /kaggle/working/. /content/dr_out/ 2>/dev/null || true
cp /content/dr_kernel.log /content/dr_out/ 2>/dev/null || true
tar czf /content/dr_results.tgz -C /content/dr_out .
du -h /content/dr_results.tgz
'''
r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=240)
print((r.stdout or '') + (r.stderr or ''))
raise SystemExit(r.returncode)
PY
  rm -f "$DL_TGZ"
  colab download -s "$S" /content/dr_results.tgz "$DL_TGZ" >>"$RUNLOG" 2>&1 || return 1
  rm -rf "$OUT" && mkdir -p "$OUT"
  tar xzf "$DL_TGZ" -C "$OUT"
}
if ! pull; then
  log "output download failed"
  push "DR Colab BLOCKED: kernel finished (exit=$EXIT_V) but the output download from $S failed; rows on the VM sparse=$ROWS_S/306 dense=$ROWS_D/306"
  exit 6
fi

LOCAL_S=$(cat "$OUT"/results/kaggle/dr/runs/DecompositionReason_SparseRAG_k20.jsonl 2>/dev/null | wc -l || true)
LOCAL_D=$(cat "$OUT"/results/kaggle/dr/runs/DecompositionReason_DenseRAG_k20.jsonl 2>/dev/null | wc -l || true)
FINAL_STATUS="$(python3 -c "import json;print(json.load(open('$OUT/status.json')).get('status'))" 2>/dev/null || echo unknown)"
log "downloaded to $OUT: rows sparse=$LOCAL_S dense=$LOCAL_D status=$FINAL_STATUS exit=$EXIT_V"
find "$OUT" -maxdepth 2 -type f -printf '%10s  %p\n' | sort -k2 >>"$RUNLOG" || true

if [ "$LOCAL_S" = "306" ] && [ "$LOCAL_D" = "306" ] && [ "$FINAL_STATUS" = "complete" ]; then
  push "DR Colab milestone: run finished (status=complete, exit=$EXIT_V); downloaded 612/612 rows to results/kaggle/dr/raw"
else
  push "DR Colab milestone: run finished with status=$FINAL_STATUS exit=$EXIT_V; downloaded rows sparse=${LOCAL_S}/306 dense=${LOCAL_D}/306 to results/kaggle/dr/raw"
fi

echo "COLAB_RUN_DONE exit=$EXIT_V status=$FINAL_STATUS rows_sparse=$LOCAL_S rows_dense=$LOCAL_D" | tee -a "$RUNLOG"
