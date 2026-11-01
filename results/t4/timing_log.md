# Colab T4 timing log

Timestamps are UTC; appended by scripts/t4/t4.sh.

- 2026-11-01T08:49:02Z session mpr-t4 already exists, reusing
- 2026-11-01T08:49:44Z job thr start: throughput.py --slice results/slices/s900.json --seconds 90
- 2026-11-01T08:56:16Z job thr finished rc=1 wall=392 s
- 2026-11-01T08:56:22Z pulled results from mpr-t4
- 2026-11-01T08:59:19Z vLLM healthy after 103 s of waiting
- 2026-11-01T08:59:32Z job thr start: throughput.py --slice results/slices/s900.json --seconds 90
- 2026-11-01T09:06:12Z job thr finished rc=1 wall=400 s
- 2026-11-01T09:06:18Z pulled results from mpr-t4
- 2026-11-01T09:09:01Z vLLM healthy after 121 s of waiting
- 2026-11-01T09:09:13Z job thr start: throughput.py --slice results/slices/s900.json --seconds 90
- 2026-11-01T09:20:13Z job thr finished rc=0 wall=660 s
- 2026-11-01T09:20:20Z pulled results from mpr-t4
- 2026-11-01T09:20:56Z job thr start: throughput.py --slice results/slices/s900.json --seconds 90
- 2026-11-01T09:30:59Z job thr finished rc=0 wall=603 s
- 2026-11-01T09:31:06Z pulled results from mpr-t4
- 2026-11-01T09:31:19Z job retrieve start: retrieve.py 
- 2026-11-01T09:34:49Z job retrieve finished rc=0 wall=210 s
- 2026-11-01T09:34:57Z pulled results from mpr-t4
- 2026-11-01T09:35:01Z job grid1 start: grid.py --slice results/slices/s900.json --reason naive --memories ignoramus,oracle,sparse,dense --topk 20 --tag s1
- 2026-11-01T09:49:03Z job grid1b start: grid.py --slice results/slices/s900.json --reason naive --memories ignoramus,oracle,sparse,dense --topk 20 --tag s1
- 2026-11-01T09:49:18Z job grid1 finished rc=1 wall=857 s
- 2026-11-01T09:49:25Z pulled results from mpr-t4
- 2026-11-01T09:57:53Z job grid1b finished rc=0 wall=530 s
- 2026-11-01T09:58:01Z pulled results from mpr-t4
- 2026-11-01T09:58:41Z job smoke_seq start: grid.py --slice results/slices/s306.json --reason seq --max-step 5 --memories ignoramus,oracle,sparse,dense --topk 20 --tag smoke_seq --limit 6
- 2026-11-01T10:00:55Z job smoke_seq finished rc=0 wall=134 s
- 2026-11-01T10:01:12Z pulled results from mpr-t4
- 2026-11-01T10:01:19Z job sft_smoke start: sft.py --model Qwen/Qwen2.5-0.5B-Instruct --clusters 5 --smoke --tag sft_smoke
- 2026-11-01T10:02:27Z job sft_smoke finished rc=1 wall=68 s
- 2026-11-01T10:02:36Z pulled results from mpr-t4
- 2026-11-01T10:05:43Z vLLM healthy after 84 s of waiting
- 2026-11-01T10:05:43Z job sft_smoke_ask start: sft.py --model Qwen/Qwen2.5-0.5B-Instruct --clusters 5 --smoke --stage ask --tag sft_smoke
- 2026-11-01T10:06:51Z job sft_smoke_ask finished rc=0 wall=68 s
- 2026-11-01T10:06:59Z pulled results from mpr-t4
- 2026-11-01T10:07:07Z job sft_smoke start: sft.py --model Qwen/Qwen2.5-0.5B-Instruct --clusters 5 --smoke --stage train --tag sft_smoke
- 2026-11-01T10:08:16Z job sft_smoke finished rc=1 wall=69 s
- 2026-11-01T10:08:24Z pulled results from mpr-t4
- 2026-11-01T10:08:40Z job sft_smoke start: sft.py --model Qwen/Qwen2.5-0.5B-Instruct --clusters 5 --smoke --stage train --tag sft_smoke
- 2026-11-01T10:09:54Z job sft_smoke finished rc=1 wall=74 s
- 2026-11-01T10:10:02Z pulled results from mpr-t4
- 2026-11-01T10:11:57Z vLLM healthy after 101 s of waiting
- 2026-11-01T10:11:57Z job sft_smoke_ask start: sft.py --model Qwen/Qwen2.5-0.5B-Instruct --clusters 5 --smoke --stage ask --tag sft_smoke
- 2026-11-01T10:13:05Z job sft_smoke_ask finished rc=0 wall=68 s
- 2026-11-01T10:13:13Z pulled results from mpr-t4
- 2026-11-01T10:13:18Z job sft_smoke start: sft.py --model Qwen/Qwen2.5-0.5B-Instruct --clusters 5 --smoke --stage train --tag sft_smoke
- 2026-11-01T10:15:31Z job sft_smoke finished rc=0 wall=133 s
- 2026-11-01T10:15:53Z pulled results from mpr-t4
- 2026-11-01T10:17:52Z vLLM healthy after 102 s of waiting
- 2026-11-01T10:17:52Z job smoke_s3a start: grid.py --slice results/slices/s306.json --reason seq --max-step 2 --memories sparse,dense --topk 50 --tag smoke_s3 --limit 4
- 2026-11-01T10:19:01Z job smoke_s3a finished rc=0 wall=69 s
- 2026-11-01T10:19:13Z pulled results from mpr-t4
- 2026-11-01T10:19:13Z job smoke_s3b start: grid.py --slice results/slices/s900.json --reason naive --memories sparse,dense --topk 10 --tag smoke_s3 --limit 4
- 2026-11-01T10:20:22Z job smoke_s3b finished rc=0 wall=69 s
- 2026-11-01T10:20:35Z pulled results from mpr-t4
- 2026-11-01T10:20:57Z pulled results from mpr-t4
- 2026-11-01T10:21:00Z session mpr-t4 stopped

## Notes (written by hand after the run; the driver lines above are untouched)

- Session `mpr-t4` was created by hand with `colab new -s mpr-t4 --gpu T4` at 08:41:50Z (took about 5 s, T4 granted on the
  first try, no waiting). It was stopped by `t4.sh down` at 10:21:00Z: 99 minutes of VM time, including debugging
  and the smoke tests of sessions 2-4.
- vLLM and its dependencies were installed by hand into the VM's system Python (`uv pip install vllm ...`, vLLM 0.32.0,
  torch 2.13) while this driver was being written; the install and the first server start (weights download included)
  finished between 08:42Z and 08:49Z. The `serve` lines above (84-121 s) are restarts with the weights cached.
- `job thr` at 08:49:44Z and 08:59:32Z (rc=1): failed. First, vLLM ran out of GPU memory because the Colab kernel had
  imported vllm and held ~3.5 GB; second, `bm25s` pulled in jax, which preallocated GPU memory. No usable output.
- `job thr` at 09:09:13Z (660 s, rc=0): finished but its numbers were inflated by vLLM's prefix cache (the test cycled
  over ~165 prompts; cache hit rate 98.5% in the vLLM log). Discarded, not in the repo.
- `job thr` at 09:20:56Z (603 s, rc=0): the valid throughput test, in `throughput/throughput.jsonl`.
- `job grid1` (09:35:01Z, rc=1): Ignoramus, Oracle and BM25 finished (900 questions each); every e5 question failed
  (bug: a user index first built for BM25 had no encoder). `job grid1b` reran the same command after the fix; it skipped
  the finished memories and ran e5 only. The failed e5 attempt is not in `s1/timing.jsonl` summaries (calls = 0 rows are skipped).
- The `sft_smoke` jobs at 10:01Z, 10:07Z and 10:08Z (rc=1) failed on three real problems, now fixed: the 0.5B model cannot
  write the two-line self-ask format (pairs now come from the 7B server), Colab's torchao 0.10 is refused by peft
  (uninstalled in `up`), and a cleanup I ran deleted the pairs before the retry.
- Clean-run estimate for session 1 from the lines above: retrieval job 210 s + vLLM start ~100 s + throughput test 603 s +
  grid (900 questions x 4 memories) 1,288 s of generation (sum of `s1/timing.jsonl` wall_s) = 2,201 s, about 37 minutes, plus
  pulls and the one-minute polling granularity (about 40-45 minutes), after the one-time install. The real wall clock from
  VM creation to the end of the grid (09:58Z) was 76 minutes because of the debugging described above.
