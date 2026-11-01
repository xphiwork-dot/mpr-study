# Colab T4 timing log

Timestamps are UTC; appended by experiments/_shared/t4.sh.

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
- 2026-11-01T11:30:03Z session mpr-t4-s2 created (T4)
- 2026-11-01T11:31:51Z session mpr-t4-s2 already exists, reusing
- 2026-11-01T11:32:08Z job retrieve start: retrieve.py 
- 2026-11-01T11:35:34Z job retrieve finished rc=0 wall=206 s
- 2026-11-01T11:35:42Z pulled results from mpr-t4-s2
- 2026-11-01T11:39:34Z vLLM healthy after 225 s of waiting
- 2026-11-01T11:39:34Z job grid2_ignoramus start: grid.py --slice results/slices/s306.json --reason seq --max-step 5 --memories ignoramus --topk 20 --tag s2 --workers 48
- 2026-11-01T11:49:20Z job grid2_ignoramus finished rc=0 wall=586 s
- 2026-11-01T11:49:28Z pulled results from mpr-t4-s2
- 2026-11-01T11:49:28Z job grid2_oracle start: grid.py --slice results/slices/s306.json --reason seq --max-step 5 --memories oracle --topk 20 --tag s2 --workers 48
- 2026-11-01T12:00:29Z job grid2_oracle finished rc=0 wall=661 s
- 2026-11-01T12:00:47Z pulled results from mpr-t4-s2
- 2026-11-01T12:00:47Z job grid2_sparse start: grid.py --slice results/slices/s306.json --reason seq --max-step 5 --memories sparse --topk 20 --tag s2 --workers 48
- 2026-11-01T12:20:34Z job grid2_sparse finished rc=0 wall=1187 s
- 2026-11-01T12:20:42Z pulled results from mpr-t4-s2
- 2026-11-01T12:20:42Z job grid2_dense start: grid.py --slice results/slices/s306.json --reason seq --max-step 5 --memories dense --topk 20 --tag s2 --workers 48
- 2026-11-01T12:41:24Z job grid2_dense finished rc=0 wall=1242 s
- 2026-11-01T12:41:32Z pulled results from mpr-t4-s2

## Session 2 notes (written by hand after the run; the driver lines above are untouched)

- Session `mpr-t4-s2` (not `mpr-t4`, so a parallel worker could not collide) was created at 11:30:03Z, T4 granted on the first try. The
  first `up` call ended with a `TimeoutError` from the colab client while installing/unpacking on the VM (about 1.5 minutes after
  creation); the driver exited. Rerunning `session2.sh` (11:31:51Z, "already exists, reusing") passed `up` and everything after it,
  with no further failures. vLLM was already importable on this VM, so no install was needed.
- Retrieval re-ran (206 s) because the resume tarball from session 1 is not committed; the output is identical to the committed
  `retrieval/recall.csv.gz` (checked after decompression).
- Generation per memory type (wall of each job): no memory 586 s, Oracle 661 s, BM25 1,187 s, e5 1,242 s, sum 3,676 s (61 minutes).
  The planning estimate was 47 minutes (`estimates.md`): the real run was about 30% slower, mainly BM25 and e5 (estimate 16 min each,
  real about 20-21 min). The in-job rates (`s2/timing.jsonl`) were 937-1,079 prompt tokens/s, close to the throughput test.
- All four jobs ended rc=0 with 306 questions each (1,530 calls each, no failed calls); nothing needed a rerun.
- Wall clock from VM creation (11:30Z) to the last pull (12:41:32Z): 71 minutes. `session2.sh` does not call `t4.sh down`; the
  session was no longer listed when checked afterwards and `colab stop -s mpr-t4-s2` answered "not found", so none is left running.

## Session 3 (top-k sweep 3a, then step sweep 3b)

Driver lines appended by experiments/_shared/t4.sh (session `mpr-t4-s3`; set with `T4_SESSION`).

- 2026-11-01T16:10:46Z session mpr-t4-s3 created (T4)
- 2026-11-01T16:13:31Z session mpr-t4-s3 already exists, reusing
- 2026-11-01T16:14:00Z job retrieve start: retrieve.py 
- 2026-11-01T16:17:21Z job retrieve finished rc=0 wall=201 s
- 2026-11-01T16:17:29Z pulled results from mpr-t4-s3
- 2026-11-01T16:21:38Z vLLM healthy after 244 s of waiting
- 2026-11-01T16:21:38Z job grid3_k10 start: grid.py --slice results/slices/s900.json --reason naive --memories sparse,dense --topk 10 --tag s3_topk
- 2026-11-01T16:32:48Z job grid3_k10 finished rc=0 wall=670 s
- 2026-11-01T16:32:56Z pulled results from mpr-t4-s3
- 2026-11-01T16:32:56Z job grid3_k30 start: grid.py --slice results/slices/s900.json --reason naive --memories sparse,dense --topk 30 --tag s3_topk
- 2026-11-01T16:54:49Z job grid3_k30 finished rc=0 wall=1313 s
- 2026-11-01T16:54:57Z pulled results from mpr-t4-s3
- 2026-11-01T16:54:57Z job grid3_k50 start: grid.py --slice results/slices/s900.json --reason naive --memories sparse,dense --topk 50 --tag s3_topk
- 2026-11-01T17:30:06Z job grid3_k50 finished rc=0 wall=2109 s
- 2026-11-01T17:30:15Z pulled results from mpr-t4-s3
- 2026-11-01T17:31:01Z pulled results from mpr-t4-s3
- 2026-11-01T17:31:05Z session mpr-t4-s3 stopped

### Session 3a notes (written by hand after the run; the driver lines above are untouched)

- Session `mpr-t4-s3` was created by the driver at 16:10:46Z, T4 granted on the first try. As in session 2, the first `up` call ended with a
  `TimeoutError` from the colab client (about 2.5 minutes after creation, during the unpack/install step); rerunning `session3a.sh` (16:13:31Z,
  "already exists, reusing") passed `up` and everything after it. No other failure; vLLM was already importable on the VM. Retrieval re-ran (201 s) because
  the resume tarball is not committed (output identical in purpose to session 2). vLLM took 244 s to become healthy (weights download).
- Generation per run file (`s3_topk/timing.jsonl`, 900 questions, 900 calls each, no failed calls): k = 10: BM25 309 s, e5 332 s; k = 30: BM25 614 s, e5 664 s;
  k = 50: BM25 979 s, e5 1,079 s; sum 3,977 s = 66.3 minutes (estimate 58.8 minutes, `estimates.md`; real was 13% slower). Job walls: 670 s, 1,313 s, 2,109 s.
  Prompt rate in the job logs about 1,050-1,060 tokens/s (the estimate assumed 1,120-1,200).
- Wall clock from VM creation (16:10:46Z) to the stop (17:31:05Z): 80 minutes. The VM was stopped with `t4.sh down`; `colab sessions` then listed none.
- 2026-11-01T17:33:57Z colab new failed (attempt 1)
- 2026-11-01T17:36:12Z colab new failed (attempt 2)
- 2026-11-01T17:39:27Z colab new failed (attempt 3)
- 2026-11-01T17:43:43Z colab new failed (attempt 4)
- 2026-11-01T17:56:00Z colab new failed (attempt 1)
- 2026-11-01T17:58:16Z colab new failed (attempt 2)
- 2026-11-01T18:01:38Z colab new failed (attempt 3)
- 2026-11-01T18:05:54Z colab new failed (attempt 4)
- 2026-11-01T18:17:14Z colab new failed (attempt 1)
- 2026-11-01T18:19:30Z colab new failed (attempt 2)
- 2026-11-01T18:22:45Z colab new failed (attempt 3)
- 2026-11-01T18:27:02Z colab new failed (attempt 4)
- 2026-11-01T18:38:24Z colab new failed (attempt 1)
- 2026-11-01T18:40:40Z colab new failed (attempt 2)
- 2026-11-01T18:43:56Z colab new failed (attempt 3)
- 2026-11-01T18:48:12Z colab new failed (attempt 4)

### Session 3b notes (written by hand; the driver lines above are untouched)

- 3b (reasoning-step sweep, `session3b.sh`, session name `mpr-t4-s3b`) was NOT run. Every `colab new -s mpr-t4-s3b --gpu T4` call was answered with
  `ColabRequestError ... Service Unavailable` (HTTP 503 from the Colab runtime-assign endpoint): 16 attempts in 4 rounds of the driver's 4-attempt retry loop,
  17:33:57Z to 18:48:12Z (about 75 minutes), with growing sleeps inside a round and 10-minute pauses between rounds. The same account had been granted a T4
  on the first try at 16:10Z for 3a, which was stopped at 17:31Z. No session was created, so none needed stopping; `colab sessions` lists none. Not a
  login problem (the same `colab` call authenticated and reached the endpoint). No quota message was returned, so whether this is the free-tier T4 pool being empty or a usage limit after
  80 minutes of 3a VM time is unknown. Total generation for 3b: none; no result file under `results/t4/s3_steps`.
