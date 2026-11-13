# MR accelerator-availability evidence (sanitized)

`probes.json` records every private accelerator probe made for the MR
experiment on account `phinguynxun`: requested shape, kernel/version, zone,
observed devices and quota after the run. All probes ran CPU-only, including a
brand-new kernel id pushed after the user reported access had been fixed.

Raw Kaggle logs are not stored here because some probe output contained
per-session tokens; the JSON keeps only the device/driver/torch facts. Probe
kernels are private and remain on the account:
`mpr-mr-mount-probe`, `mpr-mr-gpu-notebook-probe`, `mpr-mr-gpu-probe2`,
`mpr-mr-gpu-fresh-2026-11-13`.

What the evidence establishes: no accelerator session was ever started for
these requests (no device, no driver, CPU-only torch, quota used 0.00 h,
`hasEverRun=false`). It does not establish why (account entitlement,
propagation delay and a platform fault remain possible). A UI-started GPU
notebook with a working `nvidia-smi` is the missing confirmation.
