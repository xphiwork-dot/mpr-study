import glob, hashlib, json, os, subprocess, sys

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def walk(root, depth=0, maxdepth=3):
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except Exception as e:
        return [{"path": root, "error": repr(e)}]
    res = []
    for e in entries:
        try:
            isdir = e.is_dir(follow_symlinks=False)
            size = e.stat(follow_symlinks=False).st_size
        except Exception as ex:
            isdir, size = None, -1
            e = e
        res.append({"path": e.path, "dir": isdir, "size": size})
        if isdir and depth < maxdepth:
            res.extend(walk(e.path, depth + 1, maxdepth))
    return res

out = {}
out["python"] = sys.version
out["cwd"] = os.getcwd()
out["kaggle_listdir"] = sorted(os.listdir("/kaggle")) if os.path.isdir("/kaggle") else None
out["input_exists"] = os.path.isdir("/kaggle/input")
out["input_listdir"] = sorted(os.listdir("/kaggle/input")) if out["input_exists"] else None
out["input_tree"] = walk("/kaggle/input") if out["input_exists"] else []
for name in ("manifest.json", "dr.py", "s306.json", "settings.json", "score.py", "lib.py",
             "baseline_nr_SparseRAG_k20.jsonl", "baseline_nr_DenseRAG_k20.jsonl",
             "question_answer_0.json"):
    out.setdefault("found", {})[name] = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))[:12]
mf = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))
if mf:
    out["manifest_sha256"] = {p: sha(p) for p in mf}
    try:
        m = json.load(open(mf[0], encoding="utf-8"))
        out["manifest_keys"] = sorted(m)
        out["manifest_file_entries"] = sorted(m.get("files", {}))
    except Exception as e:
        out["manifest_error"] = repr(e)
out["env"] = {k: v for k, v in os.environ.items()
              if (k.startswith("KAGGLE") and not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASS")))
              or k.startswith(("MPR_", "T4_"))}
try:
    out["df"] = subprocess.run(["df", "-h", "/kaggle/input", "/kaggle/working"], capture_output=True,
                               text=True).stdout
except Exception as e:
    out["df_error"] = repr(e)
out["input_mount_lines"] = [l for l in open("/proc/mounts").read().splitlines() if "kaggle" in l][:40]
print(json.dumps(out, indent=1))
with open("/kaggle/working/mountprobe.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
