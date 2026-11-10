"""Shared helpers: dataset loading. Stdlib only."""
import json
import os
import sys

DEFAULT_DATA_DIR = os.path.join("data", "unz", "mpr_dataset", "question_answer")


def data_file(data_dir, user):
    return os.path.join(data_dir, f"question_answer_{user}.json")


def load_user(data_dir, user):
    """Return {'message': {id: text}, 'qa': {hop: {rc_id: {...}}}} for one user."""
    path = data_file(data_dir, user)
    if not os.path.exists(path):
        sys.exit(f"dataset file not found: {path} (run ./setup.sh first)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hops(data):
    return sorted(data["qa"], key=int)
