#!/usr/bin/env python3
"""Portable, standard-library integrity check for the Dynamic MSGT result bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def check_counts() -> None:
    primary = {"U2K": "rmse", "K2U_proxy": "rmse", "KERAAL": "brier", "irds": "brier", "rehab24_6": "brier"}
    _, rows = table(HERE / "data/primary/five_settings_all_models.csv")
    paired: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        if row["metric"] != primary[row["task"]]:
            continue
        key = row["task"], row["model"], row["method"]
        paired.setdefault(key, {})[row["rule"]] = float(row["value"])
        assert not row["source"].startswith("/"), "non-portable absolute source path"
    decisions = [(k, v["dynamic"] < v["adapted"]) for k, v in paired.items() if {"dynamic", "adapted"} <= v.keys()]
    assert len(decisions) == 28 and sum(w for _, w in decisions) == 25
    other = [(k, w) for k, w in decisions if k[1] != "msgt"]
    assert len(other) == 18 and sum(w for _, w in other) == 17

    _, rows = table(HERE / "data/primary/fair_u2k_irds_summary.csv")
    fair: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        metric = "rmse" if row["dataset"] == "U2K" else "brier"
        if row["metric"] != metric or row["value"] in {"", "NA"}:
            continue
        key = row["dataset"], row["model"], row["budget"], row["method"]
        fair.setdefault(key, {})[row["rule"]] = float(row["value"])
    assert len(fair) == 24
    assert all(v["dynamic"] < v["direct"] for v in fair.values())
    assert sum(k[1] == "msgt" for k in fair) == 12
    assert sum(k[1] == "ri_gcn" for k in fair) == 12

    _, rows = table(HERE / "data/exploratory/new_weight_summary.csv")
    exploratory: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        if row["value"] in {"", "NA"}:
            continue
        key = row["dataset"], row["model"], row["budget"], row["method"]
        exploratory.setdefault(key, {})[row["rule"]] = float(row["value"])
    assert len(exploratory) == 24
    for reference, expected in {"dynamic": (21, 0, 3), "direct": (22, 0, 2), "fixed_half": (8, 8, 8)}.items():
        outcomes = [v["risk_gradient_shrinkage"] - v[reference] for v in exploratory.values()]
        actual = sum(d < 0 for d in outcomes), sum(d == 0 for d in outcomes), sum(d > 0 for d in outcomes)
        assert actual == expected, (reference, actual, expected)

    claims = json.loads((HERE / "CLAIMS_INDEX.json").read_text())
    assert claims["five_settings_dynamic_vs_same_model_direct"]["wins"] == 25
    assert claims["five_settings_other_models_dynamic_vs_direct"]["wins"] == 17
    assert claims["fair_u2k_irds_dynamic_vs_same_model_direct"]["by_model_wins"] == {"msgt": 12, "ri_gcn": 12}


def main() -> None:
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    assert manifest["repository_version"] == "1.0.0"
    listed = set()
    for entry in manifest["files"]:
        path = HERE / entry["path"]
        assert path.is_file(), f"missing: {entry['path']}"
        assert sha(path) == entry["sha256"], f"SHA mismatch: {entry['path']}"
        assert path.stat().st_size == entry["bytes"], f"size mismatch: {entry['path']}"
        if path.suffix == ".csv":
            columns, rows = table(path)
            assert columns == entry["columns"], f"columns mismatch: {entry['path']}"
            assert len(rows) == entry["rows"], f"row count mismatch: {entry['path']}"
        listed.add(entry["path"])
    actual_data = {p.relative_to(HERE).as_posix() for p in (HERE / "data").rglob("*") if p.is_file()}
    assert listed == actual_data, f"unmanifested data: {sorted(actual_data - listed)}"

    checksum_rows = (HERE / "SHA256SUMS").read_text().splitlines()
    checked = set()
    for line in checksum_rows:
        digest, relative = line.split("  ", 1)
        path = HERE / relative
        assert path.is_file() and sha(path) == digest, f"bundle checksum mismatch: {relative}"
        checked.add(relative)
    expected = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    assert checked == expected, f"checksum inventory differs: {sorted(expected ^ checked)}"
    check_counts()
    print(f"PASS: {len(listed)} data files, {len(checked)} checksummed bundle files; key results verified")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
