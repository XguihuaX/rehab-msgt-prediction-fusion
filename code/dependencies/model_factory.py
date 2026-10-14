#!/usr/bin/env python3
"""Frozen model factory for the three formal classification systems."""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch

CANON_TREE = ((0, 1, 17, 2), (0, 11, 12, 13), (0, 14, 15, 16), (17, 3, 4, 5, 6), (17, 7, 8, 9, 10))


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def adjacency() -> np.ndarray:
    out = np.zeros((18, 18), dtype=np.float32)
    for chain in CANON_TREE:
        for parent, child in zip(chain[:-1], chain[1:]):
            out[parent, child] = out[child, parent] = 1.0
    return out


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def make_model(name: str, code_dir: Path, seed: int) -> torch.nn.Module:
    set_seed(seed)
    if name == "msgt":
        module = import_file("model_snapshot_frozen", code_dir / "model_snapshot_frozen.py")
        set_seed(seed)
        model = module.Net(18, adjacency(), cin=6, multiscale=True, transformer=True, adaptA=False)
        if model.g1.in_channels != 6 or not model.multiscale or not model.transformer:
            raise RuntimeError("Full MSGT contract failed")
        return model
    if name == "stgcn_rehab":
        # The class is imported verbatim from the frozen authoritative runner.
        module = import_file("run_external_eval_source_frozen", code_dir / "run_external_eval_source_frozen.py")
        set_seed(seed)
        return module.STGCNRehab2022Reimplementation()
    if name == "ctrgcn":
        module = import_file("ctrgcn_2021_regression", code_dir / "ctrgcn_2021_regression.py")
        set_seed(seed)
        return module.CTRGCN2021Regression(in_channels=6, joints=18)
    raise ValueError(name)
