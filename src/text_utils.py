from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from pathlib import Path

import torch


def bytes_to_bits(data: bytes) -> torch.Tensor:
    vals = torch.tensor(list(data), dtype=torch.uint8)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    bits = ((vals.unsqueeze(1) >> shifts) & 1).reshape(-1).long()
    return bits


def bits_to_bytes(bits: torch.Tensor) -> bytes:
    bits = bits.reshape(-1).long()
    n = bits.numel() // 8
    if n == 0:
        return b""
    bits = bits[: n * 8].reshape(n, 8).to(torch.uint8)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    vals = torch.sum(bits << shifts, dim=1).to(torch.uint8)
    return bytes(vals.tolist())


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def assert_disjoint_test_file(test_path: str | Path, train_paths: list[str | Path]) -> None:
    test_hash = sha256_file(test_path)
    train_hashes = {sha256_file(p) for p in train_paths}
    if test_hash in train_hashes:
        raise ValueError("Data leakage detected: test .txt is identical to one of the train texts")


def byte_error_rate(ref: bytes, hyp: bytes) -> float:
    n = min(len(ref), len(hyp))
    if n == 0:
        return 0.0
    mismatches = sum(1 for i in range(n) if ref[i] != hyp[i])
    mismatches += abs(len(ref) - len(hyp))
    return mismatches / max(len(ref), 1)


def char_mismatch_rate(ref_text: str, hyp_text: str) -> float:
    if not ref_text and not hyp_text:
        return 0.0
    # 1 - sequence similarity ratio is a stable approximation for text mismatch.
    return 1.0 - SequenceMatcher(a=ref_text, b=hyp_text).ratio()
