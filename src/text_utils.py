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


def estimate_qam16_bit_priors_from_bytes(data: bytes) -> tuple[float, list[float]]:
    bits = bytes_to_bits(data)
    n_symbols = bits.numel() // 4
    if n_symbols == 0:
        raise ValueError("Need at least one 16-QAM symbol worth of bits (4 bits)")
    bits_4 = bits[: n_symbols * 4].reshape(n_symbols, 4).float()
    global_p = float(bits_4.mean().item())
    per_pos = [float(bits_4[:, i].mean().item()) for i in range(4)]
    return global_p, per_pos


def estimate_qam16_bit_priors_from_text_files(
    text_paths: list[str | Path],
    max_bytes_per_file: int = 0,
) -> tuple[float, list[float], int]:
    if not text_paths:
        raise ValueError("text_paths must not be empty")
    if max_bytes_per_file < 0:
        raise ValueError("max_bytes_per_file must be non-negative")

    blobs: list[bytes] = []
    for path in text_paths:
        raw = Path(path).read_bytes()
        if max_bytes_per_file > 0:
            raw = raw[:max_bytes_per_file]
        if raw:
            blobs.append(raw)
    if not blobs:
        raise ValueError("No bytes available from provided text files")

    merged = b"".join(blobs)
    global_p, per_pos = estimate_qam16_bit_priors_from_bytes(merged)
    return global_p, per_pos, len(merged)
