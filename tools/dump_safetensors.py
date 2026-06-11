#!/usr/bin/env python3
"""Dump safetensors headers without loading tensor data.

The offsets printed here are relative to the start of the safetensors data
payload, which is exactly what we want when mapping one copied GPU block.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


@dataclass(frozen=True)
class TensorInfo:
    file: Path
    name: str
    dtype: str
    shape: list[int]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def layer(self) -> int | None:
        match = LAYER_RE.match(self.name)
        return int(match.group(1)) if match else None

    @property
    def suffix(self) -> str:
        match = LAYER_RE.match(self.name)
        return match.group(2) if match else self.name


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    header_len: int
    file_size: int
    payload_size: int
    tensors: list[TensorInfo]
    metadata: dict[str, Any]


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def shape_str(shape: list[int]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def tensor_element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def read_safetensors_header(path: Path) -> ShardInfo:
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: file is too small to contain safetensors header length")

        header_len = struct.unpack("<Q", prefix)[0]
        header_bytes = file.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")

    header = json.loads(header_bytes)
    metadata = header.get("__metadata__", {})
    tensors: list[TensorInfo] = []
    payload_size = 0

    for name, value in header.items():
        if name == "__metadata__":
            continue

        dtype = value["dtype"]
        shape = [int(dim) for dim in value["shape"]]
        begin, end = [int(offset) for offset in value["data_offsets"]]
        if begin < 0 or end < begin:
            raise ValueError(f"{path}: invalid offsets for {name}: {begin}, {end}")

        expected_bytes = tensor_element_count(shape) * DTYPE_SIZES.get(dtype, -1)
        actual_bytes = end - begin
        if expected_bytes >= 0 and expected_bytes != actual_bytes:
            raise ValueError(
                f"{path}: byte size mismatch for {name}: "
                f"shape*dtype={expected_bytes}, offsets={actual_bytes}"
            )

        payload_size = max(payload_size, end)
        tensors.append(TensorInfo(path, name, dtype, shape, begin, end))

    file_size = path.stat().st_size
    expected_file_size = 8 + header_len + payload_size
    if file_size != expected_file_size:
        print(
            f"warning: {path} size is {format_bytes(file_size)}, "
            f"but header implies {format_bytes(expected_file_size)}"
        )

    tensors.sort(key=lambda tensor: (tensor.begin, tensor.name))
    return ShardInfo(path, header_len, file_size, payload_size, tensors, metadata)


def collect_safetensors_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".safetensors":
        return [path]

    if path.is_file() and path.name.endswith(".safetensors.index.json"):
        index = json.loads(path.read_text())
        weight_map = index.get("weight_map", {})
        return sorted({path.parent / shard for shard in weight_map.values()})

    if path.is_dir():
        direct = sorted(path.glob("*.safetensors"))
        if direct:
            return direct
        return sorted(path.rglob("*.safetensors"))

    raise ValueError(f"{path}: expected a .safetensors file, an index json, or a directory")


def print_summary(shards: list[ShardInfo]) -> None:
    total_payload = sum(shard.payload_size for shard in shards)
    total_file_size = sum(shard.file_size for shard in shards)
    all_tensors = [tensor for shard in shards for tensor in shard.tensors]
    dtypes = sorted({tensor.dtype for tensor in all_tensors})
    layers = sorted({tensor.layer for tensor in all_tensors if tensor.layer is not None})
    non_layer = [tensor for tensor in all_tensors if tensor.layer is None]

    print("== Model summary ==")
    print(f"shards: {len(shards)}")
    print(f"tensors: {len(all_tensors)}")
    print(f"dtypes: {', '.join(dtypes) if dtypes else '(none)'}")
    print(f"layer range: {layers[0]}..{layers[-1]} ({len(layers)} layers)" if layers else "layer range: none")
    print(f"non-layer tensors: {len(non_layer)}")
    print(f"total payload bytes: {total_payload} ({format_bytes(total_payload)})")
    print(f"total file bytes: {total_file_size} ({format_bytes(total_file_size)})")
    print()

    print("== Shards ==")
    for shard_index, shard in enumerate(shards):
        print(
            f"[{shard_index:02d}] {shard.path} | "
            f"header={format_bytes(shard.header_len)} payload={format_bytes(shard.payload_size)} "
            f"tensors={len(shard.tensors)}"
        )
    print()

    print("== Non-layer tensors ==")
    for tensor in non_layer:
        print(
            f"{tensor.name} | dtype={tensor.dtype} shape={shape_str(tensor.shape)} "
            f"bytes={format_bytes(tensor.nbytes)} file={tensor.file.name} offsets=[{tensor.begin}, {tensor.end})"
        )
    print()


def print_tensors(shards: list[ShardInfo], name_filter: re.Pattern[str] | None, limit: int | None) -> None:
    tensors = [tensor for shard in shards for tensor in shard.tensors]
    if name_filter is not None:
        tensors = [tensor for tensor in tensors if name_filter.search(tensor.name)]

    tensors.sort(key=lambda tensor: (tensor.layer if tensor.layer is not None else -1, tensor.name))
    if limit is not None:
        tensors = tensors[:limit]

    print("== Tensors ==")
    for tensor in tensors:
        layer_text = f"layer={tensor.layer}" if tensor.layer is not None else "layer=-"
        print(
            f"{tensor.name} | {layer_text} suffix={tensor.suffix} dtype={tensor.dtype} "
            f"shape={shape_str(tensor.shape)} bytes={format_bytes(tensor.nbytes)} "
            f"file={tensor.file.name} offsets=[{tensor.begin}, {tensor.end})"
        )
    print()


def print_layer_inventory(shards: list[ShardInfo]) -> None:
    tensors = [tensor for shard in shards for tensor in shard.tensors if tensor.layer is not None]
    by_layer: dict[int, set[str]] = {}
    for tensor in tensors:
        assert tensor.layer is not None
        by_layer.setdefault(tensor.layer, set()).add(tensor.suffix)

    if not by_layer:
        return

    common_suffixes = set.intersection(*(suffixes for suffixes in by_layer.values()))
    all_suffixes = set.union(*(suffixes for suffixes in by_layer.values()))

    print("== Layer inventory ==")
    print(f"common per-layer tensors ({len(common_suffixes)}):")
    for suffix in sorted(common_suffixes):
        print(f"  {suffix}")

    missing_by_layer = {
        layer: sorted(all_suffixes - suffixes)
        for layer, suffixes in sorted(by_layer.items())
        if all_suffixes - suffixes
    }
    if missing_by_layer:
        print("layers with missing tensors:")
        for layer, missing in missing_by_layer.items():
            print(f"  layer {layer}: {', '.join(missing)}")
    else:
        print("all discovered layers have the same tensor suffix set")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="safetensors file, model.safetensors.index.json, or model directory")
    parser.add_argument("--filter", help="regex for tensor names to print")
    parser.add_argument("--limit", type=int, help="maximum number of tensor rows to print")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = collect_safetensors_paths(args.path)
    if not paths:
        raise ValueError(f"{args.path}: no safetensors files found")

    shards = [read_safetensors_header(path) for path in paths]
    name_filter = re.compile(args.filter) if args.filter else None

    print_summary(shards)
    print_layer_inventory(shards)
    print_tensors(shards, name_filter, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
