#!/usr/bin/env python3
"""
Convert precomputed LSD ``.npy`` caches (from ``process_lsd.py``) to chunked Zarr stores.

Writes ``<stem>.zarr`` next to each ``<stem>.npy`` (or under ``--output-dir``) as a single
root array with shape ``(10, H, W, Z)``, matching ``utils.dataloader_preLSD`` layout.

Default ``--compressor zstd`` (Blosc) typically shrinks on-disk size by several fold versus
raw ``.npy`` (e.g. ~200G npy → ~30G zarr is normal); use ``--compressor none`` if you need
uncompressed zarr for debugging or to match npy footprint.

Example:
    python tran_lsd_to_zarr.py \\
        --input-dir /mnt/shared-storage-user/ai4sdata2-share/dengjuntao/LSD_cache \\
        --jobs 2
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Sequence, Tuple

import numpy as np
import zarr


def _default_chunks(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("Expected 4D LSD (C,H,W,Z), got {}".format(shape))
    c, h, w, z = shape
    return (
        min(c, 10),
        min(h, 512),
        min(w, 512),
        min(z, 32),
    )


def _make_compressor(name: str):
    if name in ("none", ""):
        return None
    if name == "zstd":
        try:
            from numcodecs import Blosc

            return Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
        except ImportError:
            return None
    raise ValueError("Unknown --compressor {!r} (use none or zstd)".format(name))


def _copy_blocked(
    src: np.ndarray,
    dest: zarr.Array,
    chunks: Tuple[int, int, int, int],
) -> None:
    c, h, w, z = src.shape
    ch, hh, wh, zh = chunks
    for z0 in range(0, z, zh):
        z1 = min(z0 + zh, z)
        for h0 in range(0, h, hh):
            h1 = min(h0 + hh, h)
            for w0 in range(0, w, wh):
                w1 = min(w0 + wh, w)
                dest[:, h0:h1, w0:w1, z0:z1] = np.asarray(
                    src[:, h0:h1, w0:w1, z0:z1], dtype=dest.dtype
                )


def convert_one_npy(
    npy_path: str,
    zarr_path: str,
    *,
    chunk_c: Optional[int],
    chunk_h: int,
    chunk_w: int,
    chunk_z: int,
    compressor,
    overwrite: bool,
    skip_existing: bool,
) -> Tuple[str, str, Optional[str]]:
    """
    Returns (npy_path, status, error_message).
    status is 'ok', 'skip', or 'fail'.
    """
    try:
        if skip_existing and os.path.isdir(zarr_path):
            return npy_path, "skip", None
        if os.path.isdir(zarr_path):
            if overwrite:
                shutil.rmtree(zarr_path)
            else:
                return npy_path, "fail", "destination exists (use --overwrite)"

        src = np.load(npy_path, mmap_mode="r")
        shape = tuple(int(x) for x in src.shape)
        dtype = np.dtype(src.dtype)
        if len(shape) != 4:
            return npy_path, "fail", "expected 4D array, got shape {}".format(shape)

        dc = _default_chunks(shape)
        cc = min(chunk_c or dc[0], shape[0])
        hh = min(chunk_h, shape[1])
        wh = min(chunk_w, shape[2])
        zh = min(chunk_z, shape[3])
        chunks = (cc, hh, wh, zh)

        dest = zarr.open(
            zarr_path,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )
        _copy_blocked(src, dest, chunks)
        return npy_path, "ok", None
    except Exception:
        return npy_path, "fail", traceback.format_exc()


def _worker_mp(payload: dict) -> Tuple[str, str, Optional[str]]:
    payload = dict(payload)
    comp_name = payload.pop("_compressor_name", "none")
    payload["compressor"] = _make_compressor(comp_name)
    return convert_one_npy(**payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/shared-storage-user/ai4sdata2-share/dengjuntao/LSD_cache",
        help="Directory containing ``*.npy`` LSD caches",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="If set, write ``<name>.zarr`` here; otherwise next to each ``.npy``",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob under input-dir (default: *.npy)",
    )
    p.add_argument("--jobs", type=int, default=1, help="Parallel conversions (default 1)")
    p.add_argument("--overwrite", action="store_true", help="Replace existing ``.zarr`` dirs")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if destination ``.zarr`` already exists",
    )
    p.add_argument(
        "--compressor",
        type=str,
        default="zstd",
        choices=("none", "zstd"),
        help="Chunk compression (zstd uses Blosc if numcodecs is installed)",
    )
    p.add_argument("--chunk-h", type=int, default=512, help="H chunk size (capped by volume)")
    p.add_argument("--chunk-w", type=int, default=512, help="W chunk size (capped by volume)")
    p.add_argument("--chunk-z", type=int, default=32, help="Z chunk size (capped by volume)")
    p.add_argument(
        "--chunk-c",
        type=int,
        default=0,
        help="Channel chunk (0 = use all 10 channels per chunk)",
    )
    args = p.parse_args(argv)

    input_dir = os.path.abspath(args.input_dir)
    out_base = os.path.abspath(args.output_dir) if args.output_dir else ""
    pattern = os.path.join(input_dir, args.pattern)
    npy_files = sorted(glob.glob(pattern))
    if not npy_files:
        print("No files matched: {}".format(pattern))
        return 1
    if out_base:
        os.makedirs(out_base, exist_ok=True)

    compressor = _make_compressor(args.compressor)
    chunk_c = args.chunk_c if args.chunk_c > 0 else None

    tasks = []
    for npy_path in npy_files:
        stem = os.path.splitext(os.path.basename(npy_path))[0]
        if out_base:
            zarr_path = os.path.join(out_base, stem + ".zarr")
        else:
            zarr_path = os.path.join(os.path.dirname(npy_path), stem + ".zarr")
        tasks.append(
            {
                "npy_path": npy_path,
                "zarr_path": zarr_path,
                "chunk_c": chunk_c,
                "chunk_h": args.chunk_h,
                "chunk_w": args.chunk_w,
                "chunk_z": args.chunk_z,
                "compressor": compressor,
                "overwrite": args.overwrite,
                "skip_existing": args.skip_existing,
            }
        )

    ok = skip = fail = 0
    if args.jobs <= 1:
        for t in tasks:
            path, status, err = convert_one_npy(**t)
            if status == "ok":
                ok += 1
                print("OK {}".format(path))
            elif status == "skip":
                skip += 1
                print("SKIP {}".format(path))
            else:
                fail += 1
                print("FAIL {}".format(path))
                if err:
                    print(err)
    else:
        comp_name = args.compressor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {}
            for t in tasks:
                payload = dict(t)
                payload.pop("compressor", None)
                payload["_compressor_name"] = comp_name
                futures[ex.submit(_worker_mp, payload)] = t["npy_path"]
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    _, status, err = fut.result()
                except Exception:
                    fail += 1
                    print("FAIL {} (worker crash)".format(path))
                    print(traceback.format_exc())
                    continue
                if status == "ok":
                    ok += 1
                    print("OK {}".format(path))
                elif status == "skip":
                    skip += 1
                    print("SKIP {}".format(path))
                else:
                    fail += 1
                    print("FAIL {}".format(path))
                    if err:
                        print(err)

    print("Done: {} converted, {} skipped, {} failed".format(ok, skip, fail))
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
