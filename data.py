#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

###############################################################################
# Helpers                                                                     #
###############################################################################

def _parse_coords(raw: str) -> List[Tuple[float, float, str, str]]:
    """Convert a single *raw* string (lon,lat pairs) to a list of tuples.

    ``raw`` may use either spaces or semicolons to separate pairs, e.g.::

        "-15.6,27.3 -16.0,27.9"  or  "-15.6,27.3;-16.0,27.9"
    """
    if not raw:
        raise argparse.ArgumentTypeError("--coords string is empty.")

    pairs = raw.replace(";", " ").split()
    coords: List[Tuple[float, float, str, str]] = []
    for pair in pairs:
        try:
            lon_str, lat_str = pair.split(",")
            lon, lat = float(lon_str), float(lat_str)
            coords.append((lon, lat, lon_str, lat_str))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid coordinate '{pair}'. Use lon,lat (comma‑separated)."
            ) from exc
    if not coords:
        raise argparse.ArgumentTypeError("No valid coordinates parsed from --coords.")
    return coords


def _sanitize(raw: str) -> str:
    """Make *raw* safe for filenames without losing precision or sign."""
    sign = "p" if not raw.startswith("-") else "m"
    return sign + raw.lstrip("+-").replace(".", "_")


def _download_one(url: str, dest: Path, token: str, timeout: int = 20) -> None:
    if dest.exists():
        return
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_all(urls: List[str], cache_dir: Path, token: str, workers: int = 8) -> List[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dests = [cache_dir / Path(u).name for u in urls]
    to_dl = [(u, d) for u, d in zip(urls, dests) if not d.exists()]
    if to_dl:
        logging.info("Downloading %d files …", len(to_dl))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_download_one, u, d, token): d for u, d in to_dl}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
                fut.result()
    else:
        logging.info("All %d files already present in cache.", len(urls))
    return dests


def _csv_name(dataset: str, var: str, lon_raw: str, lat_raw: str, out_dir: Path) -> Path:
    lon_tok, lat_tok = _sanitize(lon_raw), _sanitize(lat_raw)
    return out_dir / f"{dataset}_{var}_{lon_tok}_{lat_tok}.csv"


def extract_point_csv(
    nc_files: List[Path],
    variable: str,
    lon: float,
    lat: float,
    lon_raw: str,
    lat_raw: str,
    dataset: str,
    out_dir: Path,
) -> None:
    out_path = _csv_name(dataset, variable, lon_raw, lat_raw, out_dir)
    if out_path.exists():
        logging.info("%s exists – skipping.", out_path.name)
        return
    
    if nc_files[0].suffix.lower() == ".hdf5": # Fix for GPM_3IMERGHH
        ds = xr.open_mfdataset(
            [str(p) for p in nc_files],
            group="Grid",
            combine="nested",          
            concat_dim="time",
            engine="h5netcdf",
            chunks={},                 
            parallel=False,            
        )
    else:
        ds = xr.open_mfdataset(
            [str(p) for p in nc_files],
            combine="by_coords",
            chunks={"time": 96},
            preprocess=lambda d: d[[variable]],
            engine="netcdf4",
            parallel=False,
        )

    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in dataset.")

    series = ds[variable].sel(lon=lon, lat=lat, method="nearest").load()
    df = pd.DataFrame({"time": series.time.values, "value": series.values.astype("float32")})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info("Saved %s (%d rows).", out_path.name, len(df))


###############################################################################
# CLI                                                                         #
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download NetCDF list and extract point time‑series to CSV.")

    p.add_argument("list_file", type=Path, help="Text file containing HTTP URLs (one per line).")
    p.add_argument("output_dir", type=Path, help="Destination directory (a '.cache' sub‑dir will be created).")

    p.add_argument("--variables", "-v", required=True, nargs="+", metavar="VAR", help="Variable name(s) to extract.")
    p.add_argument(
        "--coords",
        "-c",
        required=True,
        metavar="COORDS",
        type=str,
        help="String with lon,lat pairs separated by spaces or semicolons. Enclose in quotes if it contains spaces.",
    )
    p.add_argument("--token", "-t", default=os.getenv("EARTHDATA_TOKEN"), help="Earthdata bearer token.")
    p.add_argument("--workers", "-w", type=int, default=8, help="Concurrent download threads (default 8).")
    p.add_argument("--log-level", default="INFO", choices=_LOG_LEVELS, help="Logging verbosity (default INFO).")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=_LOG_LEVELS[args.log_level],
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.token:
        logging.error("No Earthdata token provided – use --token or set EARTHDATA_TOKEN.")
        sys.exit(1)

    url_list = [u.strip() for u in args.list_file.read_text().splitlines() if u.strip() and ".pdf" not in u]
    if not url_list:
        logging.error("%s is empty.", args.list_file)
        sys.exit(1)

    cache_dir = args.output_dir / ".cache"
    nc_files = download_all(url_list, cache_dir, args.token, workers=args.workers)

    coords = _parse_coords(args.coords)
    dataset = args.list_file.stem

    for var in args.variables:
        for lon, lat, lon_raw, lat_raw in coords:
            try:
                extract_point_csv(nc_files, var, lon, lat, lon_raw, lat_raw, dataset, args.output_dir)
            except Exception as exc:
                logging.error("Failed for %s @ (%s,%s): %s", var, lon_raw, lat_raw, exc)

    logging.info("All done.")


if __name__ == "__main__":
    main()
