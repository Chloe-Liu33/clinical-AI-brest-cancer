"""Download METABRIC raw files from cBioPortal Datahub (GitHub LFS).

Public-tier METABRIC: clinical + mRNA microarray, no credentialing required.
Re-runnable: existing files are skipped unless --force.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

BASE = ("https://media.githubusercontent.com/media/"
        "cBioPortal/datahub/master/public/brca_metabric")

FILES = [
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_mrna_illumina_microarray.txt",
    # Add these if you want multi-omics later:
    # "data_cna.txt",
    # "data_mutations.txt",
]


def download(url: str, dst: str) -> None:
    sys.stdout.write(f"  -> {os.path.basename(dst)} ... ")
    sys.stdout.flush()
    urllib.request.urlretrieve(url, dst)
    print(f"{os.path.getsize(dst):,} bytes")


def main(out_dir: str, force: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading METABRIC raw files into {out_dir}")
    for name in FILES:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and not force:
            print(f"  [skip] {name} already present "
                  f"({os.path.getsize(dst):,} bytes)")
            continue
        download(f"{BASE}/{name}", dst)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="raw")
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
