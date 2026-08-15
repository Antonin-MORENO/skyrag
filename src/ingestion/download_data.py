"""
Download the NTSB Aviation Accident Reports (2016-2023) dataset from Zenodo.
Source: https://zenodo.org/records/17096333
License: CC-BY 4.0 (Garcia, Pik & Haluch, 2025)
"""

import requests
from pathlib import Path
from tqdm import tqdm

DATA_URL = "https://zenodo.org/records/17096333/files/final_reports_2016-23_cons_2024-12-24.csv?download=1"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ntsb_final_reports_2016_2023.csv"


def download_file(url: str, output_path: Path) -> None:

    if output_path.exists():
        print(f"File already exists : {output_path}")
        return

    print(f"Downloading from Zenodo...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"File saved : {output_path}")


if __name__ == "__main__":
    download_file(DATA_URL, OUTPUT_PATH)