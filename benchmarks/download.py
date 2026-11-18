"""Download missing pins, then verify every source. Offline mode only verifies."""

import argparse

from .data import fetch_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    manifest = fetch_sources(offline=parser.parse_args().offline)
    print(f"Verified {len(manifest['files'])} pinned files.")
