#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .gui.controller import launch_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise hourly MILP state.")
    parser.add_argument("--folder", type=int, required=True,
                        help="Folder number (results/folder_<n>).")
    args = parser.parse_args()

    launch_viewer(folder=args.folder)


if __name__ == "__main__":
    main()


