#!/usr/bin/env python3
"""Normalize figure filenames by replacing spaces with underscores.

Creates copies of PNG files in `outputs/figures/` with spaces replaced by
underscores (leaves originals intact). Prints actions taken.
"""
from pathlib import Path
import shutil

FIG_DIR = Path('outputs') / 'figures'

def main():
    if not FIG_DIR.exists():
        print('No figures directory found at', FIG_DIR)
        return
    pngs = list(FIG_DIR.glob('*.png'))
    if not pngs:
        print('No PNG files found in', FIG_DIR)
        return
    for p in pngs:
        if ' ' in p.name:
            new_name = p.name.replace(' ', '_')
            dest = p.with_name(new_name)
            if not dest.exists():
                shutil.copy(p, dest)
                print(f'Copied: {p.name} -> {dest.name}')
            else:
                print(f'Already exists: {dest.name}')

if __name__ == '__main__':
    main()
