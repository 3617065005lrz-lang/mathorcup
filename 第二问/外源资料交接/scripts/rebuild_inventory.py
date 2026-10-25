"""Maintenance only: rebuild the local delivery inventory after reviewed changes."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for file in sorted(ROOT.rglob('*')):
        if not file.is_file(): continue
        relative = file.relative_to(ROOT)
        if any(part in ('.work', '__pycache__', 'python_deps') for part in relative.parts): continue
        if relative.as_posix().startswith('P0-4_Sentinel-1/GRD/'): continue
        if relative.as_posix() in ('file_inventory.csv', 'delivery_validation.json'): continue
        digest = hashlib.sha256()
        with file.open('rb') as handle:
            while chunk := handle.read(4 * 1024 * 1024): digest.update(chunk)
        rows.append({'relative_path': relative.as_posix(), 'size_bytes': file.stat().st_size, 'sha256': digest.hexdigest()})
    with (ROOT / 'file_inventory.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['relative_path', 'size_bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Local inventory rebuilt: {len(rows)} files. External GRD inventory unchanged.')


if __name__ == '__main__': main()
