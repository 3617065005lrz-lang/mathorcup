"""Verify or import the original seven GRD parts, without making a merged ZIP."""
import argparse
import bisect
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class Parts(io.RawIOBase):
    def __init__(self, directory):
        self.files = [(directory / f'帮助_GRD_分卷.zip.{n:03}').open('rb') for n in range(1, 8)]
        self.ends, self.pos = [0], 0
        for file in self.files:
            self.ends.append(self.ends[-1] + file.seek(0, 2))
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, n, w=0):
        self.pos = n if w == 0 else self.pos + n if w == 1 else self.ends[-1] + n
        if self.pos < 0: raise ValueError('Negative position')
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.ends[-1] - self.pos
        chunks = []
        while n and self.pos < self.ends[-1]:
            index = bisect.bisect_right(self.ends, self.pos) - 1
            file = self.files[index]
            file.seek(self.pos - self.ends[index])
            data = file.read(min(n, self.ends[index + 1] - self.pos))
            if not data: raise EOFError('Incomplete part')
            self.pos += len(data)
            n -= len(data)
            chunks.append(data)
        return b''.join(chunks)
    def close(self):
        for file in getattr(self, 'files', []): file.close()
        super().close()


def digest(file):
    result = hashlib.sha256()
    with file.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024): result.update(chunk)
    return result.hexdigest()


def verify(directory, import_grd=False):
    expected = json.loads((ROOT / '分卷独立校验.json').read_text(encoding='utf-8'))
    merged = hashlib.sha256()
    result = {'parts': [], 'archives': []}
    for row in expected['parts']:
        file = directory / row['name']
        sha = hashlib.sha256()
        with file.open('rb') as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                sha.update(chunk)
                merged.update(chunk)
        ok = file.stat().st_size == row['bytes'] and sha.hexdigest() == row['sha256']
        result['parts'].append({'name': file.name, 'ok': ok})
        if not ok: raise ValueError('Part verification failed: ' + file.name)
    result['merged_sha256_ok'] = merged.hexdigest() == expected['merged_sha256']
    if not result['merged_sha256_ok']: raise ValueError('Merged stream hash mismatch')
    inventory = list(csv.DictReader((ROOT / 'external_grd_inventory.csv').open(encoding='utf-8-sig')))
    by_name = {Path(row['relative_path']).name: row for row in inventory}
    with Parts(directory) as source, zipfile.ZipFile(source) as archive:
        members = [item for item in archive.infolist() if item.filename.endswith('.SAFE.zip')]
        if len(members) != 4: raise ValueError('Expected four GRD archives')
        for member in members:
            name = Path(member.filename).name
            expected_file = by_name[name]
            sha = hashlib.sha256()
            with archive.open(member) as handle:
                while chunk := handle.read(8 * 1024 * 1024): sha.update(chunk)
            ok = sha.hexdigest() == expected_file['sha256'] and member.file_size == int(expected_file['size_bytes'])
            if not ok: raise ValueError('GRD hash mismatch: ' + name)
            # A matching SHA256 binds this to the separately CRC-validated received ZIP.
            if import_grd:
                destination = ROOT / 'P0-4_Sentinel-1/GRD' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if digest(destination) != expected_file['sha256']:
                        raise FileExistsError('Existing GRD differs; refusing overwrite: ' + str(destination))
                else:
                    temporary = destination.with_suffix('.zip.part')
                    with archive.open(member) as handle, temporary.open('wb') as out:
                        shutil.copyfileobj(handle, out, 8 * 1024 * 1024)
                    if digest(temporary) != expected_file['sha256']:
                        raise ValueError('Imported copy hash mismatch')
                    temporary.replace(destination)
            result['archives'].append({'name': name, 'sha256_ok': ok})
    result['passed'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts-dir', type=Path, required=True)
    parser.add_argument('--import-grd', action='store_true', help='Copy four archives into the expected GRD directory; uses another 6.9 GB.')
    args = parser.parse_args()
    result = verify(args.parts_dir, args.import_grd)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
