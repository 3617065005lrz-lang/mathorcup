"""Fetch only 400/450/550 hPa TMP/HGT GRIB messages and retain cropped arrays."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import csv
import hashlib
import json
import re
import threading
import time
import urllib.request
import numpy as np
from netCDF4 import Dataset
from download_gfs_subset import parse_inventory, extract_values

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (400, 450, 550)
DECODE_LOCK = threading.Lock()


def fetch(url, start=None, end=None):
    headers = {'User-Agent': 'MathorCup-regional-level-supplement/1.0'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if start is not None:
                    expected = f'bytes {start}-{end}/'
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Server did not honor exact byte range')
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise ValueError('Unexpected GRIB message length')
                else:
                    data = response.read(2 * 1024 * 1024)
                return data
        except Exception as exc:
            error = exc
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f'Could not retrieve {url}: {error}')


def task(row):
    cycle = row['forecast_reference_time_utc']
    lead = int(row['lead_hours'])
    key = cycle.replace(':', '').replace('-', '') + f'_{lead:03d}'
    cache = ROOT / '.work/levels_cache'
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / (key + '.npz')
    proof = cache / (key + '.json')
    if npz.exists() and proof.exists():
        return row, npz, proof
    inventory = fetch(row['inventory_url']).decode('utf-8')
    records = parse_inventory(inventory, int(row['full_grib_size_bytes']))
    selected = [r for r in records if r['short'] in ('TMP', 'HGT') and r['level'] in {f'{p} mb' for p in LEVELS}]
    if len(selected) != 6:
        raise ValueError('Expected six TMP/HGT records')
    arrays, provenance, grid = {}, [], {}
    # Adjacent HGT/TMP messages at each level share one exact HTTP request.
    blobs = {}
    for level in LEVELS:
        group = sorted([r for r in selected if r['level'] == f'{level} mb'], key=lambda r: r['start'])
        start, end = group[0]['start'], group[-1]['end']
        combined = fetch(row['grib_url'], start, end)
        for record in group:
            blobs[record['start']] = combined[record['start'] - start:record['end'] - start + 1]
    for record in selected:
        blob = blobs[record['start']]
        if blob[:4] != b'GRIB' or blob[-4:] != b'7777':
            raise ValueError('Invalid GRIB message framing')
        with DECODE_LOCK:
            values, metadata = extract_values(blob, grid)
        arrays[record['short'] + '_' + record['level'].split()[0]] = values
        provenance.append({'source_grib_url': row['grib_url'], 'inventory_url': row['inventory_url'],
                           'inventory_line': record['inventory_line'], 'byte_start': record['start'],
                           'byte_end': record['end'], 'message_sha256': hashlib.sha256(blob).hexdigest(),
                           'units': metadata['units']})
    g = next(iter(grid.values()))
    arrays['latitude'] = g['latitudes']
    arrays['longitude'] = g['longitudes']
    np.savez_compressed(npz, **arrays)
    proof.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return row, npz, proof


def write_cycle(cycle, items, output):
    items.sort(key=lambda x: int(x[0]['lead_hours']))
    if len(items) != 41:
        raise ValueError('Incomplete supplementary forecast cycle')
    ref_path = ROOT / 'P0-5_GFS/data' / f'gfs_{cycle}_f000-f120_gyirong.nc'
    with Dataset('reference.nc', memory=ref_path.read_bytes()) as source:
        target = Dataset('supplement.nc', 'w', format='NETCDF4', memory=1024 * 1024)
        try:
            for name in ('forecast_period', 'latitude', 'longitude'):
                target.createDimension(name, len(source.dimensions[name]))
            target.createDimension('pressure_level', len(LEVELS))
            for name in ('forecast_period', 'latitude', 'longitude', 'forecast_reference_time', 'valid_time'):
                old = source[name]
                v = target.createVariable(name, old.dtype, old.dimensions)
                v.setncatts(old.__dict__)
                v[...] = old[...]
            v = target.createVariable('pressure_level', 'i4', ('pressure_level',))
            v.units = 'hPa'
            v[:] = LEVELS
            for short, name, units in (('TMP', 'air_temperature_isobaric', 'K'), ('HGT', 'geopotential_height_isobaric', 'gpm')):
                v = target.createVariable(name, 'f4', ('forecast_period', 'pressure_level', 'latitude', 'longitude'), zlib=True)
                v.units = units
                for index, (_, npz, _) in enumerate(items):
                    with np.load(npz) as data:
                        if not np.array_equal(data['latitude'], source['latitude'][:]) or not np.array_equal(data['longitude'], source['longitude'][:]):
                            raise ValueError('Supplement grid does not match source grid')
                        v[index] = np.stack([data[f'{short}_{level}'] for level in LEVELS])
            target.source = 'NOAA NCEI GFS operational archive, grid 004; byte-range extraction'
            target.processing = 'Supplementary 400, 450, 550 hPa temperatures and geopotential heights; original downloaded fields untouched.'
        finally:
            buffer = target.close()
        destination = output / ref_path.name.replace('.nc', '_levels_400_450_550.nc')
        destination.write_bytes(buffer)
    records = []
    for row, _, proof in items:
        for record in json.loads(proof.read_text(encoding='utf-8')):
            records.append({'forecast_reference_time_utc': row['forecast_reference_time_utc'], 'lead_hours': int(row['lead_hours']), **record})
    destination.with_suffix('.provenance.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    rows = list(csv.DictReader((ROOT / 'P0-5_GFS/catalog/gfs_official_inventory.csv').open(encoding='utf-8-sig')))
    output = ROOT / 'P0-5_GFS/pressure_supplement'
    output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(task, row): row for row in rows}
        for future in as_completed(jobs):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({'row': jobs[future], 'error': str(exc)})
            done = len(results) + len(failures)
            if done % 40 == 0 or done == len(rows):
                print(f'Levels: {done}/{len(rows)} leads; failed={len(failures)}', flush=True)
    if failures:
        (output / 'download_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
        raise RuntimeError('Incomplete supplementary download. Rerun to reuse successful cached subsets.')
    groups = {}
    for item in results:
        cycle = datetime.strptime(item[0]['forecast_reference_time_utc'], '%Y%m%dT%H:%M:%SZ').strftime('%Y%m%d%H')
        groups.setdefault(cycle, []).append(item)
    for cycle, items in sorted(groups.items()):
        write_cycle(cycle, items, output)
    print(f'Supplementary pressure levels saved: {len(groups)} cycles', flush=True)


if __name__ == '__main__':
    main()
