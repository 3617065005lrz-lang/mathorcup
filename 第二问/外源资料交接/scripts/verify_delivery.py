"""Offline delivery checks; --parts-dir additionally verifies the seven external parts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import rasterio
from grd_parts import verify as verify_parts, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts-dir', type=Path)
    args = parser.parse_args()
    if args.parts_dir is None:
        for candidate in (ROOT, ROOT.parent):
            if all((candidate / f'帮助_GRD_分卷.zip.{index:03}').exists() for index in range(1, 8)):
                args.parts_dir = candidate
                break
    report = {'files': {}, 'gfs': [], 'dem': [], 'grd': {}, 'errors': []}
    try:
        expected = list(csv.DictReader((ROOT / 'file_inventory.csv').open(encoding='utf-8-sig')))
        for row in expected:
            file = ROOT / row['relative_path']
            ok = file.is_file() and file.stat().st_size == int(row['size_bytes']) and digest(file) == row['sha256']
            report['files'][row['relative_path']] = ok
            if not ok: report['errors'].append('File hash/size mismatch: ' + row['relative_path'])
        files = sorted((ROOT / 'P0-5_GFS/analysis_ready').glob('*.nc'))
        if len(files) != 20: raise ValueError('Expected 20 derived GFS files')
        for file in files:
            with Dataset('check.nc', memory=file.read_bytes()) as nc:
                lead = np.asarray(nc['forecast_period'][:])
                if not np.array_equal(lead, np.arange(0, 121, 3)): raise ValueError('Incorrect forecast leads')
                if not np.all(nc['valid_time'][:] == int(nc['forecast_reference_time'][...]) + lead * 3600): raise ValueError('Incorrect valid times')
                for raw, derived, bounds, mean in (
                    ('total_precipitation_accumulation', 'precipitation_3h', 'precipitation_source_period_bounds', False),
                    ('downward_shortwave_radiation_flux', 'shortwave_mean_3h', 'shortwave_source_period_bounds', True),
                ):
                    original, values = nc[raw][:], nc[derived][:]
                    if int(np.ma.count_masked(values)) != 6: raise ValueError('Unexpected derived missing values')
                    period = nc[bounds][:]
                    for index in range(1, 41):
                        start, end = map(int, period[index])
                        use = (lead > start) & (lead <= end)
                        restored = np.ma.mean(values[use], axis=0) if mean else np.ma.sum(values[use], axis=0)
                        if not np.allclose(restored, original[index], atol=0.001, rtol=0): raise ValueError('Interval conservation failure: ' + file.name)
                if 'air_temperature_extended' not in nc.variables:
                    raise ValueError('Missing supplementary pressure levels')
                if not np.array_equal(nc['extended_pressure_level'][:], [400, 450, 500, 550, 600, 700, 850]):
                    raise ValueError('Incorrect extended pressure levels')
                pressure = np.asarray(nc['extended_pressure_level'][:])
                surface = np.asarray(nc['surface_pressure'][:])
                t = np.asarray(nc['air_temperature_extended'][:])
                eligible = pressure[None, :, None, None] * 100 <= surface[:, None, :, :]
                bracket = np.any(eligible & (t >= 273.15), axis=1) & np.any(eligible & (t <= 273.15), axis=1) & (eligible.sum(axis=1) >= 2)
                if not np.array_equal(bracket, nc['zero_degree_bracket_available'][:]):
                    raise ValueError('Zero-degree eligibility check failed')
                freezing = nc['freezing_level_geopotential_height'][:]
                z = np.ma.masked_where(~eligible, nc['geopotential_height_extended'][:])
                if np.any(np.ma.filled((freezing < z.min(axis=1)) | (freezing > z.max(axis=1)), False)):
                    raise ValueError('Interpolated freezing level outside eligible heights')
                report['gfs'].append({'file': file.name, 'time_and_interval_checks': True})
        for file in (ROOT / 'P0-6_Copernicus_DEM_GLO-30/tiles').rglob('*.tif'):
            with rasterio.MemoryFile(file.read_bytes()) as mem, mem.open() as ds:
                ds.read(1)
                if str(ds.crs) != 'EPSG:4326': raise ValueError('Unexpected DEM coordinate system')
                report['dem'].append({'file': file.name, 'read_ok': True})
        if args.parts_dir:
            report['grd'] = verify_parts(args.parts_dir)
        else:
            rows = list(csv.DictReader((ROOT / 'external_grd_inventory.csv').open(encoding='utf-8-sig')))
            verified = []
            for row in rows:
                file = ROOT / row['relative_path']
                verified.append(file.exists() and digest(file) == row['sha256'])
            report['grd'] = {'passed': all(verified) and len(verified) == 4, 'message': 'If archives are external, rerun with --parts-dir.'}
    except Exception as exc:
        report['errors'].append(str(exc))
    report['local_package_passed'] = not report['errors']
    report['complete_delivery_passed'] = not report['errors'] and report['grd'].get('passed', False)
    (ROOT / 'delivery_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'local_package_passed': report['local_package_passed'], 'complete_delivery_passed': report['complete_delivery_passed'], 'errors': report['errors']}, ensure_ascii=False))
    if not report['local_package_passed']: sys.exit(1)
    if not report['complete_delivery_passed']: sys.exit(2)


if __name__ == '__main__': main()
