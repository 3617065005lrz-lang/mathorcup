"""Create interval-aware GFS files without changing the received source data."""
from pathlib import Path
import csv
import json
import re
import sys
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def intervals(path):
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['output_variable'] not in ('total_precipitation_accumulation', 'downward_shortwave_radiation_flux'):
                continue
            match = re.search(r':(\d+)-(\d+) hour (acc|ave) fcst:', row['inventory_line'])
            if not match:
                raise ValueError('Unrecognized statistical interval: ' + row['inventory_line'])
            key = row['output_variable'], int(row['lead_hours'])
            if key in result:
                raise ValueError('Duplicate interval record: ' + str(key))
            result[key] = tuple(map(int, match.group(1, 2)))
    return result


def normalize(raw, bounds, mean=False):
    """Return nonoverlapping intervals and a conservation diagnostic."""
    derived = np.ma.masked_all(raw.shape, dtype='f4')
    lookup = {tuple(pair): index for index, pair in enumerate(bounds) if pair[0] >= 0}
    errors = []
    for index, (start, end) in enumerate(bounds):
        if start < 0:
            continue
        duration = end - start
        if duration == 3:
            derived[index] = raw[index]
        elif duration == 6:
            previous = lookup[(start, end - 3)]
            derived[index] = (raw[index] * 6 - raw[previous] * 3) / 3 if mean else raw[index] - raw[previous]
            restored = (derived[previous] + derived[index]) / 2 if mean else derived[previous] + derived[index]
            errors.append(float(np.ma.max(np.ma.abs(restored - raw[index]))))
        else:
            raise ValueError(f'Unsupported interval duration: {duration}')
    return derived, max(errors, default=0.0)


def process(path, destination):
    bound_map = intervals(path.with_name(path.stem + '_provenance.csv'))
    source = Dataset('source.nc', memory=path.read_bytes())
    target = Dataset('derived.nc', 'w', format='NETCDF4', memory=2 * 1024 * 1024)
    try:
        for name, dim in source.dimensions.items():
            target.createDimension(name, len(dim))
        for name, value in source.__dict__.items():
            target.setncattr(name, value)
        target.analysis_processing = 'Source values retained. Statistical intervals decoded from provenance; nonoverlapping 3-hour precipitation and radiation derived.'
        target.model_version_status = 'Operational GFS software version not independently established; grid 004 is a grid designation.'
        for name, variable in source.variables.items():
            kwargs = {'fill_value': variable._FillValue} if '_FillValue' in variable.ncattrs() else {}
            v = target.createVariable(name, variable.dtype, variable.dimensions, zlib=bool(variable.dimensions), **kwargs)
            v.setncatts({k: variable.getncattr(k) for k in variable.ncattrs() if k != '_FillValue'})
            v[...] = variable[...]
        target.createDimension('bounds', 2)
        lead = np.asarray(source['forecast_period'][:], dtype=int)
        result = {'file': destination.name, 'source_file': path.name, 'variables': {}}
        for old, prefix, new, is_mean in (
            ('total_precipitation_accumulation', 'precipitation', 'precipitation_3h', False),
            ('downward_shortwave_radiation_flux', 'shortwave', 'shortwave_mean_3h', True),
        ):
            bounds = np.array([bound_map.get((old, int(t)), (-1, -1)) for t in lead], dtype='i4')
            b = target.createVariable(prefix + '_source_period_bounds', 'i4', ('forecast_period', 'bounds'), fill_value=-1)
            b.units = 'hours since forecast_reference_time'
            b.long_name = 'Original GRIB statistical interval [start, end] in lead hours'
            b[:] = np.ma.masked_equal(bounds, -1)
            target[old].ancillary_variables = prefix + '_source_period_bounds'
            derived, error = normalize(source[old][:], bounds, mean=is_mean)
            v = target.createVariable(new, 'f4', ('forecast_period', 'latitude', 'longitude'), zlib=True, fill_value=np.float32(9.96921e36))
            v.units = 'W m-2' if is_mean else 'mm'
            v.long_name = 'Nonoverlapping preceding 3-hour mean downward shortwave flux' if is_mean else 'Nonoverlapping preceding 3-hour precipitation depth'
            v.comment = 'Missing at +000. Negative values, if any, are retained and counted in the QA report; no silent clipping. 1 kg m-2 precipitation equals 1 mm water depth.'
            v.cell_methods = 'forecast_period: mean' if is_mean else 'forecast_period: sum'
            v[:] = derived
            result['variables'][new] = {'min': float(derived.min()), 'max': float(derived.max()), 'negative_count': int(np.ma.sum(derived < 0)), 'missing_count': int(np.ma.count_masked(derived)), 'max_reconstruction_error': error}
            if is_mean:
                qc = derived.copy()
                small = np.ma.filled((derived < 0) & (derived >= -0.1), False)
                large = np.ma.filled(derived < -0.1, False)
                qc[small] = 0
                qc[large] = np.ma.masked
                v = target.createVariable('shortwave_mean_3h_qc', 'f4', ('forecast_period', 'latitude', 'longitude'), zlib=True, fill_value=np.float32(9.96921e36))
                v.units = 'W m-2'
                v.comment = 'Explicit optional QC: values in [-0.1, 0) W m-2 set to 0; values below -0.1 masked. Threshold is an operational tolerance, not an independently established error bound. Unchanged interval-derived values remain in shortwave_mean_3h and are used for conservation checks.'
                v[:] = qc
                flag = np.zeros(derived.shape, dtype='i1')
                flag[small], flag[large] = 1, 2
                flag[np.ma.getmaskarray(derived)] = 3
                v = target.createVariable('shortwave_3h_qc_flag', 'i1', ('forecast_period', 'latitude', 'longitude'))
                v.flag_values = np.array([0, 1, 2, 3], dtype='i1')
                v.flag_meanings = 'unchanged small_negative_set_to_zero large_negative_masked original_missing'
                v[:] = flag
                result['shortwave_optional_qc'] = {'small_negative_set_to_zero': int(small.sum()), 'large_negative_masked': int(large.sum()), 'threshold_W_m2': 0.1}
            if np.ma.count_masked(derived) != 6:
                raise ValueError('Unexpected derived missing values')
            if error > 0.001:
                raise ValueError('Interval reconstruction failed')
        bounds = np.column_stack([lead - 3, lead]).astype('i4')
        bounds[0] = -1
        v = target.createVariable('analysis_period_bounds', 'i4', ('forecast_period', 'bounds'), fill_value=-1)
        v.units = 'hours since forecast_reference_time'
        v[:] = np.ma.masked_equal(bounds, -1)
        target['precipitation_3h'].coordinates = 'valid_time latitude longitude'
        target['shortwave_mean_3h'].coordinates = 'valid_time latitude longitude'
        target['precipitation_3h'].ancillary_variables = 'analysis_period_bounds'
        target['shortwave_mean_3h'].ancillary_variables = 'analysis_period_bounds'
        temp = np.asarray(source['air_temperature_isobaric'][:])
        pressure = np.asarray(source['pressure_level'][:])
        surface = np.asarray(source['surface_pressure'][:])
        above_ground = pressure[None, :, None, None] * 100 <= surface[:, None, :, :]
        warm = np.any(above_ground & (temp >= 273.15), axis=1)
        cold = np.any(above_ground & (temp <= 273.15), axis=1)
        bracket = warm & cold & (above_ground.sum(axis=1) >= 2)
        original_bracket = bracket.copy()
        supplement_path = ROOT / 'P0-5_GFS/pressure_supplement' / path.name.replace('.nc', '_levels_400_450_550.nc')
        if supplement_path.exists():
            with Dataset('supplement.nc', memory=supplement_path.read_bytes()) as supplement:
                for coord in ('forecast_period', 'latitude', 'longitude', 'valid_time', 'forecast_reference_time'):
                    if not np.array_equal(source[coord][...], supplement[coord][...]):
                        raise ValueError('Supplement coordinate mismatch: ' + coord)
                combined_pressure = np.concatenate([pressure, supplement['pressure_level'][:]])
                order = np.argsort(combined_pressure)
                pressure = combined_pressure[order]
                temp = np.concatenate([temp, supplement['air_temperature_isobaric'][:]], axis=1)[:, order]
                heights = np.concatenate([source['geopotential_height_isobaric'][:], supplement['geopotential_height_isobaric'][:]], axis=1)[:, order]
                target.createDimension('extended_pressure_level', len(pressure))
                v = target.createVariable('extended_pressure_level', 'i4', ('extended_pressure_level',))
                v.units = 'hPa'
                v[:] = pressure
                for name, data, units in (('air_temperature_extended', temp, 'K'), ('geopotential_height_extended', heights, 'gpm')):
                    v = target.createVariable(name, 'f4', ('forecast_period', 'extended_pressure_level', 'latitude', 'longitude'), zlib=True)
                    v.units = units
                    v[:] = data
                above_ground = pressure[None, :, None, None] * 100 <= surface[:, None, :, :]
                warm = np.any(above_ground & (temp >= 273.15), axis=1)
                cold = np.any(above_ground & (temp <= 273.15), axis=1)
                bracket = warm & cold & (above_ground.sum(axis=1) >= 2)
                target.pressure_supplement_file = supplement_path.name
                freezing = np.ma.masked_all(surface.shape, dtype='f4')
                for t, y, x in np.ndindex(surface.shape):
                    eligible = np.flatnonzero(above_ground[t, :, y, x])
                    eligible = eligible[np.argsort(heights[t, eligible, y, x])]
                    for low, high in zip(eligible[:-1], eligible[1:]):
                        tl, th = float(temp[t, low, y, x]), float(temp[t, high, y, x])
                        zl, zh = float(heights[t, low, y, x]), float(heights[t, high, y, x])
                        if tl >= 273.15 >= th and zh > zl:
                            fraction = 0.0 if tl == th else (tl - 273.15) / (tl - th)
                            freezing[t, y, x] = zl + fraction * (zh - zl)
                            break
                v = target.createVariable('freezing_level_geopotential_height', 'f4', ('forecast_period', 'latitude', 'longitude'), zlib=True, fill_value=np.float32(9.96921e36))
                v.units = 'gpm'
                v.comment = 'Lowest warm-to-cold 0 deg C crossing among available above-ground pressure levels, linearly interpolated in geopotential height. No extrapolation. Missing if no eligible crossing. Coarse model estimate, not source-area local freezing height; gpm is not ellipsoidal/DSM geometric height.'
                v[:] = freezing
                result['freezing_level_geopotential_height'] = {'available': int(freezing.count()), 'missing': int(np.ma.count_masked(freezing)), 'min_gpm': float(freezing.min()), 'max_gpm': float(freezing.max())}
        v = target.createVariable('zero_degree_bracket_available', 'i1', ('forecast_period', 'latitude', 'longitude'))
        v.flag_values = np.array([0, 1], dtype='i1')
        v.flag_meanings = 'not_bracketed bracketed'
        v.comment = 'Only pressure levels at or above model ground are eligible. Uses extended levels if a matched supplement is present. A bracket alone does not specify which crossing occurs in an inversion profile.'
        v[:] = bracket.astype('i1')
        result['zero_degree_bracket_available'] = {'yes': int(bracket.sum()), 'no': int((~bracket).sum()), 'total': int(bracket.size)}
        result['original_four_level_bracket'] = {'yes': int(original_bracket.sum()), 'no': int((~original_bracket).sum()), 'total': int(original_bracket.size)}
        return result
    finally:
        source.close()
        buffer = target.close()
        # Write the memory image after successful computations; Unicode paths are supported.
        if sys.exc_info()[0] is None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(buffer)


def main():
    output = ROOT / 'P0-5_GFS/analysis_ready'
    files = sorted((ROOT / 'P0-5_GFS/data').glob('*.nc'))
    if len(files) != 20:
        raise ValueError(f'Expected 20 source cycles, got {len(files)}')
    report = [process(f, output / f.name.replace('.nc', '_interval_aware.nc')) for f in files]
    qa = {'cycles': report, 'source_data_unchanged': True, 'version': 1,
          'zero_degree_bracket_available': {key: sum(r['zero_degree_bracket_available'][key] for r in report) for key in ('yes', 'no', 'total')}}
    (output / 'analysis_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'cycles': len(report), 'zero_degree_bracket_available': qa['zero_degree_bracket_available']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
