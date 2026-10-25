"""Download exact, unmodified official GRIB2 messages; never decode field values."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GFS = ROOT / 'P0-5_GFS_raw_supplement'
VARIABLES = {'albedo': ('ALBDO',), 'snow_cover': ('SNOC', 'SNOWC'),
             'snow_depth': ('SNOD',), 'snow_water_equivalent': ('WEASD',), 'downward_longwave': ('DLWRF',)}
DECODE_LOCK = threading.Lock()


def sha(data): return hashlib.sha256(data).hexdigest()


def get(url, start=None, end=None):
    headers = {'User-Agent': 'MathorCup-official-raw-supplement/1.0'}
    if start is not None: headers['Range'] = f'bytes={start}-{end}'
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if start is not None:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor exact Range request')
                    payload = response.read(end-start+2)
                    if len(payload) != end-start+1: raise ValueError('Range length mismatch')
                else:
                    payload = response.read(4*1024*1024)
                metadata = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'status': response.status,
                            'headers': {name: response.headers.get(name) for name in ('Last-Modified', 'ETag', 'Content-Length', 'Content-Range', 'Date')},
                            'information_available_time': None,
                            'information_available_time_status': 'unresolved_not_inferred_from_archive_headers'}
                return payload, metadata
        except Exception as exc:
            last = exc
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f'{url}: {last}')


def parse(text, size):
    records = []
    for line in text.splitlines():
        fields = line.split(':')
        if len(fields) < 6: continue
        records.append({'start': int(fields[1]), 'short_name': fields[3], 'level': fields[4], 'inventory_line': line})
    for index, row in enumerate(records):
        row['end'] = records[index+1]['start']-1 if index+1 < len(records) else size-1
    return records


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory_one(row):
    destination = GFS / 'inventory' / (row['filename'] + '.inv')
    if destination.exists():
        payload = destination.read_bytes()
    else:
        payload, metadata = get(row['inventory_url'])
        if not payload.startswith(b'1:'): raise ValueError('Unexpected inventory format')
        destination.write_bytes(payload)
        destination.with_suffix(destination.suffix+'.http.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    records = parse(payload.decode('utf-8'), int(row['full_grib_size_bytes']))
    selected, available = [], []
    for variable, names in VARIABLES.items():
        matches = [record for record in records if record['short_name'] in names and record['level']=='surface']
        available.append({'forecast_reference_time_utc': row['forecast_reference_time_utc'], 'lead_hours': row['lead_hours'],
                          'variable_requested': variable, 'status': 'catalog_only' if matches else 'not_available',
                          'reason': 'Exact short name and surface level found' if matches else 'Not present in this grid-004 inventory; not a claim about other products',
                          'match_count': len(matches), 'inventory_url': row['inventory_url']})
        for record in matches:
            selected.append({**row, **record, 'variable_requested': variable, 'message_bytes': record['end']-record['start']+1,
                             'inventory_file': str(destination.relative_to(ROOT)).replace('\\','/')})
    return selected, available


def metadata_from_message(blob):
    from eccodes import codes_new_from_message, codes_get, codes_release
    with DECODE_LOCK:
        handle = codes_new_from_message(blob)
        try:
            result = {}
            for name in ('name', 'shortName', 'units', 'typeOfLevel', 'level', 'dataDate', 'dataTime',
                         'forecastTime', 'stepType', 'startStep', 'endStep', 'stepUnits', 'edition',
                         'centre', 'generatingProcessIdentifier', 'productionStatusOfProcessedData'):
                try: result[name] = codes_get(handle, name)
                except Exception: result[name] = None
            return result
        finally: codes_release(handle)


def download_one(records):
    source = records[0]
    cycle = datetime.strptime(source['forecast_reference_time_utc'], '%Y%m%dT%H:%M:%SZ').strftime('%Y%m%d%H')
    destination = GFS / 'grib2' / cycle / source['filename'].replace('.grb2', '_raw_supplement.grib2')
    proof_path = destination.with_suffix('.provenance.json')
    if destination.exists() and proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding='utf-8'))
        if sha(destination.read_bytes()) == proof['file_sha256']: return proof
        raise ValueError('Existing file does not match recorded hash; refusing to overwrite: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda row: row['start'])
    messages, proofs = [], []
    offset = 0
    for row in records:
        blob, http = get(row['grib_url'], row['start'], row['end'])
        if blob[:4] != b'GRIB' or blob[-4:] != b'7777' or blob[7] != 2 or int.from_bytes(blob[8:16], 'big') != len(blob):
            raise ValueError('GRIB2 framing/length failed')
        fields = metadata_from_message(blob)
        proofs.append({'variable_requested': row['variable_requested'], 'original_short_name': row['short_name'],
                       'original_level': row['level'], 'forecast_reference_time_utc': row['forecast_reference_time_utc'],
                       'lead_hours': int(row['lead_hours']), 'valid_time_utc': row['valid_time_utc'],
                       'source_url': row['grib_url'], 'inventory_url': row['inventory_url'], 'inventory_line': row['inventory_line'],
                       'source_byte_start': row['start'], 'source_byte_end': row['end'],
                       'local_byte_start': offset, 'local_byte_end': offset+len(blob)-1,
                       'message_sha256': sha(blob), 'grib_metadata': fields, 'http_metadata': http})
        messages.append(blob)
        offset += len(blob)
    content = b''.join(messages)
    temporary = destination.with_suffix('.grib2.part')
    temporary.write_bytes(content)
    temporary.replace(destination)
    proof = {'file': str(destination.relative_to(ROOT)).replace('\\','/'), 'bytes': len(content), 'file_sha256': sha(content),
             'message_count': len(proofs), 'processing': 'Exact official GRIB2 messages concatenated without re-encoding, cropping, or value transformation', 'messages': proofs}
    proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['catalog', 'download'])
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.mode == 'catalog':
        rows = list(csv.DictReader((GFS/'inventory/source_gfs_official_inventory.csv').open(encoding='utf-8-sig')))
        selected, available, failures = [], [], []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs = {executor.submit(inventory_one, row): row for row in rows}
            done = 0
            for future in as_completed(jobs):
                try:
                    a,b = future.result(); selected.extend(a); available.extend(b)
                except Exception as exc: failures.append({'row': jobs[future], 'error': str(exc)})
                done += 1
                if done%100 == 0 or done==len(rows): print(f'Inventory: {done}/{len(rows)}, failed={len(failures)}', flush=True)
        if failures:
            (ROOT/'logs/gfs_inventory_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
            raise RuntimeError('Not all inventories were retrieved')
        selected.sort(key=lambda row:(row['forecast_reference_time_utc'],int(row['lead_hours']),row['start']))
        available.sort(key=lambda row:(row['forecast_reference_time_utc'],int(row['lead_hours']),row['variable_requested']))
        write_csv(GFS/'inventory/selected_messages.csv',selected)
        write_csv(GFS/'inventory/variable_availability.csv',available)
        total = sum(row['message_bytes'] for row in selected)
        summary={'inventory_count':len(rows), 'selected_message_count':len(selected), 'estimated_raw_download_bytes':total,
                 'estimated_GiB':total/1024**3,'by_variable':{name:sum(row['variable_requested']==name for row in selected) for name in VARIABLES}}
        (GFS/'inventory/download_plan.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    else:
        plan=json.loads((GFS/'inventory/download_plan.json').read_text(encoding='utf-8'))
        if plan['estimated_raw_download_bytes']>10*1024**3: raise RuntimeError('Planned download exceeds 10 GiB; approval required')
        rows=list(csv.DictReader((GFS/'inventory/selected_messages.csv').open(encoding='utf-8-sig')))
        groups={}
        for row in rows:
            for key in ('start','end','message_bytes'): row[key]=int(row[key])
            if row['message_bytes']>2*1024**3: raise RuntimeError('Single message exceeds 2 GiB; approval required')
            groups.setdefault(row['filename'],[]).append(row)
        if any(sum(r['message_bytes'] for r in group)>2*1024**3 for group in groups.values()): raise RuntimeError('Single output exceeds 2 GiB')
        results,failures=[],[]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs={executor.submit(download_one,group):name for name,group in groups.items()}
            for future in as_completed(jobs):
                try: results.append(future.result())
                except Exception as exc: failures.append({'file':jobs[future],'error':str(exc)})
                done=len(results)+len(failures)
                if done%100==0 or done==len(groups): print(f'Raw files: {done}/{len(groups)}, failed={len(failures)}',flush=True)
        (ROOT/'logs/gfs_download_results.json').write_text(json.dumps({'files':len(results),'messages':sum(row['message_count'] for row in results),'bytes':sum(row['bytes'] for row in results),'failures':failures},indent=2),encoding='utf-8')
        if failures: raise RuntimeError('Some raw downloads failed; rerun to reuse verified files')
        print('Raw download completed without changing message bytes.',flush=True)


if __name__=='__main__': main()
