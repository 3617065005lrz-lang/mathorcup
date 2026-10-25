#!/usr/bin/env python3
"""Fetch BIPAD event records for discovery; keeps source event labels unchanged."""
import argparse, json
from pathlib import Path
from urllib.request import Request, urlopen

URL="https://bipadportal.gov.np/api/v1/event/"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--timeout",type=int,default=60); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    req=Request(URL,headers={"Accept":"application/json","User-Agent":"mathorcup-computer3-catalog/1.0"})
    try:
        with urlopen(req,timeout=a.timeout) as r: raw=r.read()
        (a.out/"bipad_events_raw.json").write_bytes(raw)
        data=json.loads(raw)
        (a.out/"bipad_events_pretty.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"Saved raw API response ({len(raw)} bytes). Inspect pagination and event taxonomy before filtering.")
    except Exception as e:
        print(f"BIPAD request failed: {type(e).__name__}: {e}")
        raise SystemExit(2)
if __name__=="__main__": main()
