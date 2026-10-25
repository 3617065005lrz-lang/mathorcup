#!/usr/bin/env python3
"""Query official CDSE and USGS STAC catalogs and save all matched items to CSV."""
import argparse, csv, json, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

AOI = [85.19, 28.25, 85.69, 28.56]
START, END = "2026-06-01T00:00:00Z", "2026-09-15T23:59:59Z"

def get_json(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, method=method, headers={"User-Agent":"mathorcup-computer3-catalog/1.0", "Accept":"application/geo+json, application/json", "Content-Type":"application/json"})
    with urlopen(req, timeout=90) as r:
        return json.load(r)

def stac_items(url, method="GET", body=None):
    while url:
        page = get_json(url, method, body)
        yield from page.get("features", [])
        nxt = next((x for x in page.get("links", []) if x.get("rel") == "next"), None)
        if not nxt: break
        url = nxt["href"]
        method = nxt.get("method", "GET").upper()
        body = nxt.get("body") if method == "POST" else None
        time.sleep(0.2)

def write_s2(out):
    url = "https://stac.dataspace.copernicus.eu/v1/search"
    body = {"collections":["sentinel-2-l2a"], "datetime":f"{START}/{END}", "bbox":AOI, "limit":100}
    rows=[]
    for f in stac_items(url, "POST", body):
        p=f.get("properties", {})
        dt=(p.get("datetime") or f.get("id", ""))
        d=dt[:10]
        phase="pre_event" if d <= "2026-08-25" else "post_event"
        rows.append({"item_id":f.get("id"),"date":d,"platform":p.get("platform"),"orbit":p.get("sat:relative_orbit"),"grid":p.get("s2:tile_id"),"cloud_cover_percent":p.get("eo:cloud_cover"),"size_bytes":p.get("product_size"),"phase":phase,"post_event_validation_only":str(phase=="post_event").lower(),"collection":"sentinel-2-l2a","item_url":f.get("links",[{}])[0].get("href","")})
    save(out/"sentinel2_full_catalog.csv", rows)

def write_landsat(out):
    base="https://landsatlook.usgs.gov/stac-server/search"
    params={"collections":"landsat-c2l2-sr","datetime":f"{START}/{END}","bbox":",".join(map(str,AOI)),"limit":100}
    rows=[]
    for f in stac_items(base+"?"+urlencode(params)):
        p=f.get("properties",{}); dt=(p.get("datetime") or ""); d=dt[:10]
        phase="pre_event" if d <= "2026-08-25" else "post_event"
        rows.append({"item_id":f.get("id"),"date":d,"platform":p.get("platform"),"path":p.get("landsat:wrs_path"),"row":p.get("landsat:wrs_row"),"cloud_cover_percent":p.get("eo:cloud_cover"),"phase":phase,"post_event_validation_only":str(phase=="post_event").lower(),"collection":"landsat-c2l2-sr","item_url":f.get("links",[{}])[0].get("href","")})
    save(out/"landsat_full_catalog.csv", rows)

def save(path, rows):
    if not rows:
        print(f"No items: {path}"); return
    with path.open("w",encoding="utf-8-sig",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{path}: {len(rows)} items")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("catalog")); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    write_s2(a.out); write_landsat(a.out)
