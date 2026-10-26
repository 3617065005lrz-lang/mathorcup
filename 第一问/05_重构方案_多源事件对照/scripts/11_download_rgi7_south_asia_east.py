from __future__ import annotations

"""Download official RGI 7.0 South Asia East regional archives."""

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin

import earthaccess


BASE = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/"
OUT = Path(r"D:\MathorCup_Data\External_Gyirong\P1-4_RGI_7")


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.values.append(href)


def listing(session, url: str) -> list[str]:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    return [urljoin(url, href) for href in parser.values]


def candidates(session) -> list[str]:
    queue: list[tuple[str, int]] = [(BASE, 0)]
    seen: set[str] = set()
    urls: list[str] = []
    while queue:
        current, depth = queue.pop(0)
        if current in seen or depth > 4:
            continue
        seen.add(current)
        try:
            children = listing(session, current)
        except Exception:
            continue
        for url in children:
            if not url.startswith(BASE) or url in seen:
                continue
            urls.append(url)
            if url.endswith("/") and url != current:
                queue.append((url, depth + 1))
    selected = []
    for url in urls:
        name = url.rsplit("/", 1)[-1].lower()
        if name.endswith(".zip") and "15" in name and "south" in name and "asia" in name and "east" in name:
            selected.append(url)
    return sorted(set(selected))


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")
    session = auth.get_session()
    urls = candidates(session)
    if not urls:
        raise RuntimeError("No RGI 7 South Asia East archive found in official directory")
    manifest = []
    for index, url in enumerate(urls, start=1):
        target = OUT / url.rsplit("/", 1)[-1]
        if not target.exists() or target.stat().st_size == 0:
            partial = target.with_suffix(target.suffix + ".part")
            with session.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with partial.open("wb") as fh:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            partial.replace(target)
        manifest.append({"source_url": url, "target": str(target), "bytes": target.stat().st_size, "sha256": digest(target)})
        print(f"RGI {index}/{len(urls)} {target.name} {target.stat().st_size / 1024**2:.1f} MB", flush=True)
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
