#!/usr/bin/env python3
"""
Build the static jewelry-collection page.

Reads the HTML template and the jewelry.json data, inlines the JSON into the
template, and writes a single self-contained HTML file ready for GitHub Pages.

Also copies the optimized web images alongside the HTML so the relative
img src paths resolve.

Usage:
    python3 build.py [--out OUTDIR]

Default output: ../build/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SRC_HTML = PROJECT / "src" / "index.html"
DATA_JSON = PROJECT / "data" / "jewelry.json"
IMAGES_ROOT = PROJECT / "images"
DEFAULT_OUT = PROJECT / "build"

PLACEHOLDER = "__CATALOG_JSON__"
SECRET_PLACEHOLDER = "__JEWELRY_SYNC_SECRET__"
SECRET_FILE = Path("/Users/keithlloydsmith/Claude/ops-agent-bridge/cf-worker/.jewelry-sync-secret.txt")


def discover_image_subdir() -> Path | None:
    """Find the vault-* subdirectory under images/ (or fall back to images/web)."""
    if not IMAGES_ROOT.exists():
        return None
    vaults = [d for d in IMAGES_ROOT.iterdir() if d.is_dir() and d.name.startswith("vault-")]
    if vaults:
        return vaults[0]
    legacy = IMAGES_ROOT / "web"
    return legacy if legacy.exists() else None


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    template = SRC_HTML.read_text(encoding="utf-8")
    data_text = DATA_JSON.read_text(encoding="utf-8")

    # Sanity-validate JSON
    json.loads(data_text)

    if PLACEHOLDER not in template:
        raise SystemExit(f"Template is missing placeholder: {PLACEHOLDER}")

    rendered = template.replace(PLACEHOLDER, data_text)

    # Inject the jewelry-sync shared secret (read from secret file).
    # This sits inside the StatiCrypt-encrypted body, so it's only visible
    # to anyone who already has the page password / magic link.
    if SECRET_PLACEHOLDER in rendered:
        if SECRET_FILE.exists():
            secret = SECRET_FILE.read_text().strip()
            rendered = rendered.replace(SECRET_PLACEHOLDER, secret)
        else:
            print(f"WARNING: secret file missing at {SECRET_FILE} - sync button will fail auth")

    out_html = out_dir / "index.html"
    out_html.write_text(rendered, encoding="utf-8")

    # Copy images alongside the HTML, mirroring the vault-* subdir name
    # so the relative img src paths in jewelry.json resolve.
    src_dir = discover_image_subdir()
    if src_dir:
        rel = src_dir.relative_to(IMAGES_ROOT)
        target_images = out_dir / "images" / rel
        target_images.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.jpg")):
            dest = target_images / src.name
            try:
                if dest.exists():
                    dest.unlink()
            except PermissionError:
                pass
            shutil.copy2(src, dest)

    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the jewelry collection page.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()

    out_html = build(args.out)
    print(f"Built: {out_html}")
    src_dir = discover_image_subdir()
    if src_dir:
        rel = src_dir.relative_to(IMAGES_ROOT)
        print(f"Images: {args.out / 'images' / rel}")


if __name__ == "__main__":
    main()
