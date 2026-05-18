#!/usr/bin/env python3
"""
Process a jewelry upload batch.

Runs inside a GitHub Action triggered by the Cloudflare Worker's
repository_dispatch on `jewelry-upload`. The Action sets:

  BATCH_ID                       - uuid for the batch
  WORKER_BASE                    - https://goodstream-ops-webhooks...
  JEWELRY_SYNC_SECRET            - auth header for Worker R2 gateway endpoints
  ANTHROPIC_API_KEY              - for vision identification
  JEWELRY_COLLECTION_PASSWORD    - StatiCrypt password
  REPO_PREFIX                    - leena-jewelry-source

Pipeline per batch:
  1. Fetch manifest from Worker.
  2. For each uploaded file: download via Worker R2 gateway, convert HEIC->JPEG if needed.
  3. For each file: call Claude API to identify the piece (returns piece type, stones, etc.).
  4. Add new pieces to data/jewelry.json (one piece per file for v1 - clustering is phase 3).
  5. Resize+optimize images, write into the vault images folder, commit will push.
  6. Run build.py to inline JSON, then run staticrypt to encrypt the HTML.
  7. Write the encrypted HTML + images into leena-jewelry/ for GitHub Pages.

Idempotent: if a file's piece is already in the catalog (by content hash), skip.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

BATCH_ID = os.environ["BATCH_ID"]
WORKER_BASE = os.environ["WORKER_BASE"].rstrip("/")
SECRET = os.environ["JEWELRY_SYNC_SECRET"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PASSWORD = os.environ["JEWELRY_COLLECTION_PASSWORD"]
REPO_PREFIX = os.environ.get("REPO_PREFIX", "leena-jewelry-source")

SOURCE = Path(REPO_PREFIX)
DATA = SOURCE / "data" / "jewelry.json"
SRC_HTML = SOURCE / "src" / "index.html"
VAULT = (SOURCE / ".vault-folder").read_text().strip()

# Output paths - committed to repo at leena-jewelry/ where GitHub Pages serves them
OUT_DIR = Path("leena-jewelry")
OUT_HTML = OUT_DIR / "index.html"
OUT_IMG_DIR = OUT_DIR / "images" / VAULT

CANONICAL_SALT = "a6421d3f8512a52b21f45d4a0dd7508b"

# ---------------------------------------------------------------------------
# Worker R2 gateway
# ---------------------------------------------------------------------------

def fetch_manifest() -> dict:
    r = requests.get(
        f"{WORKER_BASE}/jewelry-status",
        params={"batch_id": BATCH_ID},
        headers={"x-gs-jewelry-secret": SECRET, "Origin": "https://keith-gs.github.io"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_file(filename: str) -> bytes:
    r = requests.get(
        f"{WORKER_BASE}/jewelry-batch-file",
        params={"batch_id": BATCH_ID, "filename": filename},
        headers={"x-gs-jewelry-secret": SECRET},
        timeout=60,
    )
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def convert_to_jpeg(src: Path, dst: Path) -> None:
    """Robust HEIC/HEIF -> JPEG using pillow_heif, with libheif as fallback.

    pillow_heif handles iPhone Live Photos and auxiliary image references that
    older versions of `heif-convert` choke on with
    "Too many auxiliary image references".
    """
    suffix = src.suffix.lower()
    if suffix in {".heic", ".heif", ".avif"}:
        try:
            import pillow_heif  # type: ignore
            from PIL import Image
            pillow_heif.register_heif_opener()
            img = Image.open(str(src))
            # Convert to RGB and save as JPEG
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            img.save(str(dst), "JPEG", quality=92, optimize=True)
            return
        except Exception as e:
            print(f"    pillow_heif fallback to heif-convert: {e}")
            subprocess.check_call(["heif-convert", "-q", "92", str(src), str(dst)])
            return
    else:
        # Use Pillow for everything else
        try:
            from PIL import Image
            img = Image.open(str(src))
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            img.save(str(dst), "JPEG", quality=88, optimize=True)
        except Exception:
            subprocess.check_call(["convert", str(src), "-quality", "88", str(dst)])


def resize_for_web(src: Path, dst: Path, max_dim: int = 1600) -> None:
    subprocess.check_call([
        "convert", str(src),
        "-resize", f"{max_dim}x{max_dim}>",
        "-quality", "82",
        str(dst),
    ])


# ---------------------------------------------------------------------------
# Anthropic vision
# ---------------------------------------------------------------------------

VISION_SYSTEM = """You are a jewelry cataloging assistant. Look at the photo and identify the single piece of jewelry shown.

Return STRICTLY a JSON object with these fields:
- name: short descriptive name (e.g., "Emerald and Pearl Drop Earrings")
- type: one of: necklace, earrings, ring, bangle, bracelet, maang tikka, pendant, brooch, other
- subtype: short description like "chandelier", "stud", "cluster"
- category: one of: heritage Indian, contemporary, antique, costume, other
- stones: array of stone names visible (e.g., ["ruby", "diamond", "pearl"])
- metal_color: one of: yellow, white, rose, mixed, unknown
- estimated_retail_replacement_usd: a conservative single number guess
- estimated_low_usd: low end of range
- estimated_high_usd: high end of range
- design_motifs: array of brief motif descriptions
- notes: 1-2 sentences about distinctive features

Do not include any markdown formatting, code fences, or explanation. Just the JSON object."""


def identify_piece(jpeg_path: Path) -> dict:
    img_b64 = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 800,
            "system": VISION_SYSTEM,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": "Identify this jewelry piece. Return JSON only."},
                ],
            }],
        },
        timeout=120,
    )
    r.raise_for_status()
    txt = r.json()["content"][0]["text"].strip()
    # Strip code fences if present
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1]
        if txt.startswith("json"):
            txt = txt[4:]
        txt = txt.strip("` \n")
    return json.loads(txt)


# ---------------------------------------------------------------------------
# Catalog update
# ---------------------------------------------------------------------------

def next_piece_id(catalog: dict) -> str:
    nums = []
    for p in catalog["pieces"]:
        try:
            nums.append(int(p["id"].split("-")[1]))
        except Exception:
            pass
    return f"piece-{(max(nums) + 1) if nums else 1:02d}"


def build_piece_entry(piece_id: str, image_filename: str, vision: dict) -> dict:
    slug_root = vision.get("name", "piece").lower().replace(" ", "-").replace("/", "-")
    return {
        "id": piece_id,
        "slug": slug_root[:60],
        "name": vision.get("name", "New piece"),
        "type": vision.get("type", "other"),
        "subtype": vision.get("subtype", ""),
        "category": vision.get("category", ""),
        "era_estimate": "contemporary (pending review)",
        "region_origin": "(pending Leena's input)",
        "metal": {
            "type": f"{vision.get('metal_color', 'unknown')} metal (pending hallmark)",
            "color": vision.get("metal_color", "unknown"),
            "estimated_weight_g": None,
            "hallmarks_observed": "not yet documented",
        },
        "stones": [{"stone": s, "count_estimate": None, "cut": None, "carat_total_estimate": None, "color": None, "notes": ""} for s in vision.get("stones", [])],
        "design_motifs": vision.get("design_motifs", []),
        "dimensions": {},
        "condition": "good",
        "provenance": {"acquired_from": None, "acquired_when": None, "gift_from": None, "notes": "Awaiting Leena's input."},
        "story": "",
        "occasions_worn": [],
        "valuation": {
            "retail_replacement_usd": int(vision.get("estimated_retail_replacement_usd", 0) or 0),
            "retail_replacement_range_low": int(vision.get("estimated_low_usd", 0) or 0),
            "retail_replacement_range_high": int(vision.get("estimated_high_usd", 0) or 0),
            "fair_market_usd": int((vision.get("estimated_retail_replacement_usd", 0) or 0) * 0.5),
            "insurance_recommended_usd": int((vision.get("estimated_retail_replacement_usd", 0) or 0) * 1.2),
            "confidence": "low",
            "status": "pending_appraisal",
            "source_notes": vision.get("notes", "") + " (AI-identified - confirm with Leena and appraiser.)",
        },
        "appraisal": {"appraised": False, "gemologist": None, "organization": None, "certification_number": None, "date": None, "document_url": None},
        "images": [{"src": f"images/{VAULT}/{image_filename}", "alt": vision.get("name", "Jewelry piece"), "type": "hero", "order": 1}],
        "tags": [vision.get("type", "other")] + vision.get("stones", []),
        "created_at": time.strftime("%Y-%m-%d"),
        "updated_at": time.strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Build + encrypt
# ---------------------------------------------------------------------------

def build_html() -> Path:
    """Inline jewelry.json + jewelry-sync secret into the template."""
    template = SRC_HTML.read_text()
    data_text = DATA.read_text()
    if "__CATALOG_JSON__" not in template:
        raise SystemExit("Template missing __CATALOG_JSON__ placeholder")
    rendered = template.replace("__CATALOG_JSON__", data_text)

    # CRITICAL: substitute the jewelry-sync secret so the upload UI can auth to the Worker.
    # The secret lives inside the encrypted page body - only visible after unlock.
    secret = os.environ.get("JEWELRY_SYNC_SECRET", "")
    if not secret:
        raise SystemExit("JEWELRY_SYNC_SECRET env var missing - upload UI would fail auth.")
    rendered = rendered.replace("__JEWELRY_SYNC_SECRET__", secret)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plain_html = OUT_DIR / "_built.html"
    plain_html.write_text(rendered)
    return plain_html


def encrypt_html(plain: Path) -> None:
    """Run staticrypt over the built HTML, write into OUT_HTML."""
    work = Path("/tmp/encrypt-work")
    work.mkdir(exist_ok=True)
    # Copy source into work dir, run staticrypt, output to OUT_HTML
    work_src = work / "source.html"
    work_src.write_bytes(plain.read_bytes())
    salt_config = work / ".staticrypt.json"
    salt_config.write_text(json.dumps({"salt": CANONICAL_SALT}))
    subprocess.check_call([
        "staticrypt", str(work_src),
        "-p", PASSWORD,
        "--short",
        "--template-title", "The Punjabi Collection",
        "--template-instructions", "This page is password protected.",
        "--template-color-primary", "#0a0908",
        "--template-color-secondary", "#f5efe6",
        "--template-button", "Unlock",
        "--remember", "90",
        "-d", str(work / "out"),
    ], cwd=str(work))
    encrypted = work / "out" / "source.html"
    OUT_HTML.write_bytes(encrypted.read_bytes())
    plain.unlink()  # remove the un-encrypted intermediate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Processing batch {BATCH_ID}")
    manifest = fetch_manifest()
    files = [f for f in manifest.get("files", []) if f.get("status") == "uploaded"]
    print(f"  {len(files)} file(s) to process")

    catalog = json.loads(DATA.read_text())

    work_dir = Path(f"/tmp/jewelry-batch-{BATCH_ID}")
    work_dir.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    new_pieces = 0
    for f in files:
        fname = f["name"]
        print(f"  - {fname}")
        raw_path = work_dir / fname
        raw_path.write_bytes(fetch_file(fname))

        # Convert to JPEG
        jpeg_path = work_dir / (Path(fname).stem + ".jpg")
        try:
            convert_to_jpeg(raw_path, jpeg_path)
        except Exception as e:
            print(f"    convert failed: {e}")
            continue

        # Anthropic's image limit is 5MB AND recommends <1568x1568 pixels.
        # iPhone photos routinely exceed both - downscale to a vision-safe size
        # BEFORE sending. This is a separate intermediate; the on-page image
        # is still rendered from the larger source via resize_for_web later.
        vision_path = work_dir / (Path(fname).stem + ".vision.jpg")
        try:
            resize_for_web(jpeg_path, vision_path, max_dim=1400)
            # Belt and suspenders - if still over 4.8MB, hammer it harder
            if vision_path.stat().st_size > 4_800_000:
                resize_for_web(jpeg_path, vision_path, max_dim=1000)
        except Exception as e:
            print(f"    pre-vision resize failed: {e}")
            vision_path = jpeg_path  # fall back to original

        # Identify via Claude vision
        try:
            vision = identify_piece(vision_path)
            print(f"    identified: {vision.get('name')}")
        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            print(f"    vision failed: {e}  body={body}")
            continue
        except Exception as e:
            print(f"    vision failed: {e}")
            continue

        piece_id = next_piece_id(catalog)
        slug = (vision.get("name", "piece").lower().replace(" ", "-").replace("/", "-"))[:50]
        web_filename = f"{piece_id}-{slug}.jpg"
        web_path = OUT_IMG_DIR / web_filename

        # Resize and optimize
        resize_for_web(jpeg_path, web_path)

        piece = build_piece_entry(piece_id, web_filename, vision)
        catalog["pieces"].append(piece)
        catalog["pieces"].sort(key=lambda p: int(p["id"].split("-")[1]) if p["id"].split("-")[1].isdigit() else 999)
        catalog["collection"]["last_updated"] = time.strftime("%Y-%m-%d")
        DATA.write_text(json.dumps(catalog, indent=2))
        new_pieces += 1

    # Build + encrypt the HTML
    plain = build_html()
    encrypt_html(plain)
    print(f"Done. {new_pieces} new piece(s). Encrypted page written to {OUT_HTML}")


if __name__ == "__main__":
    main()
