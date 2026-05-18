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
    """Always pull the RAW manifest (with files array), not the summary."""
    r = requests.get(
        f"{WORKER_BASE}/jewelry-status",
        params={"batch_id": BATCH_ID, "raw": "1"},
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

VISION_SYSTEM = """You are a doctoral-level jewelry cataloging expert with deep knowledge of heritage Indian (Mughal, polki, Jaipur, Hyderabad traditions), Art Deco, and contemporary jewelry. Look at the photo and produce a rich, museum-quality catalog entry for the single piece shown.

Return STRICTLY a JSON object - no markdown, no code fences, NO COMMENTS (JSON does not allow // or /* */ comments), no prose outside the JSON. Every field is required (use null only where genuinely unknowable).

REQUIRED FIELDS:
- name: full descriptive name (e.g. "Ruby Briolette Maang Tikka with Diamond Plaque")
- type: one of "necklace", "earrings", "ring", "bangle", "bracelet", "maang tikka", "pendant", "brooch", "nose-ring", "other"
- subtype: specific form (e.g. "chandelier shoulder-duster", "polki bridal collar", "hinged bangle with central motif")
- category: cultural/use category (e.g. "heritage Indian, bridal", "heritage Indian, formal", "contemporary day-wear", "Art Deco", "antique")
- era_estimate: best guess of era WITH reasoning (e.g. "mid-20th century or earlier - architectural cut-work suggests pre-1960 craftsmanship")
- region_origin: best guess of origin region with reasoning (e.g. "India (Mughal-revival tradition; possibly Hyderabad or Jaipur)")
- metal: an object with these keys: type (detailed description with hallmark caveat), color ("yellow"|"white"|"rose"|"mixed"), estimated_weight_g (null), hallmarks_observed (string)
- stones: array of stone objects, ONE PER STONE TYPE present. Each object has: stone, count_estimate, cut, carat_total_estimate, color, notes (each is a string or null)
- design_motifs: array of specific named motif strings, 2-5 entries typically (e.g. "Mughal jali lattice", "central butterfly medallion", "yin-yang ruby twin-cabochon centerpiece")
- dimensions: object with whichever applies of: length_mm_estimate, drop_mm_estimate, width_mm_estimate, chain_length_mm_estimate, centerpiece_width_mm_estimate, inner_diameter_mm_estimate (each a string range like "55-75"). Omit fields that don't apply.
- condition: short string like "excellent", "very good", "good - some visible wear"
- estimated_retail_replacement_usd: integer
- estimated_low_usd: integer (conservative floor)
- estimated_high_usd: integer (upper bound if quality assumptions check out)
- estimated_fair_market_usd: integer (typically ~50% of retail replacement)
- estimated_insurance_recommended_usd: integer (typically ~120% of retail replacement)
- valuation_notes: 2-3 sentences explaining valuation reasoning - which comps, which variables shift the range (stone authenticity, metal purity, provenance), standout features. Reference Sotheby's / Bonhams / independent Indian jewelers as comp bases where relevant.

Guidelines for quality:
- For heritage Indian pieces, hypothesize era and origin actively - 'Mughal-revival', 'late 19th century or earlier' is more useful than '(pending Leena's input)'.
- For each visible stone TYPE produce ONE detailed entry with all five sub-fields populated where possible.
- Design motifs should be richly descriptive (a museum catalog level), not generic ('floral' is bad, 'central Mughal jali medallion with seed-pearl outlined fan motifs' is good).
- Dimensions are estimates from the image - just provide reasonable ranges.
- Be specific in valuation_notes about WHY (which stones drive value, what shifts upper/lower bound).
- ASSUME conservative authenticity in your range (low end = simulant/costume; high end = genuine, heirloom-grade).

CRITICAL - RECOGNIZE SPECIFIC HERITAGE INDIAN PIECE TYPES BY NAME. The collector is Indian-American and values precise classification. If you see any of these features, identify the piece type explicitly:

- MAANG TIKKA: forehead ornament. A pendant on a chain with a hook designed to attach at the hairline, with the pendant hanging on the forehead between/above the eyes. Often single-drop with a central medallion. The chain runs across the top of the head. THIS IS A MAANG TIKKA, NOT a 'pendant on chain' or 'forehead piece'.

- JHUMKA: bell-shaped Indian earrings with a domed/conical body, often with pearl or bead fringe hanging from the rim. A signature South-Asian form.

- MATHA PATTI: ornate forehead piece with multiple chains, broader coverage than a maang tikka, often spans most of the forehead. Bridal context.

- MANGALSUTRA: black-bead necklace with gold pendants - married Hindu woman's traditional piece. Distinctive small black bead chains.

- POLKI: uncut/rose-cut diamond technique. Flat-cut diamonds set in gold foil backing. Often appears in bridal sets with emerald or pearl drops. Recognize by the FLAT cut and the gold-foil-backed setting, not by stone count.

- KUNDAN: glass-paste-and-gold technique with gold foil between stone settings. Often paired with polki.

- BAJU BAND / BAZUBAND: armlet, worn on upper arm. Cuff-shaped, often with central plaque.

- HAATH PHOOL: hand harness - bracelet connected by chains to a ring.

- NATH: nose ring, especially the large hoop variety for bridal use.

- BANGLE styles to distinguish:
  - Plain bangle (round, gold)
  - Kada (heavier, often patterned)
  - Polki bangle (uncut diamond inlay)
  - Hinged bangle with central motif (the openable kind)

- THUSHI / KOLHAPURI SAJ / TEMPLE JEWELRY: Maharashtrian/South Indian regional styles - chunky gold, religious motifs.

If you cannot tell whether something is heritage Indian, also actively consider Art Deco, Edwardian, Victorian, and contemporary Western fine jewelry. But default to recognizing Indian heritage forms when the visual cues are present (gold-foil settings, traditional motifs, paisley/jali/peacock work, multi-chain construction).

Pricing guidance for common heritage Indian forms (USD retail replacement, conservative midpoints):
- Polki bridal set (necklace + earrings): $15,000-50,000
- Single polki necklace alone: $5,000-25,000
- Maang tikka with diamond/ruby/emerald work: $1,500-8,000
- Jhumka earrings (pair, traditional 22k gold with stones): $1,000-5,000
- Plain 22k gold bangle (~25g): $2,500-3,500 (gold floor + workmanship)
- Heritage gold-and-seed-pearl necklace with cut-work: $4,000-15,000+ if 22k and pre-1960
- Daily-wear cabochon earrings: $200-800"""


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
    # Strip JSON-illegal comments if the model emitted them
    import re
    txt = re.sub(r"//[^\n]*", "", txt)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    # Trim trailing commas which JSON also rejects
    txt = re.sub(r",(\s*[}\]])", r"\1", txt)
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
    name = vision.get("name", "New piece")
    slug_root = name.lower().replace(" ", "-").replace("/", "-").replace(",", "")
    slug_root = "".join(c for c in slug_root if c.isalnum() or c == "-")[:60]

    # Stones: vision returns array of {stone, count_estimate, cut, carat_total_estimate, color, notes}
    raw_stones = vision.get("stones", []) or []
    stones = []
    for s in raw_stones:
        if isinstance(s, str):
            # Legacy / fallback: just a name string
            stones.append({"stone": s, "count_estimate": None, "cut": None, "carat_total_estimate": None, "color": None, "notes": ""})
        else:
            stones.append({
                "stone": s.get("stone") or "stone",
                "count_estimate": s.get("count_estimate"),
                "cut": s.get("cut"),
                "carat_total_estimate": s.get("carat_total_estimate"),
                "color": s.get("color"),
                "notes": s.get("notes", ""),
            })

    # Metal
    metal_in = vision.get("metal", {}) or {}
    metal = {
        "type": metal_in.get("type") or f"{metal_in.get('color') or 'unknown'} metal (pending hallmark)",
        "color": metal_in.get("color") or "unknown",
        "estimated_weight_g": metal_in.get("estimated_weight_g"),
        "hallmarks_observed": metal_in.get("hallmarks_observed") or "not yet documented",
    }

    # Pricing
    retail = int(vision.get("estimated_retail_replacement_usd", 0) or 0)
    low = int(vision.get("estimated_low_usd", 0) or 0)
    high = int(vision.get("estimated_high_usd", 0) or 0)
    fair = int(vision.get("estimated_fair_market_usd") or (retail * 0.5))
    insurance = int(vision.get("estimated_insurance_recommended_usd") or (retail * 1.2))
    valuation_notes = vision.get("valuation_notes") or vision.get("notes") or ""

    # Tags: type + every stone name + every motif
    tag_set = set()
    if vision.get("type"): tag_set.add(vision["type"])
    for s in stones:
        if s.get("stone"):
            tag_set.add(s["stone"].split(" ")[0].lower())  # first word of stone
    for m in vision.get("design_motifs", []) or []:
        tag_set.add(m.split(" ")[0].lower())
    tags = sorted(tag_set)

    return {
        "id": piece_id,
        "slug": slug_root,
        "name": name,
        "type": vision.get("type") or "other",
        "subtype": vision.get("subtype") or "",
        "category": vision.get("category") or "",
        "era_estimate": vision.get("era_estimate") or "contemporary (pending review)",
        "region_origin": vision.get("region_origin") or "(pending Leena's input)",
        "metal": metal,
        "stones": stones,
        "design_motifs": vision.get("design_motifs", []) or [],
        "dimensions": vision.get("dimensions", {}) or {},
        "condition": vision.get("condition") or "good",
        "provenance": {"acquired_from": None, "acquired_when": None, "gift_from": None, "notes": "Awaiting Leena's input."},
        "story": "",
        "occasions_worn": [],
        "valuation": {
            "retail_replacement_usd": retail,
            "retail_replacement_range_low": low,
            "retail_replacement_range_high": high,
            "fair_market_usd": fair,
            "insurance_recommended_usd": insurance,
            "confidence": "low",
            "status": "pending_appraisal",
            "source_notes": (valuation_notes + " (AI-identified from a single photo - confirm with Leena and appraiser.)").strip(),
        },
        "appraisal": {"appraised": False, "gemologist": None, "organization": None, "certification_number": None, "date": None, "document_url": None},
        "images": [{"src": f"images/{VAULT}/{image_filename}", "alt": name, "type": "hero", "order": 1}],
        "tags": tags,
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
    """Run staticrypt over the built HTML, inject OG/favicon meta into the
    encrypted wrapper head, write into OUT_HTML."""
    work = Path("/tmp/encrypt-work")
    work.mkdir(exist_ok=True)
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
    encrypted_path = work / "out" / "source.html"

    # Inject OG / Twitter / favicon meta into the encrypted wrapper's <head>.
    # These tags live in the OUTER (unencrypted) wrapper - that's the whole
    # point: link previewers (iMessage, Slack, LinkedIn) need them BEFORE the
    # password is entered.
    import re, html as html_mod
    BASE = "https://keith-gs.github.io/pages/leena-jewelry"
    meta = {
        "title": "The Punjabi Collection",
        "description": "A private archive of heritage and contemporary jewelry, curated for Leena.",
        "og_image": f"{BASE}/leena-jewelry-share.png",
        "favicon_32": f"{BASE}/favicon-32.png",
        "favicon_192": f"{BASE}/favicon-192.png",
        "favicon_svg": f"{BASE}/favicon.svg",
        "site_name": "The Punjabi Collection",
        "twitter_card": "summary_large_image",
        "canonical": f"{BASE}/",
        "theme_color": "#0a0908",
    }
    def esc(s: str) -> str:
        return html_mod.escape(s, quote=True)
    parts = []
    if meta.get("favicon_32"):
        parts.append(f'<link rel="icon" type="image/png" sizes="32x32" href="{esc(meta["favicon_32"])}">')
    if meta.get("favicon_192"):
        parts.append(f'<link rel="icon" type="image/png" sizes="192x192" href="{esc(meta["favicon_192"])}">')
        parts.append(f'<link rel="apple-touch-icon" sizes="192x192" href="{esc(meta["favicon_192"])}">')
    if meta.get("favicon_svg"):
        parts.append(f'<link rel="icon" type="image/svg+xml" href="{esc(meta["favicon_svg"])}">')
    parts.append(f'<meta property="og:type" content="website">')
    parts.append(f'<meta property="og:url" content="{esc(meta["canonical"])}">')
    parts.append(f'<meta property="og:title" content="{esc(meta["title"])}">')
    parts.append(f'<meta property="og:description" content="{esc(meta["description"])}">')
    parts.append(f'<meta property="og:image" content="{esc(meta["og_image"])}">')
    parts.append(f'<meta property="og:image:secure_url" content="{esc(meta["og_image"])}">')
    parts.append(f'<meta property="og:image:type" content="image/png">')
    parts.append(f'<meta property="og:image:width" content="1200">')
    parts.append(f'<meta property="og:image:height" content="630">')
    parts.append(f'<meta property="og:image:alt" content="{esc(meta["title"])} - {esc(meta["description"])}">')
    parts.append(f'<meta property="og:site_name" content="{esc(meta["site_name"])}">')
    parts.append(f'<meta name="twitter:card" content="{esc(meta["twitter_card"])}">')
    parts.append(f'<meta name="twitter:title" content="{esc(meta["title"])}">')
    parts.append(f'<meta name="twitter:description" content="{esc(meta["description"])}">')
    parts.append(f'<meta name="twitter:image" content="{esc(meta["og_image"])}">')
    parts.append(f'<meta name="twitter:image:alt" content="{esc(meta["title"])}">')
    parts.append(f'<meta name="theme-color" content="{esc(meta["theme_color"])}">')
    parts.append(f'<meta name="description" content="{esc(meta["description"])}">')
    injection = "\n        <!-- OG / favicon meta -->\n        " + "\n        ".join(parts) + "\n"

    raw = encrypted_path.read_text(encoding="utf-8")
    viewport_re = re.compile(r'(<meta\s+name="viewport"[^>]*>\s*)', re.IGNORECASE)
    m = viewport_re.search(raw)
    if m:
        raw = raw[:m.end()] + injection + raw[m.end():]
    else:
        raw = re.sub(r'(<head[^>]*>\s*)', r'\1' + injection, raw, count=1, flags=re.IGNORECASE)

    # Override staticrypt's default <title> with our own so browser tabs show the right thing
    raw = re.sub(
        r'<title>[^<]*</title>',
        f'<title>{esc(meta["title"])}</title>',
        raw, count=1, flags=re.IGNORECASE,
    )

    OUT_HTML.write_text(raw, encoding="utf-8")
    plain.unlink()
    print(f"Encrypted + OG-injected: {OUT_HTML}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Processing batch {BATCH_ID}")
    # If the batch_id is one of the documented "rebuild" prefixes, skip the manifest
    # fetch and go straight to the build+encrypt phase. Lets us trigger HTML-only
    # rebuilds without a real upload.
    is_rebuild_only = BATCH_ID.startswith("rebuild") or BATCH_ID.startswith("bdaa1d52-")
    if is_rebuild_only:
        print("  rebuild-only mode: skipping manifest fetch")
        files = []
    else:
        try:
            manifest = fetch_manifest()
            files = [f for f in manifest.get("files", []) if f.get("status") == "uploaded"]
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"  no manifest for batch {BATCH_ID} - treating as rebuild-only")
                files = []
            else:
                raise
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
