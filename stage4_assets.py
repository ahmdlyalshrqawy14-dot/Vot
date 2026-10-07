import os
import json
import uuid
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Stage4Assets")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "outputs" / "assets"

def _ensure_assets_dir() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return ASSETS_DIR

def _safe_filename(keyword: str) -> str:
    safe_kw = "".join([c if c.isalnum() else "_" for c in keyword.lower().strip()])
    return f"{safe_kw}.png"

def _fetch_from_wikimedia(keyword: str, output_path: Path) -> bool:
    """Fetch a transparent PNG/SVG from Wikimedia Commons and save it as PNG."""
    try:
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "generator": "search",
            "gsrsearch": f"{keyword} transparent icon filetype:png|svg",
            "gsrlimit": 3
        }
        headers = {"User-Agent": "VOT-App/1.0 (test@example.com)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            imageinfo = page_info.get("imageinfo", [{}])[0]
            img_url = imageinfo.get("url")
            if img_url and (img_url.lower().endswith(".png") or img_url.lower().endswith(".svg")):
                img_resp = requests.get(img_url, timeout=10)
                img_resp.raise_for_status()

                if img_url.lower().endswith(".png"):
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(img_resp.content))
                        img = img.convert("RGBA")
                        img.thumbnail((320, 320))
                        img.save(output_path, "PNG")
                        return True
                    except Exception as e:
                        logger.warning(f"PIL failed to process image: {e}")
                        output_path.write_bytes(img_resp.content)
                        return True
                else:
                    output_path.write_bytes(img_resp.content)
                    return True
        return False
    except Exception as e:
        logger.warning(f"Failed to fetch asset for '{keyword}': {e}")
        return False

def get_asset_for_keyword(keyword: str) -> Optional[Path]:
    if not keyword:
        return None

    assets_dir = _ensure_assets_dir()
    safe_name = _safe_filename(keyword)
    output_path = assets_dir / safe_name

    if output_path.exists():
        logger.info(f"Asset for '{keyword}' found in cache.")
        return output_path

    logger.info(f"Fetching asset for '{keyword}'...")
    success = _fetch_from_wikimedia(keyword, output_path)

    if success and output_path.exists():
        logger.info(f"Asset for '{keyword}' fetched successfully.")
        return output_path

    logger.warning(f"No asset found for '{keyword}'.")
    return None
