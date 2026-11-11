from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "AgenticOPD整理"
OUTPUT = ROOT / "documents" / "Agentic-OPD-development-roadmap-2026.10-facsimile.pdf"


def page_number(path: Path) -> int:
    if not (match := re.search(r"_(\d+)_", path.name)):
        raise ValueError(f"Cannot determine page number from {path.name}")
    return int(match.group(1))


def build_pdf() -> Path:
    sources = sorted(SOURCE_DIR.glob("*.jpg"), key=page_number)
    page_numbers = [page_number(path) for path in sources]
    if page_numbers != list(range(2, 12)):
        raise ValueError(f"Expected source pages 2-11, found {page_numbers}")

    pages = []
    for source in sources:
        with Image.open(source) as image:
            pages.append(image.convert("RGB"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        OUTPUT,
        "PDF",
        resolution=144.0,
        save_all=True,
        append_images=pages[1:],
        quality=95,
        subsampling=0,
        title="Agentic On-Policy Distillation 发展脉络（2026.10）",
    )
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())