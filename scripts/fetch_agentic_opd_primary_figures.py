from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "agentic-opd-reconstructed"
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg-full.js"


@dataclass(frozen=True)
class FigureSpec:
    slug: str
    arxiv_version: str
    source_path: str


FIGURES = (
    FigureSpec("sod", "2605.07725v3", "pic/framework7.png"),
    FigureSpec("stepopsd", "2605.27140v2", "stepopsd_review.png"),
    FigureSpec("turnopd", "2607.05804v2", "figures/TurnOPD_introduction.pdf"),
    FigureSpec("dagger", "2605.12913v1", "figures/scaling_4B_verified.pdf"),
    FigureSpec("guided-opd", "2606.15912v3", "figures/figure2.pdf"),
    FigureSpec("sage-opd", "2606.19659v1", "figure/diagram.png"),
    FigureSpec("reopd", "2607.04763v3", "figs/aopd.pdf"),
    FigureSpec("skill-sd", "2604.10674v1", "figures/Skill-SD.pdf"),
    FigureSpec("ar-opd", "2606.10385v1", "figure/fig2_AR-OPD_architecture.png"),
    FigureSpec("hint", "2606.11559v2", "figs/pipeline_cropped.pdf"),
    FigureSpec("oppo", "2605.21851v2", "figure/analysis.pdf"),
    FigureSpec("pbsd", "2606.09348v2", "figures/main.pdf"),
    FigureSpec("agentopsd", "2608.05987v1", "figures/final5.drawio.pdf"),
    FigureSpec("crest", "2608.13179v2", "doc/figures/method.png"),
    FigureSpec("taspo", "2608.31077v2", "overview.png"),
)


def download_source(arxiv_version: str) -> bytes:
    request = urllib.request.Request(
        f"https://arxiv.org/e-print/{arxiv_version}",
        headers={"User-Agent": "harness-opd-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def find_source(root: Path, relative_path: str) -> Path:
    exact = root / relative_path
    if exact.exists():
        return exact
    matches = list(root.rglob(Path(relative_path).name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {relative_path}, found {matches}")
    return matches[0]


def render_asset(source: Path, destination: Path, target_width: int = 1800) -> None:
    if source.suffix.lower() == ".pdf":
        document = fitz.open(source)
        page = document[0]
        scale = target_width / page.rect.width
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
        pixmap.save(destination)
        document.close()
        return

    with Image.open(source) as image:
        image = image.convert("RGBA")
        if image.width > target_width:
            height = round(image.height * target_width / image.width)
            image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        image.save(destination, optimize=True)


def fetch_all() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    grouped = {}
    for spec in FIGURES:
        grouped.setdefault(spec.arxiv_version, []).append(spec)

    with tempfile.TemporaryDirectory(prefix="agentic-opd-primary-") as temp_dir:
        temp_root = Path(temp_dir)
        for arxiv_version, specs in grouped.items():
            archive = download_source(arxiv_version)
            source_root = temp_root / arxiv_version
            source_root.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
                tar.extractall(source_root, filter="data")

            for spec in specs:
                source = find_source(source_root, spec.source_path)
                destination = OUTPUT_DIR / f"{spec.slug}.png"
                render_asset(source, destination)
                outputs.append(destination)

    expected = {f"{spec.slug}.png" for spec in FIGURES}
    for stale in OUTPUT_DIR.glob("*.png"):
        if stale.name not in expected:
            stale.unlink()
    mathjax = OUTPUT_DIR / "mathjax-tex-svg-full.js"
    if not mathjax.exists():
        request = urllib.request.Request(MATHJAX_URL, headers={"User-Agent": "harness-opd-research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            mathjax.write_bytes(response.read())
    outputs.append(mathjax)
    return outputs


if __name__ == "__main__":
    for output in fetch_all():
        print(output.relative_to(ROOT))