"""Ask ngspice to draw its *own* plots of the two demo analyses.

This is deliberately not a re-plot of exported data: the deck runs ngspice's
``hardcopy`` command, so the axes, gridlines and curves are ngspice's renderer.
The build shipped in ``tools/Spice64`` has no PNG device but does have SVG, so
the SVG is rasterised afterwards with the same headless Chromium the UI
screenshots use.

    python docs/scripts/ngspice_plots.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ngspice_crosscheck import convert, find_ngspice  # noqa: E402

OUT = ROOT / "out" / "ngspice"

JOBS = [
    ("inverter_transient.net", "ng_transient.svg",
     "tran 20p 8n\n"
     "set hcopydevtype=svg\n"
     "hardcopy ng_transient.svg v(in) v(out)\n"
     "print all\n"),
    ("two_stage_opamp.net", "ng_ac_mag.svg",
     "ac dec 12 1 1G\n"
     "set hcopydevtype=svg\n"
     "hardcopy ng_ac_mag.svg vdb(out) vdb(n2) vdb(nz)\n"),
    ("two_stage_opamp.net", "ng_ac_phase.svg",
     "ac dec 12 1 1G\n"
     "set units=degrees\n"
     "set hcopydevtype=svg\n"
     "hardcopy ng_ac_phase.svg vp(out)\n"),
]


def rasterise(svgs: list[Path]) -> None:
    """SVG -> PNG through headless Chromium (no extra image dependency)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; SVGs left as-is")
        return
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 800},
                                device_scale_factor=2)
        for svg in svgs:
            # ngspice emits width="100%" height="100%" and a DOCTYPE pointing at
            # the W3C DTD.  Loading that file directly makes a full_page capture
            # hang, so inline the markup at its viewBox size instead.
            markup = svg.read_text(encoding="utf-8")
            markup = markup[markup.index("<svg"):]
            markup = markup.replace('width="100%" height="100%"',
                                    'width="1024" height="768"', 1)
            wrapper = svg.with_suffix(".html")
            wrapper.write_text(
                "<body style='margin:0;background:#ffffff'>" + markup,
                encoding="utf-8")
            page.goto(wrapper.resolve().as_uri())
            page.wait_for_timeout(500)
            png = svg.with_suffix(".png")
            page.locator("svg").screenshot(path=str(png))
            wrapper.unlink()
            print(f"  rasterised {png.relative_to(ROOT)}")
        browser.close()


def main() -> int:
    binary = find_ngspice()
    if binary is None:
        print("ngspice not found", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for example, target, control in JOBS:
        source = (ROOT / "examples" / example).read_text(encoding="utf-8")
        deck = convert(source, control)
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck_path = workdir / "deck.cir"
            deck_path.write_text(deck, encoding="utf-8")
            proc = subprocess.run([binary, "-b", str(deck_path)],
                                  capture_output=True, text=True,
                                  cwd=workdir, timeout=300)
            made = workdir / target
            if not made.is_file():
                print(f"{example}: ngspice produced no {target}")
                print((proc.stdout + proc.stderr)[-1200:])
                continue
            # Keep the deck too -- it is the evidence of what was actually run.
            shutil.copy(made, OUT / target)
            shutil.copy(deck_path, OUT / f"{target}.cir")
            produced.append(OUT / target)
            print(f"{example}: {target} ({made.stat().st_size} B)")

    rasterise(produced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
