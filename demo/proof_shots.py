"""Screenshots of the proof layer on the demo page: one per (clip, example, second).

    /tmp/pw/bin/python demo/proof_shots.py https://safety-mode.vercel.app demo/screenshots/prod

Runs the example through the page (Check, Run), pauses the video on the given second and
saves the monitor (video with overlay, timeline strip and its caption) as proof-<clip>-<rule>-sNN.png.
Logs the caption under the strip and the state badge for each shot.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1].rstrip("/")
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
CASES = [("warehouse-team", "r3", [4, 7]), ("warehouse-team", "r2", [5, 13]), ("busy-warehouse", "r11", [2, 12]),
         ("busy-warehouse", "r2", [10]), ("warehouse-team", "r1", [2]), ("forklift-operator", "r2", [4]),
         ("busy-warehouse", "r3", [5])]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_context(viewport={"width": 1440, "height": 1000}).new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(BASE + "/", wait_until="networkidle")
    page.wait_for_function("document.getElementById('vid').readyState >= 2", timeout=30000)
    for clip, rid, secs in CASES:
        page.click(f"#clips button[data-clip='{clip}']")
        page.wait_for_function("document.getElementById('vid').readyState >= 2", timeout=30000)
        page.click(f"#examples button[data-rule='{rid}']")
        page.click("#check")
        page.wait_for_selector("#understood dl", timeout=60000)
        page.click("#run")
        page.wait_for_function("document.querySelectorAll('#ticks i').length > 0 && document.getElementById('tl-caption').textContent.length > 0", timeout=60000)
        page.wait_for_timeout(800)
        caption = page.inner_text("#tl-caption")
        print(f"{clip} {rid}: {caption}", flush=True)
        for s in secs:
            page.evaluate(f"""() => {{ const v = document.getElementById('vid'); v.pause(); v.currentTime = {s} + 0.5; }}""")
            page.wait_for_function("document.getElementById('vid').seeking === false", timeout=10000)
            page.wait_for_timeout(700)
            page.evaluate("() => window.dispatchEvent(new Event('resize'))")
            page.wait_for_timeout(200)
            name = f"proof-{clip}-{rid}-s{s:02d}.png"
            page.locator(".monitor").screenshot(path=str(OUT / name))
            print(f"  {name}", flush=True)
    print("console errors:", errors or "none")
    browser.close()
