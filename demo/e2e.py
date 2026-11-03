"""The demo page in a real headless Chrome, with screenshots.

    /tmp/pw/bin/python demo/e2e.py http://localhost:8090 demo/screenshots/local [--webcam] [--matrix] [--scout] [--scout-live] [--custom] [--report] [--live-rule]

Default path: the page loads (time to the first playable clip), then on each sample clip the default
example is checked and run through the page; the strip under the video must read idle -> candidate ->
alert -> idle with the alert on the expected seconds (EXPECTED, from test-footage/expected.md and
test-footage/candidates/REPORT-2026-10-30.md). Screenshots judge-<clip>-alert.png (paused inside the
alert) and judge-<clip>-end.png (whole strip and caption after the clip).
--matrix runs the four examples on the four clips and logs every alert in the feed. --scout opens the
Rule Scout panel on each clip (recorded traces, scout-<clip>.png) and tries one accepted rule.
--scout-live presses "Run Rule Scout live" once (about 8 Muse Spark calls). --custom compiles a typed
rule (1 Spark call unless cached), --report asks for the session report (1 Spark call).
Every path runs as a plain visitor: no key, no special header. The start checks that /api/health shows
the limits and what is left for this address, and that the page has no key field. --live-rule runs a rule
with an object SAM 3.1 has no cached results for on clips A and B (SAM 3.1 live on every frame, about 24 + 12
image requests, plus 2 compiles and up to 6 explanations), then runs it again (must reuse the session's
live results). --own-clip [--own-file=PATH] picks a local video in the "Your clip" tab (Playwright
set_input_files, default the Pexels desk clip test-footage/raw/pexels-7579570-desk-glasses-off.mp4), draws a zone with
mouse clicks, names it, compiles OWN_RULE, runs it (SAM 3.1 live on every second, up to 30), runs it again (must reuse
the session's results) and asks for the session report; screenshots own-clip-*.png. --desk [--desk-rule=cup|glasses|
headphones] [--desk-y4m=PATH] [--desk-frames=N] feeds a desk clip as the camera (y4m with the clip time burned in,
see demo/README.md), presses the "Try at your desk" chip (the cup rule draws its keyboard zone first), runs N webcam
frames and logs every frame: the engine's state and the JPEG sent to SAM (OUT/desk-<rule>/NN.jpg) to compare with the
clip time; screenshots desk-<rule>-*.png. Live calls count against the per-address hourly limits
(GET /api/health); the log ends with what is left.

Uses the installed Google Chrome (channel="chrome"): Playwright's Chromium has no H.264.
--webcam feeds /tmp/fakecam.y4m as the camera (ffmpeg from the operator clip, see demo/README.md);
that costs about 3 SAM 3.1 requests per frame.
"""
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1].rstrip("/")
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
WEBCAM = "--webcam" in sys.argv
MATRIX = "--matrix" in sys.argv
SCOUT = "--scout" in sys.argv or "--scout-live" in sys.argv
SCOUT_LIVE = "--scout-live" in sys.argv
CUSTOM = "--custom" in sys.argv
REPORT = "--report" in sys.argv
LIVE_RULE = "--live-rule" in sys.argv
OWN_CLIP = "--own-clip" in sys.argv
DESK = "--desk" in sys.argv
ONLY_NEW = "--only-new" in sys.argv


def opt(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


REPO = Path(__file__).resolve().parents[1]
OWN_FILE = opt("own-file", str(REPO / "test-footage/raw/pexels-7579570-desk-glasses-off.mp4"))
OWN_ZONE = ("desk", [(0.12, 0.22), (0.88, 0.22), (0.92, 1.0), (0.08, 1.0)])   # (title, corners in 0..1 of the picture)
OWN_RULE = "Everyone at the desk must wear glasses."
DESK_RULE = opt("desk-rule", "cup")
DESK_Y4M = opt("desk-y4m", f"/tmp/fakecam-desk-{DESK_RULE}.y4m")
DESK_FRAMES = int(opt("desk-frames", "16"))
# the cup clip (Pexels 6785501): the keyboard zone covers the laptop's right half and the table where the cup stands
DESK_ZONE = [(0.30, 0.62), (0.80, 0.62), (0.80, 0.97), (0.30, 0.97)]
# clip: rule with an object the cached detections do not have
LIVE_RULES = [("man-crosses-floor", "Nobody may stand on the shop floor without gloves."),
              ("forklift-crosses-aisle", "No pallet may stay on the walkway crossing for more than 5 seconds.")]
CLIPS = ["man-crosses-floor", "site-worker-no-vest", "forklift-crosses-aisle", "warehouse-team"]
EXAMPLES = ["r2", "r2-vest", "r3", "r8"]
# clip: (default example, alert intervals [(start_s, end_s)])
EXPECTED = {
    "man-crosses-floor": ("r2", [(7, 21)]),
    "site-worker-no-vest": ("r2-vest", [(2, 5)]),
    "forklift-crosses-aisle": ("r8", [(6, 7)]),
    "warehouse-team": ("r3", [(1, 6), (10, 12), (15, 15), (18, 22)]),
}
log = []
failures = []


def step(msg):
    log.append(f"{time.strftime('%H:%M:%S')} {msg}")
    print(log[-1], flush=True)


def check(ok, msg):
    step(("OK   " if ok else "FAIL ") + msg)
    if not ok:
        failures.append(msg)


def shot(page, name):
    page.screenshot(path=str(OUT / f"{name}.png"))
    step(f"screenshot {name}.png")


STRIP_JS = """[...document.querySelectorAll('#ticks i')].map(i => i.classList.contains('cand') ? 'c'
  : i.classList.contains('gap') ? 'G' : i.classList.contains('ok') ? '.' : i.classList.contains('ne') ? '?'
  : i.style.background ? 'A' : ' ').join('')"""


def phases(strip):
    """'..ccAAAG A..' -> ['idle', 'candidate', 'alert', 'idle'] (gap = alert kept open)."""
    name = {".": "idle", "c": "candidate", "A": "alert", "G": "alert", "?": "unknown", " ": "empty"}
    out = []
    for ch in strip:
        n = name[ch]
        if not out or out[-1] != n:
            out.append(n)
    return out


def alert_intervals(strip):
    return [(m.start(), m.end() - 1) for m in re.finditer(r"[AG]+", strip)]


def pick_clip(page, clip):
    page.click(f"#clips button[data-clip='{clip}']")
    if clip != "webcam":
        page.wait_for_function("document.getElementById('vid').readyState >= 2", timeout=30000)


def health(page):
    """GET /api/health as a plain visitor (no header): the limits and what is left for this address."""
    r = page.request.get(BASE + "/api/health")
    return r.status, (r.json() if r.ok else {})


def check_and_run(page, rid=None, rate=1.0):
    if rid:
        page.click(f"#examples button[data-rule='{rid}']")
    page.click("#check")
    page.wait_for_function("!document.getElementById('check').disabled && document.querySelector('#understood dl, #understood .unsupported')", timeout=60000)
    if page.is_disabled("#run"):
        return False
    page.evaluate(f"const v = document.getElementById('vid'); v.defaultPlaybackRate = {rate}; v.playbackRate = {rate}")
    page.click("#run")
    page.wait_for_function("!document.getElementById('run').classList.contains('busy') && document.querySelectorAll('#ticks i').length > 0", timeout=90000)
    page.evaluate(f"document.getElementById('vid').playbackRate = {rate}")
    return True


def draw_zone(page, corners):
    """Real mouse clicks on the picture: corners in 0..1 of the video content, double-click on the last one."""
    rect = page.evaluate("""() => { const v = document.getElementById('vid'), o = document.getElementById('overlay');
        const r = o.getBoundingClientRect(), vw = v.videoWidth || 16, vh = v.videoHeight || 9;
        const k = Math.min(r.width / vw, r.height / vh), w = vw * k, h = vh * k;
        return { x: r.left + (r.width - w) / 2, y: r.top + (r.height - h) / 2, w, h }; }""")
    pts = [(rect["x"] + cx * rect["w"], rect["y"] + min(cy, 0.999) * rect["h"]) for cx, cy in corners]
    for x, y in pts[:-1]:
        page.mouse.click(x, y)
        page.wait_for_timeout(120)
    page.mouse.dblclick(*pts[-1])
    page.wait_for_timeout(300)


def detect_count(timings, n0):
    return sum(1 for u, _ in timings[n0:] if u.startswith("detect"))


def own_clip(page, timings):
    """The user's own clip: pick a file, draw and name a zone, compile, run (all prompts live), run again, report."""
    page.click("#clear-alerts")
    page.evaluate("Object.keys(localStorage).filter(k => k.startsWith('sm-zones:own')).forEach(k => localStorage.removeItem(k))")
    page.click("#clips button[data-clip='own']")
    page.wait_for_selector(".ownpick", state="visible", timeout=10000)
    check("nothing is uploaded" in page.inner_text(".ownpick"), "own clip: the tab says the file stays in the browser: "
          + page.inner_text(".ownpick").replace("\n", " ")[:200])
    check("sample clips only" in page.evaluate("document.getElementById('scout-body').textContent"),
          "own clip: Rule Scout says it runs on the sample clips: " + page.inner_text("#scout-sum"))
    shot(page, "own-clip-01-pick")
    t = time.time()
    page.set_input_files("#own-file", OWN_FILE)
    page.wait_for_function("document.getElementById('vid').readyState >= 2 && document.getElementById('setup').textContent.includes('Your clip.')", timeout=30000)
    step(f"own clip: {Path(OWN_FILE).name} playable in {time.time() - t:.2f} s; setup {page.inner_text('#setup').replace(chr(10), ' | ')!r}")
    page.click("#zone-draw")
    draw_zone(page, OWN_ZONE[1])
    page.wait_for_selector("#setup input[data-i='0']", timeout=5000)
    page.fill("#setup input[data-i='0']", OWN_ZONE[0])
    page.press("#setup input[data-i='0']", "Tab")
    page.wait_for_timeout(300)
    check(page.input_value("#setup input[data-i='0']") == OWN_ZONE[0], f"own clip: zone drawn with clicks and named {OWN_ZONE[0]!r}")
    stored = page.evaluate("JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k => k.startsWith('sm-zones:own'))) || '[]')")
    check(len(stored) == 1 and stored[0]["title"] == OWN_ZONE[0] and len(stored[0]["polygon"]) == len(OWN_ZONE[1]),
          f"own clip: zone kept in localStorage {stored}")
    shot(page, "own-clip-02-zone")
    page.fill("#rule-text", OWN_RULE)
    t = time.time()
    page.click("#check")
    page.wait_for_function("!document.getElementById('check').disabled && document.querySelector('#understood dl, #understood .unsupported')", timeout=90000)
    t_compile = time.time() - t
    card = page.inner_text("#understood")
    step(f"own clip: compiled in {t_compile:.1f} s: {card[:500]!r}")
    check(OWN_ZONE[0] in card.lower() and "desk_" not in card and "SAM 3.1 runs live" in card,
          "own clip: the card names the user's zone and says SAM runs live")
    if page.is_disabled("#run"):
        check(False, "own clip: rule not runnable")
        return
    page.locator("#understood").scroll_into_view_if_needed()
    shot(page, "own-clip-03-rule")
    for attempt in ("first", "again"):
        n0 = len(timings)
        page.evaluate("const v = document.getElementById('vid'); v.defaultPlaybackRate = 4; v.playbackRate = 4")
        t = time.time()
        page.click("#run")
        seen = []
        while page.evaluate("document.getElementById('run').classList.contains('busy')") and time.time() - t < 240:
            m = page.inner_text("#rule-msg")
            if m.startswith("SAM 3.1 live") and (not seen or seen[-1] != m):
                seen.append(m)
                if attempt == "first" and len(seen) == 3:
                    shot(page, "own-clip-04-progress")
            page.wait_for_timeout(150)
        t_run = time.time() - t
        strip = page.evaluate(STRIP_JS)
        det = detect_count(timings, n0)
        det_t = [d for u, d in timings[n0:] if u.startswith("detect")]
        step(f"own clip ({attempt}): Run -> strip {t_run:.1f} s, {det} /api/detect" + (f" ({min(det_t):.1f}-{max(det_t):.1f} s each)" if det_t else "")
             + f"; strip {strip!r}; note {page.inner_text('#tl-note')!r}; caption {page.inner_text('#tl-caption')!r}")
        check(len(strip) > 0 and " " not in strip, f"own clip ({attempt}): strip has every second")
        if attempt == "again":
            check(det == 0 and "reused" in page.inner_text("#rule-msg"), f"own clip: second run reuses the session's SAM results ({det} requests)")
            break
        ivs = alert_intervals(strip)
        check(bool(ivs), f"own clip: alert(s) {ivs}")
        if ivs:
            page.wait_for_function(f"document.getElementById('vid').currentTime > {ivs[0][0]} + 1", timeout=60000)
            page.evaluate(f"const v = document.getElementById('vid'); v.pause(); v.currentTime = {ivs[0][0]} + 0.5")
            page.wait_for_function("document.getElementById('vid').seeking === false", timeout=10000)
            page.wait_for_timeout(800)
            shot(page, "own-clip-05-alert")
            page.evaluate("document.getElementById('vid').play()")
        page.wait_for_function("document.getElementById('vid').ended || document.getElementById('vid').paused", timeout=90000)
        page.wait_for_timeout(500)
        page.wait_for_function("![...document.querySelectorAll('#feed .alert .src')].some(s => s.textContent.includes('writing'))", timeout=60000)
        feed = page.evaluate("[...document.querySelectorAll('#feed .alert')].map(a => a.querySelector('h3').textContent + ' | ' + a.querySelector('.meta').textContent + ' | ' + a.querySelector('p').textContent + ' | ' + a.querySelector('.src').textContent)")
        step(f"own clip: feed {feed}")
        check(len(feed) == len(ivs) and all("Your clip" in f for f in feed), f"own clip: {len(feed)} alert(s) in the feed, labelled Your clip")
        check(any(OWN_ZONE[0] in f.lower() for f in feed), "own clip: the alert names the user's zone")
        shot(page, "own-clip-06-end")
    t = time.time()
    page.click("#open-report")
    page.wait_for_function("!document.querySelector('#report-body .wait')", timeout=90000)
    rep = page.inner_text("#report-body")
    step(f"own clip: report in {time.time() - t:.1f} s: {rep[:1500]!r}")
    check("Your clip" in rep or OWN_ZONE[0] in rep.lower(), "own clip: the report names the clip or the zone")
    check(not re.search(r"\bdesk_\w*|\bzone_\d", rep), "own clip: no zone keys in the report")
    shot(page, "own-clip-07-report")
    page.click("#close-report")
    page.evaluate("const v = document.getElementById('vid'); v.pause(); v.defaultPlaybackRate = 1; v.playbackRate = 1")


def desk(page, timings):
    """Webcam at a desk: a desk clip as the fake camera, the chip rule, N frames; per-frame log and JPEGs."""
    page.click("#clear-alerts")
    page.evaluate("localStorage.removeItem('sm-zones:webcam')")
    jpegs = []
    out = OUT / f"desk-{DESK_RULE}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    def on_req(r):
        if r.url.endswith("/api/detect") and r.method == "POST":
            import base64, json
            b = json.loads(r.post_data or "{}").get("jpeg", "")
            jpegs.append(base64.b64decode(b.split(",", 1)[-1]))
    page.on("request", on_req)
    pick_clip(page, "webcam")
    page.wait_for_selector("#desk", state="visible", timeout=5000)
    check(page.is_visible("#desk"), "desk: 'Try at your desk' chips visible on the webcam tab: " + page.inner_text("#desk").replace("\n", " "))
    page.click("#cam-start")
    page.wait_for_function("document.getElementById('vid').readyState >= 2", timeout=20000)
    page.wait_for_timeout(500)
    t = time.time()
    page.click(f"#desk button[data-desk='{DESK_RULE}']")
    if DESK_RULE == "cup":
        page.wait_for_function("document.getElementById('screen').classList.contains('drawing')", timeout=5000)
        draw_zone(page, DESK_ZONE)
        check(page.evaluate("JSON.parse(localStorage.getItem('sm-zones:webcam') || '[]').map(z => z.title).join()") == "Keyboard zone",
              "desk: keyboard zone drawn on the webcam picture and stored")
    page.wait_for_function("!document.getElementById('check').disabled && document.querySelector('#understood dl, #understood .unsupported')", timeout=90000)
    card = page.inner_text("#understood")
    step(f"desk {DESK_RULE}: rule {page.input_value('#rule-text')!r} compiled in {time.time() - t:.1f} s: {card[:400]!r}")
    if page.is_disabled("#run"):
        check(False, f"desk {DESK_RULE}: not runnable")
        return
    shot(page, f"desk-{DESK_RULE}-ready")
    t = time.time()
    page.click("#run")
    first_alert = None
    alert_shot = False
    while time.time() - t < DESK_FRAMES * 2 + 60:
        n = page.evaluate("document.querySelectorAll('#ticks i[title]').length")
        if first_alert is None and page.evaluate("[...document.querySelectorAll('#feed .alert .meta')].some(m => m.textContent.startsWith('Webcam'))"):
            first_alert = time.time() - t
        if first_alert is not None and not alert_shot:
            page.wait_for_timeout(1500)
            shot(page, f"desk-{DESK_RULE}-alert")
            alert_shot = True
        if n >= DESK_FRAMES:
            break
        page.wait_for_timeout(300)
    if page.evaluate("document.getElementById('run').textContent.startsWith('Stop')"):
        page.click("#run")
    page.wait_for_timeout(500)
    titles = page.evaluate("[...document.querySelectorAll('#ticks i[title]')].map(i => i.title)")
    strip = page.evaluate(STRIP_JS)
    check(first_alert is not None, f"desk {DESK_RULE}: webcam alert " + (f"{first_alert:.1f} s after Start" if first_alert else "never came"))
    step(f"desk {DESK_RULE}: {len(titles)} frames, strip {strip[:len(titles)]!r}; note {page.inner_text('#tl-note')!r}; caption {page.inner_text('#tl-caption')!r}")
    for i, (title, jpg) in enumerate(zip(titles, jpegs)):
        (out / f"{i:02d}.jpg").write_bytes(jpg)
        step(f"desk {DESK_RULE} frame {i:02d}: {title}")
    page.wait_for_function("![...document.querySelectorAll('#feed .alert .src')].some(s => s.textContent.includes('writing'))", timeout=60000)
    feed = page.evaluate("[...document.querySelectorAll('#feed .alert')].map(a => a.querySelector('h3').textContent + ' | ' + a.querySelector('.meta').textContent + ' | ' + a.querySelector('p').textContent)")
    step(f"desk {DESK_RULE}: feed {feed}")
    shot(page, f"desk-{DESK_RULE}-end")
    page.remove_listener("request", on_req)


with sync_playwright() as p:
    args = ["--autoplay-policy=no-user-gesture-required"]
    if WEBCAM or DESK:
        args += ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                 f"--use-file-for-fake-video-capture={DESK_Y4M if DESK else '/tmp/fakecam.y4m'}"]
    browser = p.chromium.launch(channel="chrome", headless=True, args=args)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    if WEBCAM or DESK:
        ctx.grant_permissions(["camera"], origin=BASE)
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    timings = []      # (path, seconds) of every API request
    page.on("requestfinished", lambda r: timings.append((r.url.split("/api/")[-1], r.timing["responseEnd"] / 1000))
            if "/api/" in r.url and r.timing.get("responseEnd", -1) > 0 else None)
    t0 = time.time()
    page.goto(BASE + "/", wait_until="domcontentloaded")
    t_dom = time.time() - t0
    page.wait_for_function("document.getElementById('vid').readyState >= 2", timeout=30000)
    t_clip = time.time() - t0
    page.wait_for_load_state("networkidle")
    nav = page.evaluate("JSON.stringify(performance.getEntriesByType('navigation')[0].toJSON())")
    step(f"page load: DOM {t_dom:.2f} s, first clip playable {t_clip:.2f} s, network idle {time.time() - t0:.2f} s; "
         f"navigation entry {nav[:200]}")
    check(page.get_attribute("#clips button[aria-pressed='true']", "data-clip") == CLIPS[0], "first clip selected on load")
    check(page.eval_on_selector_all("#clips button", "bs => bs.map(b => b.dataset.clip).join(',')")
          == ",".join(CLIPS + ["webcam", "own"]), "clip selector order: " + page.inner_text("#clips").replace("\n", " | "))
    shot(page, "01-start")
    st, h = health(page)
    lim = h.get("limits") or {}
    check(st == 200 and lim.get("spark_calls_per_hour_per_ip", 0) > 0 and lim.get("sam_frames_per_hour_per_ip", 0) > 0
          and "remaining" in h and "exempt" not in h,
          f"/api/health without a key shows the limits: {lim}; remaining for this address {h.get('remaining')}; "
          f"today {h.get('remaining_today')}")
    footer = page.inner_text("footer")
    check(not page.query_selector("footer input") and "key" not in footer.lower(), f"no key field on the page: footer {footer!r}")

    if not ONLY_NEW:   # --only-new: skip the default path (local checks of --own-clip / --desk)
        # every sample clip with its default example: alert seconds and idle -> candidate -> alert -> idle
        for clip in CLIPS:
            rid, want = EXPECTED[clip]
            pick_clip(page, clip)
            try:   # the default example is set after the clip's zones load (async)
                page.wait_for_function(f"document.querySelector(\"#examples button[aria-pressed='true']\")?.dataset.rule === '{rid}'", timeout=10000)
            except Exception:
                pass
            cur = page.get_attribute("#examples button[aria-pressed='true']", "data-rule")
            check(cur == rid, f"{clip}: default example {cur} (want {rid}): {page.input_value('#rule-text')!r}")
            t = time.time()
            check_and_run(page)
            step(f"{clip}: check + run {time.time() - t:.1f} s")
            strip = page.evaluate(STRIP_JS)
            got = alert_intervals(strip)
            check(got == want, f"{clip} {rid}: alerts {got} (want {want}), strip {strip!r}, note {page.inner_text('#tl-note')!r}")
            ph = phases(strip)
            if clip == "warehouse-team":
                check(ph[0] == "idle" and ph[-1] == "alert" and "candidate" not in ph[:1],
                      f"{clip}: phases {ph} (alert 1 s rule: no candidate seconds; the clip ends inside the last alert)")
            elif clip == "site-worker-no-vest":
                check(ph[:2] == ["candidate", "alert"] and "idle" in ph[2:],
                      f"{clip}: phases {ph} (violation already on at 0 s: candidate -> alert -> idle)")
            else:
                check(ph[:4] == ["idle", "candidate", "alert", "idle"], f"{clip}: phases {ph}")
            step(f"{clip}: caption {page.inner_text('#tl-caption')!r}")
            mid = (want[0][0] + want[0][1]) / 2
            page.wait_for_function(f"document.getElementById('vid').currentTime > {mid} + 0.3 || document.getElementById('vid').ended", timeout=60000)
            page.evaluate(f"const v = document.getElementById('vid'); v.pause(); v.currentTime = {int(mid)} + 0.5")
            page.wait_for_function("document.getElementById('vid').seeking === false", timeout=10000)
            page.wait_for_timeout(800)
            shot(page, f"judge-{clip}-alert")
            page.evaluate("const v = document.getElementById('vid'); v.playbackRate = 4; v.play()")
            page.wait_for_function("document.getElementById('vid').ended", timeout=60000)
            page.wait_for_timeout(500)
            feed = page.evaluate("[...document.querySelectorAll('#feed .alert')].map(a => a.querySelector('h3').textContent + ' | ' + a.querySelector('.meta').textContent)")
            step(f"{clip}: feed now {feed[:len(want)]}")
            page.wait_for_function("![...document.querySelectorAll('#feed .alert .src')].some(s => s.textContent.includes('writing'))", timeout=60000)
            expl = page.evaluate("[...document.querySelectorAll('#feed .alert')].slice(-1).concat([...document.querySelectorAll('#feed .alert')].slice(0, 1))"
                                 ".map(a => (a.querySelector('p') || {}).textContent)")
            step(f"{clip}: explanation of the newest alert {expl[-1]!r}")
            if clip == "man-crosses-floor":
                check("right side" in (expl[-1] or ""), f"{clip}: explanation places person #0 on the right at 00:07")
            shot(page, f"judge-{clip}-end")

        # refusal card: example 4 on clip A (no walkway zone); zone keys must not show
        pick_clip(page, "man-crosses-floor")
        page.click("#examples button[data-rule='r8']")
        page.click("#check")
        page.wait_for_selector("#understood .unsupported", timeout=60000)
        card = page.inner_text("#understood")
        keys = [k for k in ("past_gate", "pedestrian_lane", "yellow_zone", "rack_zone", "fire_exit_zone") if k in card + page.inner_text("#rule-msg")]
        check(not keys, f"refusal card on man-crosses-floor, example 4, no zone keys {keys}: {card[:300]!r}")
        page.locator("#understood").scroll_into_view_if_needed()
        shot(page, "judge-refusal-man-crosses-floor-r8")
    if LIVE_RULE:
        page.click("#clear-alerts")
        for clip, text in LIVE_RULES:
            for attempt in ("first", "again"):
                n0 = len(timings)
                t_compile = 0.0
                if attempt == "first":      # "again": the same compiled rule, Run pressed a second time
                    pick_clip(page, clip)
                    page.fill("#rule-text", text)
                    t = time.time()
                    page.click("#check")
                    page.wait_for_function("!document.getElementById('check').disabled && document.querySelector('#understood dl, #understood .unsupported')", timeout=90000)
                    t_compile = time.time() - t
                card = page.inner_text("#understood")
                if page.is_disabled("#run"):
                    check(False, f"live rule {clip}: not runnable: {card[:400]!r}")
                    break
                check("SAM 3.1 runs live" in card, f"live rule {clip} ({attempt}): card says SAM runs live; objects "
                      + page.inner_text("#understood dd").replace("\n", " "))
                page.evaluate("const v = document.getElementById('vid'); v.defaultPlaybackRate = 4; v.playbackRate = 4")
                feed0 = page.evaluate("document.querySelectorAll('#feed .alert').length")
                t = time.time()
                page.click("#run")
                seen, progress_shot = [], False
                while page.evaluate("document.getElementById('run').classList.contains('busy')") and time.time() - t < 240:
                    m = page.inner_text("#rule-msg")
                    if m.startswith("SAM 3.1 live") and (not seen or seen[-1] != m):
                        seen.append(m)
                        if not progress_shot and re.search(r"live: [1-9]\d*/", m):
                            shot(page, f"live-rule-{clip}-progress")
                            progress_shot = True
                    page.wait_for_timeout(150)
                t_first = time.time() - t
                strip = page.evaluate(STRIP_JS)
                detects = sum(1 for u, _ in timings[n0:] if u.startswith("detect"))
                det_times = [d for u, d in timings[n0:] if u.startswith("detect")]
                step(f"live rule {clip} ({attempt}): compile {t_compile:.1f} s, Run -> strip {t_first:.1f} s, "
                     f"{detects} /api/detect requests" + (f" ({min(det_times):.1f}-{max(det_times):.1f} s each)" if det_times else "")
                     + f"; progress {seen[:2] + seen[-1:]}; msg {page.inner_text('#rule-msg')!r}")
                check(len(strip) > 0 and " " not in strip, f"live rule {clip} ({attempt}): strip {strip!r}, note {page.inner_text('#tl-note')!r}")
                check("SAM 3.1 live" in page.inner_text("#stamp"), f"live rule {clip}: video label {page.inner_text('#stamp')!r}")
                if attempt == "again":
                    check(detects == 0 and "reused" in page.inner_text("#rule-msg"), f"live rule {clip}: second run reuses the session's live results ({detects} SAM requests)")
                step(f"live rule {clip}: caption {page.inner_text('#tl-caption')!r}")
                if attempt == "first":
                    page.wait_for_timeout(1200)
                    shot(page, f"live-rule-{clip}-run")
                n_alerts = len(alert_intervals(strip))
                page.wait_for_function("document.getElementById('vid').ended", timeout=90000)
                page.wait_for_timeout(500)
                feed = page.evaluate("[...document.querySelectorAll('#feed .alert')].map(a => a.querySelector('h3').textContent + ' | ' + a.querySelector('.meta').textContent + ' | ' + a.querySelector('.src').textContent)")
                if attempt == "first":
                    check(len(feed) - feed0 == n_alerts, f"live rule {clip}: {n_alerts} alert(s) on the strip, {len(feed) - feed0} new in the feed {feed[:n_alerts]}")
                    if n_alerts:
                        page.wait_for_function("![...document.querySelectorAll('#feed .alert .src')].some(s => s.textContent.includes('writing'))", timeout=60000)
                        page.evaluate(f"const v = document.getElementById('vid'); v.pause(); v.currentTime = {alert_intervals(strip)[0][0]} + 0.5")
                        page.wait_for_function("document.getElementById('vid').seeking === false", timeout=10000)
                        page.wait_for_timeout(800)
                    page.locator("#understood").scroll_into_view_if_needed()
                    shot(page, f"live-rule-{clip}-result")
        page.evaluate("const v = document.getElementById('vid'); v.defaultPlaybackRate = 1; v.playbackRate = 1")
    if OWN_CLIP:
        own_clip(page, timings)
    if DESK:
        desk(page, timings)
    if CUSTOM:
        pick_clip(page, "warehouse-team")
        page.fill("#rule-text", "Everyone in the rack zone must wear a safety vest.")
        t = time.time()
        page.click("#check")
        page.wait_for_function("!document.getElementById('check').disabled && document.querySelector('#understood dl, #understood .unsupported')", timeout=90000)
        step(f"custom rule compiled in {time.time() - t:.1f} s")
        shot(page, "07-custom-rule")
    if REPORT:
        t = time.time()
        page.click("#open-report")
        page.wait_for_function("!document.querySelector('#report-body .wait')", timeout=90000)
        rep = page.inner_text("#report-body")
        step(f"report in {time.time() - t:.1f} s: {rep[:1500]!r}")
        check("00:07" in rep and "00:21" in rep, "report gives the A alert as 00:07-00:21")
        shot(page, "09-report")
        page.click("#close-report")
    if WEBCAM:
        pick_clip(page, "webcam")
        page.click("#cam-start")
        page.wait_for_function("document.getElementById('vid').readyState >= 2", timeout=20000)
        page.click("#examples button[data-rule='r2']")
        page.click("#check")
        page.wait_for_function("!document.getElementById('run').disabled", timeout=60000)
        shot(page, "10-webcam-ready")
        t = time.time()
        page.click("#run")
        page.wait_for_function("[...document.querySelectorAll('#feed .alert .meta')].some(m => m.textContent.startsWith('Webcam'))", timeout=60000)
        check(True, f"webcam alert {time.time() - t:.1f} s after Start")
        page.wait_for_function("![...document.querySelectorAll('#feed .alert .src')].some(s => s.textContent.includes('writing'))", timeout=60000)
        shot(page, "11-webcam-alert")
        page.click("#run")  # stop
        step(f"webcam: {page.inner_text('#rule-msg')!r}")
    if MATRIX:
        page.click("#clear-alerts")
        for clip in CLIPS:
            pick_clip(page, clip)
            for rid in EXAMPLES:
                if not check_and_run(page, rid, rate=4):
                    step(f"matrix {clip} {rid}: not runnable: {page.inner_text('#understood')[:160]!r}")
                    continue
                n0 = page.evaluate("document.querySelectorAll('#feed .alert').length")
                page.wait_for_function("document.getElementById('vid').ended", timeout=60000)
                page.wait_for_timeout(300)
                feed = page.evaluate("[...document.querySelectorAll('#feed .alert')].map(a => a.querySelector('h3').textContent + ' | ' + a.querySelector('.meta').textContent)")
                strip = page.evaluate(STRIP_JS)
                step(f"matrix {clip} {rid}: {page.inner_text('#tl-note')!r} strip {strip!r}; caption {page.inner_text('#tl-caption')!r}; "
                     f"feed {feed[:max(0, len(feed) - n0)]}")
        page.evaluate("const v = document.getElementById('vid'); v.defaultPlaybackRate = 1; v.playbackRate = 1")
    if SCOUT:
        page.set_viewport_size({"width": 1500, "height": 1300})   # the side column shows the whole panel
        for clip in CLIPS:
            pick_clip(page, clip)
            if not page.evaluate("document.getElementById('scout').open"):
                page.click("#scout summary")
            page.wait_for_function("document.querySelector('#scout-body .trace')", timeout=20000)
            acc = page.evaluate("[...document.querySelectorAll('#scout-body .scout-rules b')].map(b => b.textContent)")
            check(True, f"scout {clip}: {page.inner_text('#scout-sum')!r}; accepted: {acc}")
            page.wait_for_timeout(300)
            shot(page, f"scout-{clip}")
        # "Try this rule" on the first clip with an accepted rule: compiles from the cache and runs
        for clip in CLIPS:
            pick_clip(page, clip)
            if page.query_selector("#scout-body .scout-rules button"):
                page.click("#clear-alerts")
                page.click("#scout-body .scout-rules button")
                page.wait_for_function("!document.getElementById('run').disabled", timeout=60000)
                page.click("#run")
                page.wait_for_function("document.querySelectorAll('#ticks i').length > 0", timeout=60000)
                page.wait_for_timeout(1500)
                step(f"scout rule tried on {clip}: {page.input_value('#rule-text')!r} -> {page.inner_text('#tl-note')!r}")
                shot(page, f"scout-{clip}-try-rule")
                break
    if SCOUT_LIVE:
        pick_clip(page, CLIPS[0])
        n0 = len(timings)
        t = time.time()
        page.click("#scout-live")
        page.wait_for_function("!document.querySelector('#scout-body .busy')", timeout=600000)
        steps = [d for u, d in timings[n0:] if u.startswith("scout/step")]
        step(f"live Rule Scout on {CLIPS[0]}: {time.time() - t:.1f} s, {len(steps)} requests, "
             f"slowest {max(steps or [0]):.1f} s; {page.inner_text('#scout-sum')!r}")
        shot(page, f"scout-live-{CLIPS[0]}")
    by = {}
    for u, d in timings:
        by.setdefault(u.split("?")[0], []).append(d)
    step("API times: " + "; ".join(f"{u} n={len(v)} {min(v):.2f}-{max(v):.2f} s" for u, v in sorted(by.items())))
    step(f"left for this address this hour: {health(page)[1].get('remaining')}")
    check(not errors, f"console errors: {errors or 'none'}")
    step(f"failures: {failures or 'none'}")
    (OUT / "e2e-log.txt").write_text("\n".join(log) + "\n")
    browser.close()
    sys.exit(1 if failures else 0)
