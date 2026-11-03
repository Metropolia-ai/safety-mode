#!/usr/bin/env python3
"""Compile the 10 example rules one by one against the local endpoint, timing each; writes glimmer/results.json.
Usage: SM_LLM_* env set; python run_live.py [--repeat N] [--tag NAME]
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path
ROOT = Path("/mnt/data/work/aivm-safety-mode"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)
import test_compiler_live as T
from safety_mode import CompileError, compile_rule, llm
from safety_mode.llm import config_from_env

class RamPeak(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.peak = 0; s.first = None; s.stop = threading.Event()
    def run(s):
        p = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if s.stop.is_set(): break
            try:
                used = int(line.split("RAM ")[1].split("/")[0]); s.peak = max(s.peak, used); s.first = s.first or used
            except Exception: pass
        p.kill()

def main():
    args = sys.argv[1:]; repeat = 1; tag = "run"
    if "--repeat" in args: i = args.index("--repeat"); repeat = int(args[i+1]); del args[i:i+2]
    if "--tag" in args: i = args.index("--tag"); tag = args[i+1]; del args[i:i+2]
    only = {int(a) for a in args} or None
    cfg = config_from_env(); T.OUT.mkdir(exist_ok=True)
    ram = RamPeak(); ram.start(); rows = []
    for n, text in [(n, t) for n, t in T.example_rules() if not only or n in only]:
        exp = json.loads((ROOT / "rules" / f"r{n:02d}.json").read_text(encoding="utf-8"))
        for r in range(repeat):
            u0 = len(llm.USAGE); t0 = time.perf_counter()
            try:
                got = compile_rule(text, exp["zones"], cfg, rule_id=exp["rule_id"]); diffs = T.compare(got, exp)
                (T.OUT / f"r{n:02d}.json").write_text(json.dumps(got, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except CompileError as e:
                got, diffs = None, [f"compile error: {str(e)[:300]}"]
            dt = time.perf_counter() - t0; calls = llm.USAGE[u0:]
            row = {"rule": n, "pass": not diffs, "kind": "-" if got is None else ("unsupported" if got["condition"] is None else "condition"),
                   "diffs": diffs, "s": round(dt, 1), "calls": len(calls),
                   "prompt_tokens": sum(c["prompt_tokens"] for c in calls), "completion_tokens": sum(c["completion_tokens"] for c in calls),
                   "reasoning_tokens": sum(c.get("reasoning_tokens", 0) for c in calls), "mode": calls[-1]["mode"] if calls else None,
                   "finish": [c.get("finish_reason") for c in calls]}
            rows.append(row); print(f"r{n:02d} {'PASS' if row['pass'] else 'FAIL'} {row['kind']:<11} {dt:6.1f}s calls={len(calls)} tok in/out={row['prompt_tokens']}/{row['completion_tokens']} {'; '.join(diffs)[:160]}", flush=True)
    ram.stop.set(); time.sleep(1.5)
    summary = {"tag": tag, "model": cfg.model, "base_url": cfg.base_url, "structured": cfg.structured, "reasoning": cfg.reasoning,
               "pass": sum(r["pass"] for r in rows), "n": len(rows), "total_s": round(sum(r["s"] for r in rows), 1),
               "ram_first_mb": ram.first, "ram_peak_mb": ram.peak, "rows": rows}
    out = Path("/mnt/data/work/glimmer") / f"results_{tag}.json"; out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"PASS {summary['pass']}/{summary['n']}  total {summary['total_s']} s  RAM peak {ram.peak} MB (first {ram.first})  -> {out}")

if __name__ == "__main__":
    import logging; logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s"); main()
