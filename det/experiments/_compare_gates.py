"""Per-block admission comparison: r²>0.3 (legacy) vs cross-seed at multiple τ.

Reads `det/results/gate_tuning/*.json` and the original smoke
`det/results/smoke_sweep_2026-04-25/p7_uniform_qwton.json`, prints a
side-by-side per-block table and writes it to
`det/results/gate_tuning/per_block_compare.md`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DET = Path(__file__).resolve().parent.parent
SMOKE_DIR = DET / "results" / "smoke_sweep_2026-04-25"
TUNE_DIR = DET / "results" / "gate_tuning"


def load_per_block(p: Path) -> tuple[list, dict]:
    j = json.load(open(p))
    return j["calib"]["per_block"], j


def fmt_r2(blk: dict) -> str:
    if "r2_a" in blk:
        return f"{blk['r2_a']:+.2f}/{blk['r2_b']:+.2f}"
    if "r2" in blk:
        return f"{blk['r2']:+.2f}"
    return "?"


def fmt_cos(blk: dict) -> str:
    return f"{blk['cos_ab']:+.3f}" if "cos_ab" in blk else "—"


def main():
    sources: list[tuple[str, Path]] = [
        ("xseed_τ=0.50", SMOKE_DIR / "p7_uniform_qwton.json"),
        ("xseed_τ=0.65", TUNE_DIR / "tau_0.65_p7.json"),
        ("xseed_τ=0.75", TUNE_DIR / "tau_0.75_p7.json"),
        ("xseed_τ=0.85", TUNE_DIR / "tau_0.85_p7.json"),
        ("xseed_τ=0.90", TUNE_DIR / "tau_0.90_p7.json"),
        ("ref_r²>0.3", TUNE_DIR / "ref_r2_03_p7.json"),
    ]
    blocks_by_src: dict[str, list[dict]] = {}
    summary_by_src: dict[str, dict] = {}
    for name, p in sources:
        if not p.exists():
            print(f"[skip] {name}: {p} not present", file=sys.stderr)
            continue
        per_blk, full = load_per_block(p)
        blocks_by_src[name] = per_blk
        summary_by_src[name] = {
            "n_admit": sum(1 for b in per_blk if b.get("enabled", False)),
            "n_total": len(per_blk),
            "raw_bbox": full["results"]["sc_raw"]["bbox"]["AP"]
                       if full["results"].get("sc_raw") else None,
            "comp_bbox": full["results"]["sc_comp"]["bbox"]["AP"]
                        if full["results"].get("sc_comp") else None,
        }

    if not blocks_by_src:
        print("No source JSONs found yet; rerun after Step 1/2 finish.", file=sys.stderr)
        return

    n_blocks = max(len(v) for v in blocks_by_src.values())

    out_lines: list[str] = []
    out_lines.append("# Per-block admission decisions: r² gate vs cross-seed cosine\n")
    out_lines.append("Config: p7_uniform, n_calib=16, n_eval=10, sc_prec=7, comp_sc_prec=8, head_aligned, lookahead_veto.\n")
    out_lines.append("## Summary\n")
    out_lines.append("| source | admitted | raw bbox | comp bbox | Δ |")
    out_lines.append("|---|---:|---:|---:|---:|")
    for name in summary_by_src:
        s = summary_by_src[name]
        raw = f"{s['raw_bbox']:.2f}" if s['raw_bbox'] is not None else "—"
        comp = f"{s['comp_bbox']:.2f}" if s['comp_bbox'] is not None else "—"
        d = f"{s['comp_bbox']-s['raw_bbox']:+.2f}" if s['raw_bbox'] is not None and s['comp_bbox'] is not None else "—"
        out_lines.append(f"| {name} | {s['n_admit']}/{s['n_total']} | {raw} | {comp} | {d} |")

    out_lines.append("\n## Per-block table\n")
    hdr = "| blk |" + "".join(f" {n} |" for n in blocks_by_src.keys()) + " cos_ab | r²_a/b |"
    out_lines.append(hdr)
    out_lines.append("|---" * (1 + len(blocks_by_src) + 2) + "|")
    src_names = list(blocks_by_src.keys())
    for i in range(n_blocks):
        cells = [f"| {i:2d}"]
        cos_ref = None
        r2_ref = None
        for name in src_names:
            arr = blocks_by_src.get(name, [])
            if i >= len(arr):
                cells.append(" — ")
                continue
            blk = arr[i]
            mark = "Y" if blk.get("enabled") else "N"
            cells.append(f" {mark} ")
            if "cos_ab" in blk and cos_ref is None:
                cos_ref = blk
            if r2_ref is None:
                r2_ref = blk
        cos_str = fmt_cos(cos_ref) if cos_ref else "—"
        r2_str = fmt_r2(r2_ref) if r2_ref else "—"
        cells.append(f" {cos_str} | {r2_str} |")
        out_lines.append("|".join(cells).replace("||", "|"))

    # Disagreement counts
    out_lines.append("\n## Pairwise disagreement counts (admitted-by-A but not B)\n")
    src = src_names
    out_lines.append("| A \\ B |" + "".join(f" {b} |" for b in src))
    out_lines.append("|---" * (1 + len(src)) + "|")
    for a in src:
        row = [f"| {a} "]
        for b in src:
            ax = blocks_by_src[a]; bx = blocks_by_src[b]
            n = min(len(ax), len(bx))
            both_present = sum(1 for i in range(n) if ax[i].get("enabled") and not bx[i].get("enabled"))
            row.append(f" {both_present} ")
        out_lines.append("|".join(row) + "|")

    out_path = TUNE_DIR / "per_block_compare.md"
    out_path.write_text("\n".join(out_lines) + "\n")
    print("\n".join(out_lines))
    print(f"\n[wrote] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
