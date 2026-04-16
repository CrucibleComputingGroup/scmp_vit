"""Aggregate all results/overnight_*.json into a comparison table."""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"

# Reference baselines from the README (N=500, sc_prec=8, RTX 4080).
BASELINES = {
    "qk_only":       {"fp": 0.846, "sc_raw": 0.736, "fp_comp": 0.834, "sc_comp_naive": 0.810},
    "qk_av":         {"fp": 0.846, "sc_raw": 0.718, "fp_comp": 0.822, "sc_comp_naive": None},
    "full_attn":     {"fp": 0.846, "sc_raw": 0.696, "fp_comp": 0.786, "sc_comp_naive": None},
    "skip_worst50":  {"fp": 0.846, "sc_raw": 0.748, "fp_comp": 0.836, "sc_comp_naive": 0.840},
}


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt_pct(x):
    if x is None:
        return "  ---  "
    return f"{x*100:6.2f}"


def main():
    files = sorted(list(RESULTS.glob("overnight_*.json"))
                   + list(RESULTS.glob("pure_sc_*.json")))
    if not files:
        print("No overnight_*.json results yet.", file=sys.stderr)
        return

    print(f"{'='*120}")
    print(f"OVERNIGHT EXPERIMENT MATRIX  ({len(files)} runs)")
    print(f"{'='*120}\n")

    # Header
    print(f"{'config':<30s}  {'FP':>7s}  {'SC raw':>7s}  {'+comp':>7s}  "
          f"{'Δ vs raw':>9s}  {'Δ vs FPcmp':>10s}  {'SC%blocks':>9s}  {'top variants'}")
    print("-" * 130)

    # Group by sc_config
    rows_by_cfg = {}
    for f in files:
        d = load(f)
        cfg = d["config"]["sc_config"]
        rows_by_cfg.setdefault(cfg, []).append((f.stem, d))

    for cfg, rows in sorted(rows_by_cfg.items()):
        ref = BASELINES.get(cfg, {})
        print(f"\n[{cfg}] FP={fmt_pct(ref.get('fp'))}  "
              f"SC raw={fmt_pct(ref.get('sc_raw'))}  "
              f"FP comp baseline={fmt_pct(ref.get('fp_comp'))}  "
              f"SC comp naive={fmt_pct(ref.get('sc_comp_naive'))}")
        for name, d in sorted(rows, key=lambda x: -x[1]["results"]["sc_comp"]["top1"]):
            results = d["results"]
            calib = d.get("calib", {})
            sc_comp = results["sc_comp"]["top1"]
            sc_raw = (results.get("sc_raw") or {}).get("top1")
            fp = results["fp"]["top1"]
            n_sc = calib.get("n_sc_comp_blocks", "?")
            n_fp = calib.get("n_fp_comp_blocks", "?")
            n_total = (n_sc + n_fp) if (isinstance(n_sc, int) and isinstance(n_fp, int)) else 24
            sc_frac = (n_sc / n_total) if isinstance(n_sc, int) else None
            d_raw = (sc_comp - sc_raw) if sc_raw is not None else None
            d_fp_comp = (sc_comp - ref.get("fp_comp", 0.0)) if ref.get("fp_comp") else None

            # Top variants used
            cnt = calib.get("variant_counts", {})
            if isinstance(cnt, dict):
                top3 = ", ".join(f"{k}({v})" for k, v in
                                 Counter(cnt).most_common(3))
            else:
                top3 = ""

            short = name.replace("overnight_", "")
            print(f"  {short:<28s}  {fmt_pct(fp)}  {fmt_pct(sc_raw)}  "
                  f"{fmt_pct(sc_comp)}  "
                  f"{(f'{d_raw*100:+6.2f}pt' if d_raw is not None else '   ---   '):>9s}  "
                  f"{(f'{d_fp_comp*100:+7.2f}pt' if d_fp_comp is not None else '   ---   '):>10s}  "
                  f"{(f'{sc_frac*100:6.0f}%' if sc_frac is not None else '   ---   '):>9s}  "
                  f"{top3[:40]}")

    print("\n" + "=" * 120)
    print("LEGEND: Δ vs raw = +pt over no-comp SC; Δ vs FPcmp = vs README's "
          "FP-comp baseline (negative = behind FP comp).")
    print("        SC%blocks = fraction of blocks where calibration chose an SC comp.")
    print("=" * 120)


if __name__ == "__main__":
    main()
