#!/usr/bin/env python3
"""Builds figures/*.pdf and tables/*.tex for the report from the benchmark results.

Usage: python3 make_assets.py <path-to-results-dir> <path-to-p1-output-dir>
"""
import collections
import csv
import os
import re
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1]
P1 = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Palette: categorical slots 1 and 2 of the reference palette (light mode).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False,
})


def rows(path):
    with open(os.path.join(RES, path)) as f:
        return list(csv.DictReader(f))


def write(name, text):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def fmt(x, d=3):
    return f"{x:.{d}f}"


# ------------------------------------------------------------------ Problem 1
def p1_table():
    out = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
           r"$N$ & CPU & \multicolumn{2}{c}{Basic (ms)} & \multicolumn{2}{c}{Optimized (ms)} & \multicolumn{2}{c}{Speedup} \\",
           r" & (ms) & kernel & end-to-end & kernel & end-to-end & kernel & end-to-end \\", r"\midrule"]
    for n in (64, 128, 256):
        t = open(os.path.join(P1, f"outputs_{n}_p1.txt")).read()
        g = lambda p: st.mean(float(x) for x in re.findall(p, t))
        cpu = g(r"Serial version: (\d+) ms")
        bk = g(r"Kernel3D_basic time \(ms\): ([\d.]+)")
        be = g(r"Kernel3D_basic End-To-End time \(ms\): ([\d.]+)")
        ok = g(r"Kernel3D_opt time \(ms\): ([\d.]+)")
        oe = g(r"Kernel3D_opt End-To-End Time \(ms\): ([\d.]+)")
        out.append(f"{n} & {cpu:.1f} & {bk:.3f} & {be:.2f} & {ok:.3f} & {oe:.2f} & "
                   f"{bk/ok:.1f}$\\times$ & {be/oe:.2f}$\\times$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("p1.tex", "\n".join(out))


# ------------------------------------------------------------------ Problem 2
def p2_tables_and_fig():
    d = collections.defaultdict(list)
    for r in rows("p2/summary.csv"):
        d[r["version"]].append(float(r["total_time_s"]))
    abl = collections.defaultdict(list)
    for r in rows("ablation/p2.csv"):
        if r["total_time_s"]:
            abl[(r["version"], int(r["BLOCK"]), int(r["PRUNE"]))].append(float(r["total_time_s"]))

    entries = [
        ("v0", "Serial CPU (reference)", d["v0"]),
        ("v1", "Vanilla CUDA port", d["v1"]),
        ("v2", "v2, no pruning", d["v2"]),
        ("v2", "v2 + pruning (final)", abl[("v2", 128, 1)]),
        ("v3", "Thrust", d["v3"]),
    ]
    v0, v1 = st.median(d["v0"]), st.median(d["v1"])
    out = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Version & Description & Runs & Median (s) & Min--Max (s) & vs.\ v0 & vs.\ v1 \\", r"\midrule"]
    for v, desc, xs in entries:
        m = st.median(xs)
        rng = "--" if len(xs) == 1 else f"{min(xs):.3f}--{max(xs):.3f}"
        sp = lambda r: f"{r:.2f}" if r < 1 else f"{r:.1f}"
        out.append(f"{v} & {desc} & {len(xs)} & {m:.3f} & {rng} & "
                   f"{sp(v0/m)}$\\times$ & {sp(v1/m)}$\\times$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("p2.tex", "\n".join(out))

    blocks = [64, 128, 256, 512, 1024]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    xs = [x for x in blocks if abl[("v1", x, 0)]]
    a.plot(xs, [st.median(abl[("v1", x, 0)]) for x in xs], "o-", color=BLUE)
    a.set_xscale("log", base=2)
    a.set_xticks(blocks, [str(x) for x in blocks])
    a.set_ylim(0, 40)
    a.set_xlabel("threads per block")
    a.set_ylabel("median time (s)")
    a.set_title("v1", loc="left", fontsize=9, color=INK2)

    for pr, col, lab in ((0, BLUE, "no pruning"), (1, ORANGE, "pruning")):
        xs = [x for x in blocks if abl[("v2", x, pr)]]
        ys = [st.median(abl[("v2", x, pr)]) for x in xs]
        b.plot(xs, ys, "o-", color=col, label=lab)
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(blocks, [str(x) for x in blocks])
    b.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    b.minorticks_off()
    b.set_ylim(0.08, 20)
    b.set_xlim(55, 1200)
    b.set_xlabel("threads per block")
    b.set_ylabel("median time (s)")
    b.set_title("v2", loc="left", fontsize=9, color=INK2)
    b.plot([1024], [1.0], "x", color=INK2, markersize=7, mew=2)
    b.annotate("launch failed", (1024, 1.0), xytext=(0, 9), textcoords="offset points",
               ha="center", fontsize=8, color=INK2)
    b.legend(loc="center left", fontsize=8, bbox_to_anchor=(0.0, 0.55))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_ablation.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ Problem 3
def agg(path, keys, val="kernel_ms"):
    d = collections.defaultdict(list)
    for r in rows(path):
        d[tuple(r[k] for k in keys)].append(float(r[val]))
    return d


def p3_timing_table():
    d = agg("p3/timing/summary.csv", ["variant", "phase"])
    ops = [("insert", "Insert (950\\,000 keys)"), ("search1", "Search after insert (10$^6$)"),
           ("delete", "Delete (285\\,000 keys)"), ("search2", "Search after delete (10$^6$)")]
    out = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Kernel & Vanilla (ms) & Blocked (ms) & Blocked / vanilla \\", r"\midrule"]
    for k, lab in ops:
        v, b = d[("vanilla", k)], d[("blocked", k)]
        out.append(f"{lab} & {st.mean(v):.3f} $\\pm$ {st.stdev(v):.3f} & "
                   f"{st.mean(b):.3f} $\\pm$ {st.stdev(b):.3f} & {st.mean(b)/st.mean(v):.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("p3_timing.tex", "\n".join(out))


def p3_fp_table_and_fig():
    fp = {}
    for r in rows("n_sweep/false_positives.csv"):
        fp[(r["variant"], int(r["N"]), r["phase"])] = r
    ns = [1_000_000, 5_000_000, 10_000_000, 20_000_000]
    out = [r"\begin{tabular}{rrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{After insert (\%)} & \multicolumn{2}{c}{After delete (\%)} \\",
           r"$N$ & Vanilla & Blocked & Vanilla & Blocked \\", r"\midrule"]
    for n in ns:
        c = []
        for ph in ("after insert", "after delete"):
            for v in ("vanilla", "blocked"):
                c.append(f"{100*float(fp[(v, n, ph)]['fp_rate_all']):.3f}")
        out.append(f"{n//1_000_000}M & " + " & ".join(c) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("p3_fp.tex", "\n".join(out))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for v, col, lab, dy in (("vanilla", BLUE, "vanilla", 0.55), ("blocked", ORANGE, "blocked", 1.8)):
        ys = [100 * float(fp[(v, n, "after insert")]["fp_rate_all"]) for n in ns]
        ax.plot([n / 1e6 for n in ns], ys, "o-", color=col, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 5, 10, 20], ["1M", "5M", "10M", "20M"])
    ax.set_xlabel("filter size $N$ (counters)")
    ax.set_ylabel("false-positive rate (%)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p3_nsweep.pdf"))
    plt.close(fig)


def p3_blockcounters():
    t = agg("ablation/p3_timing.csv", ["BLOCK_COUNTERS", "phase"])
    van = agg("p3/timing/summary.csv", ["variant", "phase"])
    bcs = [16, 32, 64, 128, 256]
    fpr = {}
    for bc in bcs:
        txt = open(os.path.join(RES, f"ablation/p3_fp_bc{bc}.txt")).read()
        fpr[bc] = [float(x) for x in re.findall(r"all searches\): ([\d.]+)", txt)]
    vtxt = open(os.path.join(RES, "p3/timing/vanilla.false_positives.txt")).read()
    vfp = [float(x) for x in re.findall(r"all searches\): ([\d.]+)", vtxt)]

    out = [r"\begin{tabular}{rrrrr}", r"\toprule",
           r"Block size & Bytes & FP after insert (\%) & Search (ms) & Insert (ms) \\", r"\midrule"]
    for bc in bcs:
        out.append(f"{bc} counters & {bc} & {100*fpr[bc][0]:.3f} & "
                   f"{st.mean(t[(str(bc), 'search1')]):.3f} & {st.mean(t[(str(bc), 'insert')]):.3f} \\\\")
    out.append(r"\midrule")
    out.append(f"vanilla (no blocks) & -- & {100*vfp[0]:.3f} & "
               f"{st.mean(van[('vanilla', 'search1')]):.3f} & {st.mean(van[('vanilla', 'insert')]):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("p3_blocks.tex", "\n".join(out))

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a.plot(bcs, [100 * fpr[x][0] for x in bcs], "o-", color=ORANGE, label="blocked")
    a.axhline(100 * vfp[0], color=BLUE, ls="--", lw=1.5, label="vanilla")
    b.plot(bcs, [st.mean(t[(str(x), "search1")]) for x in bcs], "o-", color=ORANGE, label="blocked")
    b.axhline(st.mean(van[("vanilla", "search1")]), color=BLUE, ls="--", lw=1.5, label="vanilla")
    for ax, yl in ((a, "false-positive rate (%)"), (b, "search kernel time (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(bcs, [str(x) for x in bcs])
        ax.set_xlabel("counters per block")
        ax.set_ylabel(yl)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p3_blocks.pdf"))
    plt.close(fig)


def p3_ncu():
    cols = [("vanilla", "bloom_insert"), ("vanilla", "bloom_search"),
            ("blocked", "bloom_insert"), ("blocked", "bloom_search")]
    raw, det = {}, {}
    for v in ("vanilla", "blocked"):
        with open(os.path.join(RES, f"p3/ncu/{v}_raw.csv")) as f:
            rr = list(csv.DictReader(f))[1:]  # first row after header holds units
        for r in rr:
            raw[(v, r["Kernel Name"].split("(")[0])] = r
        for r in csv.DictReader(open(os.path.join(RES, f"p3/ncu/{v}_details.csv"))):
            k = (v, r["Kernel Name"].split("(")[0])
            det.setdefault(k, {})
            if r["Metric Name"]:
                det[k].setdefault(r["Metric Name"], r["Metric Value"])
            if r["Rule Name"]:
                det[k].setdefault("rule:" + r["Rule Name"], r["Rule Description"])
                det[k].setdefault("speedup:" + r["Rule Name"], r["Estimated Speedup"])

    def num(s):
        return float(str(s).replace(",", ""))

    def dur_ms(k):  # raw csv has ms or us depending on magnitude; use details page
        v = det[k]["Duration"]
        return num(v)

    metrics = collections.OrderedDict()
    metrics["Duration"] = []
    metrics["DRAM throughput (\\% of peak)"] = []
    metrics["Achieved DRAM bandwidth (GB/s)"] = []
    metrics["Arithmetic intensity (int.\\ ops/byte)"] = []
    metrics["Achieved occupancy (\\%)"] = []
    metrics["Warp efficiency (\\%)"] = []
    metrics["L2 hit rate (\\%)"] = []
    metrics["Long-scoreboard stall (cycles/instr.)"] = []
    for k in cols:
        r, dd = raw[k], det[k]
        # unit-aware duration -> microseconds
        unit = None
        for rr in csv.DictReader(open(os.path.join(RES, f"p3/ncu/{k[0]}_details.csv"))):
            if rr["Kernel Name"].startswith(k[1]) and rr["Metric Name"] == "Duration":
                unit = rr["Metric Unit"]
                break
        dur_us = num(dd["Duration"]) * (1000 if unit == "ms" else 1)
        dram_mb = num(r["dram__bytes.sum"])
        # raw bytes column unit is Mbyte
        dram_bytes = dram_mb * 1e6
        gbps = dram_bytes / (dur_us * 1e-6) / 1e9
        ints = num(r["smsp__sass_thread_inst_executed_op_integer_pred_on.sum"])
        metrics["Duration"].append(f"{dur_us:.0f}\\,$\\mu$s")
        metrics["DRAM throughput (\\% of peak)"].append(f"{num(dd['DRAM Throughput']):.1f}")
        metrics["Achieved DRAM bandwidth (GB/s)"].append(f"{gbps:.0f}")
        metrics["Arithmetic intensity (int.\\ ops/byte)"].append(f"{ints/dram_bytes:.2f}")
        metrics["Achieved occupancy (\\%)"].append(f"{num(dd['Achieved Occupancy']):.1f}")
        metrics["Warp efficiency (\\%)"].append(
            f"{100*num(r['smsp__thread_inst_executed_per_inst_executed.ratio'])/32:.1f}")
        metrics["L2 hit rate (\\%)"].append(f"{num(dd['L2 Hit Rate']):.1f}")
        metrics["Long-scoreboard stall (cycles/instr.)"].append(
            f"{num(r['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']):.0f}")

    out = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r" & \multicolumn{2}{c}{Vanilla} & \multicolumn{2}{c}{Blocked} \\",
           r"Metric & insert & search & insert & search \\", r"\midrule"]
    for m, vals in metrics.items():
        out.append(f"{m} & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("p3_ncu.tex", "\n".join(out))

    # Top-3 bottlenecks, ranked by the profiler's estimated speedup (excluding the
    # generic "HighPipeUtilization" and informational rules, which only restate low
    # utilisation).
    def grab(k, rule, pat):
        m = re.search(pat, det[k]["rule:" + rule])
        return m.group(1) if m else "?"

    lines = []
    ex = {k: grab(k, "UncoalescedGlobalAccess", r"\((\d+)% of the total") for k in cols}
    ex = {k: grab(k, "UncoalescedGlobalAccess", r"\(([\d.]+)% of the total") for k in cols}
    us = {k: det[k]["speedup:UncoalescedGlobalAccess"] for k in cols}
    by = {k: grab(k, "MemoryCacheAccessPattern", r"only ([\d.]+) of the 32 bytes") for k in cols}
    miss = {k: grab(k, "MemoryCacheAccessPattern", r"applies to the ([\d.]+)% of sectors missed") for k in cols}
    cpi = {k: grab(k, "CPIStall", r"spends ([\d.,]+) cycles") for k in cols}
    sp = {k: det[k]["speedup:CPIStall"] for k in cols}
    vi, vs, bi, bs = cols
    lines.append(r"\begin{enumerate}")
    lines.append(
        r"\item \textbf{Uncoalesced global accesses.} Counter lookups are random, so "
        f"{ex[vi]}\\%/{ex[vs]}\\% (vanilla insert/search) and {ex[bi]}\\%/{ex[bs]}\\% (blocked) of the "
        f"transferred sectors are excess; estimated speedup if fixed: {us[vi]}\\%/{us[vs]}\\% "
        f"(vanilla), {us[bi]}\\%/{us[bs]}\\% (blocked).")
    lines.append(
        r"\item \textbf{Long-scoreboard stalls.} Warps wait on L1TEX/global loads: "
        f"{cpi[vi]}/{cpi[vs]} cycles per issued instruction (vanilla insert/search) and "
        f"{cpi[bi]}/{cpi[bs]} (blocked).")
    lines.append(
        r"\item \textbf{Poor DRAM/L2 access pattern.} Only "
        f"{by[vi]}--{by[vs]} of every 32 bytes in a fetched sector are used, and "
        f"{miss[vi]}\\%/{miss[vs]}\\% of sectors miss L2 in vanilla versus "
        f"{miss[bi]}\\%/{miss[bs]}\\% in blocked.")
    lines.append(r"\end{enumerate}")
    write("p3_bottlenecks.tex", "\n".join(lines))


p1_table()
p2_tables_and_fig()
p3_timing_table()
p3_fp_table_and_fig()
p3_blockcounters()
p3_ncu()
print("done")
