import glob
import json
import math
import statistics as st
import numpy as np
from scipy.stats import spearmanr

OCC = "results/runs/20260830T002223Z_6f98de3_6e25cf7d"
r = json.load(open(f"{OCC}/results.json"))
C = r["cells"]
seeds = [json.load(open(f)) for f in sorted(glob.glob(f"{OCC}/seed_*.json"))]
ok = fail = 0
def chk(label, asserted, actual, tol=0.0):
    global ok, fail
    a = actual
    good = (abs(asserted - a) <= tol) if isinstance(asserted,(int,float)) else asserted == a
    print(f"{'OK ' if good else 'MISMATCH':9} {label:52} asserted={asserted!s:<16} actual={a}")
    ok, fail = ok + good, fail + (not good)

div = {k: 1 - C[k]["pooled_tk_ex3"] for k in C}
t1  = {k: 1 - C[k]["pooled_tk_ex1"] for k in C}
print("== headline ==")
chk("top-3 divergence min (best cell)", 0.43, round(min(div.values()),2), 0.005)
chk("top-3 divergence max (worst cell)", 0.94, round(max(div.values()),2), 0.005)
chk("nsep median max over cells < 3", True, max(C[k]["pooled_nsep_median"] for k in C) < 3)
nsep_ceil = [max(1, math.ceil(C[k]["n_separate_median"]["mean"])) for k in C]
chk("separation, ceil with floor 1, min", 1, min(nsep_ceil))
chk("separation, ceil with floor 1, max", 3, max(nsep_ceil))

print("\n== accuracy matching ==")
tot = sum(C[k]["n_pairs_total"]["mean"] for k in C)
tied = sum(C[k]["n_pairs_indistinguishable"]["mean"] for k in C)
chk("total pairs across grid", 2280, round(tot))
chk("tied pairs across grid", 1610, round(tied))
b = C["128x1"]
chk("128x1 held-out R2", 0.637, round(b["heldout_r2_mean"]["mean"],3), 0.0005)
chk("128x1 tied/total", "154/190", f"{b['n_pairs_indistinguishable']['mean']:.0f}/{b['n_pairs_total']['mean']:.0f}")
chk("128x1 MDE R2", 0.032, round(b["pooled_mde_r2"],3), 0.0005)
chk("128x1 frac rho<0.9", 0.37, round(b["pooled_frac_below_9"],2), 0.005)
chk("128x1 frac rho<0.8", 0.15, round(b["pooled_frac_below_8"],2), 0.005)
chk("128x1 frac rho<0.7", 0.06, round(b["pooled_frac_below_7"],2), 0.005)
chk("128x1 min rho", 0.350, round(b["pooled_rho_min"],3), 0.0005)
r2s = [C[k]["heldout_r2_mean"]["mean"] for k in C]
rhomin = [C[k]["pooled_rho_min"] for k in C]
chk("Spearman(R2, min rho) across cells", 0.650, round(spearmanr(r2s, rhomin).statistic,3), 0.0015)

print("\n== top-k set ==")
chk("128x2 top-3 differs", 0.73, round(div["128x2"],2), 0.005)
chk("128x3 top-3 differs (pooled)", 0.905, round(div["128x3"],3), 0.0005)
chk("128x2 conditioned", 0.71, round(1-C["128x2"]["pooled_tk_cex3"],2), 0.005)
chk("128x3 conditioned", 0.80, round(1-C["128x3"]["pooled_tk_cex3"],2), 0.005)
chk("128x3 cond frac instances", 0.11, round(C["128x3"]["pooled_tk_cfrac"],2), 0.005)
chk("64x3 cond frac instances", 0.003, round(C["64x3"]["pooled_tk_cfrac"],3), 0.0005)
chk("128x1 top-1 differs", 0.18, round(t1["128x1"],2), 0.005)
chk("128x3 top-1 differs", 0.68, round(t1["128x3"],2), 0.005)

print("\n== concentration ==")
conc = {k: C[k]["pooled_conc_top3"] for k in C}
eff  = {k: C[k]["pooled_eff_pos"] for k in C}
chk("min top-3 mass (64x3)", 0.54, round(min(conc.values()),2), 0.005)
chk("argmin top-3 mass cell", "64x3", min(conc, key=conc.get))
chk("max top-3 mass (128x1)", 0.98, round(max(conc.values()),2), 0.005)
chk("argmax top-3 mass cell", "128x1", max(conc, key=conc.get))
chk("grid median top-3 mass", 0.86, round(st.median(conc.values()),2), 0.005)
chk("max effective positions", 7.80, round(max(eff.values()),2), 0.005)
chk("min effective positions", 2.20, round(min(eff.values()),2), 0.005)
chk("grid median effective positions", 3.92, round(st.median(eff.values()),2), 0.005)

print("\n== boundary conditions ==")
for d,(lo,hi) in {1:(0.43,0.74), 2:(0.73,0.88), 3:(0.91,0.94)}.items():
    v=[div[k] for k in C if C[k]["depth"]==d]
    chk(f"depth {d} divergence range", f"{lo}-{hi}", f"{min(v):.2f}-{max(v):.2f}")
depths = [C[k]["depth"] for k in C]
widths = [C[k]["hidden"] for k in C]
dv = [div[k] for k in C]
chk("rank corr divergence~depth", 0.92, round(spearmanr(depths,dv).statistic,2), 0.005)
chk("rank corr divergence~width", -0.28, round(spearmanr(widths,dv).statistic,2), 0.005)
nsep=[C[k]["pooled_nsep_median"] for k in C]
chk("rank corr nsep~divergence", -0.94, round(spearmanr(nsep,dv).statistic,2), 0.005)
ex3={k:C[k]["pooled_tk_ex3"] for k in C}
for d,(lo,hi) in {1:(0.26,0.57), 2:(0.12,0.27), 3:(0.06,0.10)}.items():
    v=[ex3[k] for k in C if C[k]["depth"]==d]
    chk(f"depth {d} exact top-3 agreement", f"{lo:.2f}-{hi:.2f}", f"{min(v):.2f}-{max(v):.2f}")

print("\n== tightened match (this session) ==")
def rows(cell, thr):
    out=[]
    for s in seeds:
        c=next(x for x in s["cells"] if f"{x['hidden']}x{x['depth']}"==cell)
        ps=[p for p in c["pairs"] if thr is None or p["r2_gap"]<thr]
        if ps:
            out.append((len(ps), st.median(1-p["tk_exact_top3_frac"] for p in ps),
                           st.median(1-p["tk_exact_top2_frac"] for p in ps)))
    n=np.mean([o[0] for o in out])
    return n, np.mean([o[1] for o in out]), np.mean([o[2] for o in out])
nl, d3l, d2l = rows("128x1", None)
nt, d3t, _ = rows("128x1", 0.001)
chk("128x1 tightening factor", 18, round(nl/nt))
chk("discarded fraction ~ 17/18", 0.944, round(1-nt/nl,3), 0.006)
chk("128x1 top-3 differs at |dR2|<0.001", 0.43, round(d3t,2), 0.005)
chk("128x1 top-2 differs (tied)", 0.29, round(d2l,2), 0.005)

print(f"\n==== {ok} verified, {fail} mismatched ====")
