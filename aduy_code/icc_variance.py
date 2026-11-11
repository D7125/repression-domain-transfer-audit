"""
Variance decomposition for promoter robustness, with a profile-likelihood CI.

Why this exists. The headline number of the divergence analysis is an ICC on
the latent scale: the share of variance in robustness that the MMseqs2 family
label accounts for. Two things were missing and both change how the number may
be quoted.

First, no interval. sigma^2 sits at the boundary of its parameter space under
the null, so a Wald interval is the wrong tool and a bootstrap is awkward at
232 families. This fits the binomial GLMM directly -- the marginal likelihood
integrated over the random effect by Gauss-Hermite quadrature -- and inverts
the likelihood ratio to get a profile interval on sigma^2, which maps monotonely
onto ICC.

Second, one random level only. With 2.26 domains per protein, variance called
"within-family" is partly within-protein, so the single-level ICC attributes to
the family label variance that belongs to protein identity. The nested model
adds proteins inside families and reports both components.

  single   logit p = b0 + u_f,              u_f ~ N(0, s2_f)
  nested   logit p = b0 + u_f + u_p(f),     u_p ~ N(0, s2_p)

ICC on the latent scale uses the Bernoulli sampling variance pi^2/3 as the
residual term. That term is a fixed constant, not an estimate, so 1 - ICC is
not an estimate of within-family biological divergence; only the share the
family label accounts for is a quotable quantity.

Usage (no python -I needed; this script imports nothing local):
  python icc_variance.py --domains ../run/best_domains.csv \
      --min-members 3 --out ../run/icc_variance.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import chi2

PI2_3 = np.pi ** 2 / 3.0


def logistic_loglik(eta, y):
    """log Bernoulli likelihood, summed, stable for large |eta|."""
    return np.sum(y * eta - np.logaddexp(0.0, eta))


def gh_nodes(n):
    """Gauss-Hermite nodes/weights for the N(0,1) measure."""
    x, w = hermegauss(n)
    return x, w / np.sqrt(2.0 * np.pi)


def ll_single(params, groups, n_nodes=60):
    """Marginal log-likelihood, one random intercept per family."""
    b0, log_s = params
    s = np.exp(log_s)
    x, w = gh_nodes(n_nodes)
    total = 0.0
    for y in groups:
        # integrate over u_f: sum_k w_k prod_i p(y_i | b0 + s*x_k)
        eta = b0 + s * x[:, None]                      # [K, n_i]
        lp = np.sum(y * eta - np.logaddexp(0.0, eta), axis=1)
        total += np.logaddexp.reduce(lp + np.log(w))
    return total


def ll_nested(params, fam_groups, n_nodes=50):
    """Marginal log-likelihood, proteins nested inside families."""
    b0, log_sf, log_sp = params
    sf, sp = np.exp(log_sf), np.exp(log_sp)
    xf, wf = gh_nodes(n_nodes)
    xp, wp = gh_nodes(n_nodes)
    logwf, logwp = np.log(wf), np.log(wp)
    total = 0.0
    for prots in fam_groups:
        # for each family node, each protein contributes an inner integral
        inner = np.zeros(len(xf))
        for y in prots:
            eta = b0 + sf * xf[:, None, None] + sp * xp[None, :, None]
            lp = np.sum(y * eta - np.logaddexp(0.0, eta), axis=2)   # [Kf, Kp]
            inner += np.logaddexp.reduce(lp + logwp, axis=1)
        total += np.logaddexp.reduce(inner + logwf)
    return total


def fit_single(groups, n_nodes=60):
    from scipy.optimize import minimize
    best = None
    for s0 in (0.3, 1.0, 1.7):
        r = minimize(lambda p: -ll_single(p, groups, n_nodes),
                     x0=[0.0, np.log(s0)], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def fit_nested(fam_groups, n_nodes=50):
    from scipy.optimize import minimize
    best = None
    for sf0, sp0 in ((1.7, 0.6), (1.4, 0.2), (0.9, 1.2)):
        r = minimize(lambda p: -ll_nested(p, fam_groups, n_nodes),
                     x0=[0.0, np.log(sf0), np.log(sp0)], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 12000,
                              "maxfev": 12000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def profile_ll_nested_sp(s2p, fam_groups, n_nodes=50):
    """Max log-likelihood with the PROTEIN component held fixed."""
    from scipy.optimize import minimize
    log_sp = 0.5 * np.log(max(s2p, 1e-12))
    best = np.inf
    for sf0 in (1.7, 1.2):
        r = minimize(lambda p: -ll_nested([p[0], p[1], log_sp],
                                          fam_groups, n_nodes),
                     x0=[0.0, np.log(sf0)], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 6000})
        best = min(best, r.fun)
    return -best


def upper_bound_sp(fam_groups, ll_nested_hat, level=0.95, n_nodes=50):
    """95% upper bound for the protein variance component.

    sigma^2_protein sits at or near zero, so the quotable statement is not a
    point estimate but a ceiling: how much of the within-family variance could
    protein identity account for and still be compatible with the data.
    """
    crit = chi2.ppf(2 * level - 1, df=1)   # one-sided bound from the LR
    f = lambda s2: 2.0 * (ll_nested_hat
                          - profile_ll_nested_sp(s2, fam_groups, n_nodes)) - crit
    hi = 0.25
    for _ in range(14):
        if f(hi) > 0:
            break
        hi *= 2.0
    if f(hi) <= 0:
        return float("nan")
    return float(brentq(f, 1e-9, hi, xtol=1e-5))


def profile_ll_single(s2, groups, n_nodes=60):
    """Max log-likelihood with sigma^2 held fixed (b0 profiled out)."""
    log_s = 0.5 * np.log(s2)
    r = minimize_scalar(lambda b0: -ll_single([b0, log_s], groups, n_nodes),
                        bounds=(-8.0, 4.0), method="bounded",
                        options={"xatol": 1e-7})
    return -r.fun


def profile_ci_s2(s2_hat, ll_hat, groups, level=0.95, n_nodes=60):
    """Invert the likelihood ratio: {s2 : 2(ll_hat - profile(s2)) <= chi2_1}."""
    crit = chi2.ppf(level, df=1)
    f = lambda s2: 2.0 * (ll_hat - profile_ll_single(s2, groups, n_nodes)) - crit

    # lower bound: sigma^2 can approach 0, which is the null itself
    lo_hi = s2_hat
    lo_lo = max(1e-6, s2_hat * 1e-4)
    lo = brentq(f, lo_lo, lo_hi, xtol=1e-6) if f(lo_lo) > 0 else lo_lo
    # upper bound: expand until the ratio clears the critical value
    hi = s2_hat * 2.0
    for _ in range(40):
        if f(hi) > 0:
            break
        hi *= 1.6
    up = brentq(f, s2_hat, hi, xtol=1e-6)
    return float(lo), float(up)


def icc(s2, extra=0.0):
    return s2 / (s2 + extra + PI2_3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", required=True)
    ap.add_argument("--outcome", default="pgk_hit")
    ap.add_argument("--family-col", default="mmseqs_rep")
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--nested-nodes", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.domains)
    d = d[d[args.family_col].notna() & d[args.outcome].notna()].copy()
    d["y"] = d[args.outcome].astype(int)
    size = d.groupby(args.family_col)["y"].transform("size")
    d = d[size >= args.min_members]
    print(f"{len(d):,} domain | {d[args.family_col].nunique():,} ho "
          f"(>= {args.min_members} thanh vien) | "
          f"{d.protein.nunique():,} protein | ty le ben {d.y.mean():.4f}")
    print(f"domain moi protein: {len(d)/d.protein.nunique():.2f}", flush=True)

    groups = [g.y.values for _, g in d.groupby(args.family_col)]
    fam_groups = [[gp.y.values for _, gp in g.groupby("protein")]
                  for _, g in d.groupby(args.family_col)]

    print("\n== mo hinh mot tang: (1 | ho) ==", flush=True)
    r1 = fit_single(groups, args.nodes)
    b0, s2 = r1.x[0], float(np.exp(2 * r1.x[1]))
    ll1 = -r1.fun
    print(f"  b0 {b0:+.4f}  sigma^2 {s2:.4f}  logLik {ll1:.4f}")
    lo, up = profile_ci_s2(s2, ll1, groups, 0.95, args.nodes)
    print(f"  sigma^2 CI95 profile-likelihood [{lo:.4f}, {up:.4f}]")
    print(f"  ICC {icc(s2):.4f}  CI95 [{icc(lo):.4f}, {icc(up):.4f}]", flush=True)

    # likelihood-ratio test against sigma^2 = 0 (boundary: mixture null)
    ll0 = profile_ll_single(1e-8, groups, args.nodes)
    lrt = 2 * (ll1 - ll0)
    p_bnd = 0.5 * chi2.sf(lrt, df=1)
    print(f"  LRT vs sigma^2=0: {lrt:.3f}  p {p_bnd:.3e} (nua-nua tai bien)",
          flush=True)

    print("\n== mo hinh hai tang long: (1 | ho / protein) ==", flush=True)
    # same quadrature as the single-level fit, or the two log-likelihoods are
    # not comparable and the nested model can appear to fit WORSE than the
    # simpler model it contains
    nn = args.nested_nodes or args.nodes
    ll1_same = ll_single([b0, 0.5 * np.log(s2)], groups, nn)
    r2 = fit_nested(fam_groups, nn)
    b0n = r2.x[0]
    s2f, s2p = float(np.exp(2 * r2.x[1])), float(np.exp(2 * r2.x[2]))
    ll2 = -r2.fun
    tot = s2f + s2p + PI2_3
    print(f"  b0 {b0n:+.4f}  sigma^2_ho {s2f:.4f}  sigma^2_protein {s2p:.4f}"
          f"  logLik {ll2:.4f}")
    print(f"  kiem tinh nhat quan: logLik mot tang cung {nn} nut = {ll1_same:.4f}"
          f"  (hieu {ll2 - ll1_same:+.4f}, phai >= 0)", flush=True)
    print(f"  phan ho      {s2f/tot:.4f}")
    print(f"  phan protein {s2p/tot:.4f}")
    sp_up = upper_bound_sp(fam_groups, ll2, 0.95, nn)
    tot_up = s2f + sp_up + PI2_3
    print(f"  gioi han tren 95% cho sigma^2_protein: {sp_up:.4f}"
          f"  => phan protein toi da {sp_up/tot_up:.4f}", flush=True)

    res = {
        "n_domains": int(len(d)), "n_families": int(d[args.family_col].nunique()),
        "n_proteins": int(d.protein.nunique()),
        "domains_per_protein": float(len(d) / d.protein.nunique()),
        "base_rate": float(d.y.mean()), "outcome": args.outcome,
        "single": {"b0": float(b0), "sigma2": s2, "loglik": float(ll1),
                   "sigma2_ci95": [lo, up], "icc": float(icc(s2)),
                   "icc_ci95": [float(icc(lo)), float(icc(up))],
                   "lrt_vs_zero": float(lrt), "p_boundary": float(p_bnd)},
        "nested": {"b0": float(b0n), "sigma2_family": s2f,
                   "sigma2_protein": s2p, "loglik": float(ll2),
                   "loglik_single_same_nodes": float(ll1_same),
                   "nodes": int(nn),
                   "share_family": float(s2f / tot),
                   "share_protein": float(s2p / tot),
                   "sigma2_protein_upper95": sp_up,
                   "share_protein_upper95": float(sp_up / tot_up)},
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
