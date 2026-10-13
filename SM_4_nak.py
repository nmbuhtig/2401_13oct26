#!/usr/bin/env python3
   

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


                                                                       
                 
                                                                       

def falling_factorial_3(n: int) -> int:
    return n * (n - 1) * (n - 2)


def totient_sieve(n: int) -> np.ndarray:
                                               
    phi = np.arange(n + 1, dtype=np.int64)
    for p in range(2, n + 1):
        if phi[p] == p:
            phi[p::p] -= phi[p::p] // p
    return phi


def T_exact(M: int) -> int:
                                                      
    if M < 3:
        return 0
    phi = totient_sieve(M)
    total = 0
    for d in range(2, M):
        J = (M - 1) // d
        S = M + 2 * (J * M - d * J * (J + 1) // 2)
        total += int(phi[d]) * (S * S - M * M)
    return 3 * total


def exact_mean_strict(N: int, M: int) -> float:
                                                         
    return falling_factorial_3(N) * T_exact(M) / M ** 6


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x[0]), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def stderr_from_samples(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def safe_log(x: float) -> float:
    if x <= 0:
        raise ValueError(f"log undefined for non-positive value {x}")
    return math.log(x)


def canonical_line_key(p: Tuple[int, int], q: Tuple[int, int]) -> Tuple[int, int, int]:
           
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        raise ValueError("Identical points do not determine a unique line")

    a = dy
    b = -dx
    g = math.gcd(abs(a), abs(b))
    a //= g
    b //= g

    if a < 0 or (a == 0 and b < 0):
        a, b = -a, -b

    c = a * x1 + b * y1
    return (a, b, c)


def linear_regression_with_se(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
           
    n = len(x)
    if n < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_se": float("nan"),
            "intercept_se": float("nan"),
            "r2": float("nan"),
            "n_points": n,
        }

    X = np.column_stack([np.ones(n), x])
    beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta

    yhat = X @ beta
    resid = y - yhat

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n - 2
    if dof > 0:
        sigma2 = ss_res / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        intercept_se = float(np.sqrt(cov[0, 0]))
        slope_se = float(np.sqrt(cov[1, 1]))
    else:
        intercept_se = float("nan")
        slope_se = float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": slope_se,
        "intercept_se": intercept_se,
        "r2": r2,
        "n_points": n,
    }


                                                                       
                             
                                                                       

@dataclass
class TrialCounts:
    strict_ordered_collinear: int
    all_same_ordered_collinear: int
    exactly_two_same_ordered_collinear: int
    inclusive_ordered_collinear: int
    num_occupied_cells: int
    max_cell_occupancy: int


@dataclass
class AggregateResult:
    N: int
    M: int
    Delta: float
    num_trials: int

    mean_W_strict: float
    std_W_strict: float
    se_W_strict: float

    mean_W_all_same: float
    std_W_all_same: float
    se_W_all_same: float

    mean_W_exactly_two_same: float
    std_W_exactly_two_same: float
    se_W_exactly_two_same: float

    mean_W_inclusive: float
    std_W_inclusive: float
    se_W_inclusive: float

    mean_num_occupied_cells: float
    mean_max_cell_occupancy: float

    theory_scale_N3: float
    theory_scale_falling: float
    delta2_log_inv_delta: float

    norm_mean_strict_N3: float
    norm_se_strict_N3: float
    norm_mean_strict_falling: float
    norm_se_strict_falling: float

    norm_mean_inclusive_N3: float
    norm_se_inclusive_N3: float
    norm_mean_inclusive_falling: float
    norm_se_inclusive_falling: float

    frac_strict_of_inclusive: float
    frac_all_same_of_inclusive: float
    frac_exactly_two_same_of_inclusive: float


                                                                       
                 
                                                                       

def count_ordered_collinear_triples_from_occupancies(
    occupied_ids: np.ndarray,
    occupancies: np.ndarray,
    M: int,
) -> TrialCounts:
           
    K = occupied_ids.size
    if K == 0:
        return TrialCounts(0, 0, 0, 0, 0, 0)

    xs = occupied_ids % M
    ys = occupied_ids // M
    points = list(zip(xs.tolist(), ys.tolist()))
    counts = occupancies.astype(np.int64)

                                                       
    line_to_indices: Dict[Tuple[int, int, int], set] = {}

    for i in range(K):
        pi = points[i]
        for j in range(i + 1, K):
            pj = points[j]
            key = canonical_line_key(pi, pj)
            if key not in line_to_indices:
                line_to_indices[key] = {i, j}
            else:
                line_to_indices[key].add(i)
                line_to_indices[key].add(j)

    strict_total = 0
    for idx_set in line_to_indices.values():
        if len(idx_set) < 3:
            continue
        idx = np.fromiter(idx_set, dtype=np.int64)
        n = counts[idx]

        s1 = int(np.sum(n))
        s2 = int(np.sum(n * n))
        s3 = int(np.sum(n * n * n))

                                                                   
        strict_line = s1**3 - 3 * s1 * s2 + 2 * s3
        strict_total += strict_line

    N_total = int(np.sum(counts))

    all_same = int(np.sum(counts * (counts - 1) * (counts - 2)))
    exactly_two_same = int(np.sum(3 * counts * (counts - 1) * (N_total - counts)))
    inclusive_total = strict_total + all_same + exactly_two_same

    return TrialCounts(
        strict_ordered_collinear=int(strict_total),
        all_same_ordered_collinear=int(all_same),
        exactly_two_same_ordered_collinear=int(exactly_two_same),
        inclusive_ordered_collinear=int(inclusive_total),
        num_occupied_cells=int(K),
        max_cell_occupancy=int(np.max(counts)),
    )


def simulate_one_trial(N: int, M: int, rng: np.random.Generator) -> TrialCounts:
           
    cell_ids = rng.integers(0, M * M, size=N, endpoint=False)
    occupied_ids, occupancies = np.unique(cell_ids, return_counts=True)
    return count_ordered_collinear_triples_from_occupancies(occupied_ids, occupancies, M)


                                                                       
             
                                                                       

def aggregate_trials(N: int, M: int, trial_counts: List[TrialCounts]) -> AggregateResult:
    Delta = 1.0 / M
    delta2_log = (Delta ** 2) * math.log(1.0 / Delta)
    theory_scale_N3 = (N ** 3) * delta2_log
    theory_scale_falling = falling_factorial_3(N) * delta2_log

    strict_vals = np.array([t.strict_ordered_collinear for t in trial_counts], dtype=float)
    all_same_vals = np.array([t.all_same_ordered_collinear for t in trial_counts], dtype=float)
    two_same_vals = np.array([t.exactly_two_same_ordered_collinear for t in trial_counts], dtype=float)
    inclusive_vals = np.array([t.inclusive_ordered_collinear for t in trial_counts], dtype=float)
    occupied_vals = np.array([t.num_occupied_cells for t in trial_counts], dtype=float)
    max_occ_vals = np.array([t.max_cell_occupancy for t in trial_counts], dtype=float)

    mean_strict, std_strict = mean_std(strict_vals)
    mean_all_same, std_all_same = mean_std(all_same_vals)
    mean_two_same, std_two_same = mean_std(two_same_vals)
    mean_inclusive, std_inclusive = mean_std(inclusive_vals)

    se_strict = stderr_from_samples(strict_vals)
    se_all_same = stderr_from_samples(all_same_vals)
    se_two_same = stderr_from_samples(two_same_vals)
    se_inclusive = stderr_from_samples(inclusive_vals)

    if mean_inclusive > 0:
        frac_strict = mean_strict / mean_inclusive
        frac_all_same = mean_all_same / mean_inclusive
        frac_two_same = mean_two_same / mean_inclusive
    else:
        frac_strict = float("nan")
        frac_all_same = float("nan")
        frac_two_same = float("nan")

    return AggregateResult(
        N=N,
        M=M,
        Delta=Delta,
        num_trials=len(trial_counts),

        mean_W_strict=mean_strict,
        std_W_strict=std_strict,
        se_W_strict=se_strict,

        mean_W_all_same=mean_all_same,
        std_W_all_same=std_all_same,
        se_W_all_same=se_all_same,

        mean_W_exactly_two_same=mean_two_same,
        std_W_exactly_two_same=std_two_same,
        se_W_exactly_two_same=se_two_same,

        mean_W_inclusive=mean_inclusive,
        std_W_inclusive=std_inclusive,
        se_W_inclusive=se_inclusive,

        mean_num_occupied_cells=float(np.mean(occupied_vals)),
        mean_max_cell_occupancy=float(np.mean(max_occ_vals)),

        theory_scale_N3=theory_scale_N3,
        theory_scale_falling=theory_scale_falling,
        delta2_log_inv_delta=delta2_log,

        norm_mean_strict_N3=mean_strict / theory_scale_N3,
        norm_se_strict_N3=se_strict / theory_scale_N3,
        norm_mean_strict_falling=mean_strict / theory_scale_falling,
        norm_se_strict_falling=se_strict / theory_scale_falling,

        norm_mean_inclusive_N3=mean_inclusive / theory_scale_N3,
        norm_se_inclusive_N3=se_inclusive / theory_scale_N3,
        norm_mean_inclusive_falling=mean_inclusive / theory_scale_falling,
        norm_se_inclusive_falling=se_inclusive / theory_scale_falling,

        frac_strict_of_inclusive=frac_strict,
        frac_all_same_of_inclusive=frac_all_same,
        frac_exactly_two_same_of_inclusive=frac_two_same,
    )


                                                                       
                                     
                                                                       

def compute_summary_statistics(results: List[AggregateResult]) -> Dict[str, object]:
           
    if not results:
        return {}

    results_sorted = sorted(results, key=lambda r: (r.N, r.M))
    unique_N = sorted({r.N for r in results_sorted})
    unique_M = sorted({r.M for r in results_sorted})

    summary: Dict[str, object] = {
        "N_slope_fits": [],
        "Delta_factor_fits": [],
        "normalized_ratio_ranges": [],
        "component_fraction_summary": [],
        "global_summary": {},
    }

                                
    for M in unique_M:
        rows = [r for r in results_sorted if r.M == M]
        x = np.array([safe_log(r.N) for r in rows], dtype=float)

        for label, extractor in [
            ("strict", lambda r: r.mean_W_strict),
            ("inclusive", lambda r: r.mean_W_inclusive),
        ]:
            yvals = np.array([extractor(r) for r in rows], dtype=float)
            positive = yvals > 0
            fit = linear_regression_with_se(x[positive], np.log(yvals[positive]))

            summary["N_slope_fits"].append({
                "M": M,
                "observable": label,
                **fit,
            })

                                      
                                                      
    for N in unique_N:
        rows = [r for r in results_sorted if r.N == N]
        x = np.array([safe_log(r.delta2_log_inv_delta) for r in rows], dtype=float)

        for label, extractor in [
            ("strict", lambda r: r.mean_W_strict / (r.N ** 3)),
            ("inclusive", lambda r: r.mean_W_inclusive / (r.N ** 3)),
        ]:
            yvals = np.array([extractor(r) for r in rows], dtype=float)
            positive = yvals > 0
            fit = linear_regression_with_se(x[positive], np.log(yvals[positive]))

            summary["Delta_factor_fits"].append({
                "N": N,
                "observable": label,
                **fit,
            })

                                                    
    for N in unique_N:
        rows = [r for r in results_sorted if r.N == N]
        strict_norm = np.array([r.norm_mean_strict_N3 for r in rows], dtype=float)
        incl_norm = np.array([r.norm_mean_inclusive_N3 for r in rows], dtype=float)

        summary["normalized_ratio_ranges"].append({
            "N": N,
            "strict_min": float(np.min(strict_norm)),
            "strict_max": float(np.max(strict_norm)),
            "strict_rel_span": float((np.max(strict_norm) - np.min(strict_norm)) / np.mean(strict_norm)),
            "inclusive_min": float(np.min(incl_norm)),
            "inclusive_max": float(np.max(incl_norm)),
            "inclusive_rel_span": float((np.max(incl_norm) - np.min(incl_norm)) / np.mean(incl_norm)),
        })

                                                     
    for N in unique_N:
        rows = [r for r in results_sorted if r.N == N]
        summary["component_fraction_summary"].append({
            "N": N,
            "mean_frac_strict_of_inclusive": float(np.mean([r.frac_strict_of_inclusive for r in rows])),
            "mean_frac_all_same_of_inclusive": float(np.mean([r.frac_all_same_of_inclusive for r in rows])),
            "mean_frac_exactly_two_same_of_inclusive": float(np.mean([r.frac_exactly_two_same_of_inclusive for r in rows])),
        })

                      
    strict_slopes = [d["slope"] for d in summary["N_slope_fits"] if d["observable"] == "strict" and math.isfinite(d["slope"])]
    incl_slopes = [d["slope"] for d in summary["N_slope_fits"] if d["observable"] == "inclusive" and math.isfinite(d["slope"])]
    strict_delta = [d["slope"] for d in summary["Delta_factor_fits"] if d["observable"] == "strict" and math.isfinite(d["slope"])]
    incl_delta = [d["slope"] for d in summary["Delta_factor_fits"] if d["observable"] == "inclusive" and math.isfinite(d["slope"])]

    summary["global_summary"] = {
        "mean_N_slope_strict": float(np.mean(strict_slopes)) if strict_slopes else float("nan"),
        "mean_N_slope_inclusive": float(np.mean(incl_slopes)) if incl_slopes else float("nan"),
        "mean_Delta_factor_slope_strict": float(np.mean(strict_delta)) if strict_delta else float("nan"),
        "mean_Delta_factor_slope_inclusive": float(np.mean(incl_delta)) if incl_delta else float("nan"),
    }

    return summary


def write_summary_csv(summary: Dict[str, object], csv_path: str) -> None:
           
    rows: List[Dict[str, object]] = []

    for item in summary.get("N_slope_fits", []):
        row = {"section": "N_slope_fits"}
        row.update(item)
        rows.append(row)

    for item in summary.get("Delta_factor_fits", []):
        row = {"section": "Delta_factor_fits"}
        row.update(item)
        rows.append(row)

    for item in summary.get("normalized_ratio_ranges", []):
        row = {"section": "normalized_ratio_ranges"}
        row.update(item)
        rows.append(row)

    for item in summary.get("component_fraction_summary", []):
        row = {"section": "component_fraction_summary"}
        row.update(item)
        rows.append(row)

    if summary.get("global_summary"):
        row = {"section": "global_summary"}
        row.update(summary["global_summary"])
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["section"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


                                                                       
          
                                                                       

def apply_aps_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (3.4, 2.8),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def write_plot_json(payload: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_normalized_figure(
    results: List[AggregateResult],
    output_png: str,
    output_json: str,
    quantity: str = "strict",
    normalization: str = "N3",
) -> None:
           
    apply_aps_style()
    results_sorted = sorted(results, key=lambda r: (r.N, r.M))
    unique_N = sorted({r.N for r in results_sorted})

    fig, ax = plt.subplots()

    if quantity == "strict" and normalization == "N3":
        ylab = r"$\widehat{\mathbb{E}}[W_{N,\mathrm{str}}^{(\Delta)}] / \left(N^3 \Delta^2 \log(1/\Delta)\right)$"
        yfun = lambda r: r.norm_mean_strict_N3
        efun = lambda r: r.norm_se_strict_N3
    elif quantity == "inclusive" and normalization == "N3":
        ylab = r"$\widehat{\mathbb{E}}[W_N^{(\Delta)}] / \left(N^3 \Delta^2 \log(1/\Delta)\right)$"
        yfun = lambda r: r.norm_mean_inclusive_N3
        efun = lambda r: r.norm_se_inclusive_N3
    elif quantity == "strict" and normalization == "falling":
        ylab = r"$\widehat{\mathbb{E}}[W_{N,\mathrm{str}}^{(\Delta)}] / \left((N)_3 \Delta^2 \log(1/\Delta)\right)$"
        yfun = lambda r: r.norm_mean_strict_falling
        efun = lambda r: r.norm_se_strict_falling
    elif quantity == "inclusive" and normalization == "falling":
        ylab = r"$\widehat{\mathbb{E}}[W_N^{(\Delta)}] / \left((N)_3 \Delta^2 \log(1/\Delta)\right)$"
        yfun = lambda r: r.norm_mean_inclusive_falling
        efun = lambda r: r.norm_se_inclusive_falling
    else:
        raise ValueError("Unsupported quantity/normalization combination")

    payload = {
        "plot_type": "normalized_mean_vs_M",
        "quantity": quantity,
        "normalization": normalization,
        "xlabel": r"Grid resolution $M=1/\Delta$",
        "ylabel": ylab,
        "series": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    for N in unique_N:
        rows = [r for r in results_sorted if r.N == N]
        x = np.array([r.M for r in rows], dtype=float)
        y = np.array([yfun(r) for r in rows], dtype=float)
        yerr = np.array([efun(r) for r in rows], dtype=float)

        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=2.0, label=fr"$N={N}$")
        payload["series"].append({
            "N": N,
            "M": x.tolist(),
            "y": y.tolist(),
            "yerr": yerr.tolist(),
        })

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Grid resolution $M=1/\Delta$")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, handlelength=2.0)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    write_plot_json(payload, output_json)


def make_dual_observable_figure(
    results: List[AggregateResult],
    output_png: str,
    output_json: str,
    normalization: str = "N3",
) -> None:
           
    apply_aps_style()
    plt.rcParams["figure.figsize"] = (7.0, 2.8)

    results_sorted = sorted(results, key=lambda r: (r.N, r.M))
    unique_N = sorted({r.N for r in results_sorted})

    fig, axes = plt.subplots(1, 2, sharex=True)

    payload = {
        "plot_type": "dual_observable_normalized_mean_vs_M",
        "normalization": normalization,
        "panels": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    configs = []
    if normalization == "N3":
        configs = [
            ("strict", r"$\widehat{\mathbb{E}}[W_{N,\mathrm{str}}^{(\Delta)}] / \left(N^3 \Delta^2 \log(1/\Delta)\right)$",
             lambda r: r.norm_mean_strict_N3, lambda r: r.norm_se_strict_N3),
            ("inclusive", r"$\widehat{\mathbb{E}}[W_N^{(\Delta)}] / \left(N^3 \Delta^2 \log(1/\Delta)\right)$",
             lambda r: r.norm_mean_inclusive_N3, lambda r: r.norm_se_inclusive_N3),
        ]
    else:
        configs = [
            ("strict", r"$\widehat{\mathbb{E}}[W_{N,\mathrm{str}}^{(\Delta)}] / \left((N)_3 \Delta^2 \log(1/\Delta)\right)$",
             lambda r: r.norm_mean_strict_falling, lambda r: r.norm_se_strict_falling),
            ("inclusive", r"$\widehat{\mathbb{E}}[W_N^{(\Delta)}] / \left((N)_3 \Delta^2 \log(1/\Delta)\right)$",
             lambda r: r.norm_mean_inclusive_falling, lambda r: r.norm_se_inclusive_falling),
        ]

    for ax, (name, ylab, yfun, efun) in zip(axes, configs):
        panel_data = {
            "panel": name,
            "ylabel": ylab,
            "series": [],
        }
        for N in unique_N:
            rows = [r for r in results_sorted if r.N == N]
            x = np.array([r.M for r in rows], dtype=float)
            y = np.array([yfun(r) for r in rows], dtype=float)
            yerr = np.array([efun(r) for r in rows], dtype=float)

            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=2.0, label=fr"$N={N}$")
            panel_data["series"].append({
                "N": N,
                "M": x.tolist(),
                "y": y.tolist(),
                "yerr": yerr.tolist(),
            })

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"$M=1/\Delta$")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, handlelength=2.0)
        payload["panels"].append(panel_data)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    write_plot_json(payload, output_json)


def make_manuscript_main_figure(
    results: List[AggregateResult],
    output_png: str,
    output_json: str,
    selected_N: Optional[List[int]] = None,
) -> None:
           
    apply_aps_style()
    plt.rcParams["figure.figsize"] = (3.4, 2.6)

    results_sorted = sorted(results, key=lambda r: (r.N, r.M))
    unique_N = sorted({r.N for r in results_sorted})

    if selected_N is None:
        selected_N = unique_N[-4:] if len(unique_N) >= 4 else unique_N
    else:
        selected_N = [n for n in selected_N if n in unique_N]

    fig, ax = plt.subplots()

    ylab = r"$\widehat{\mathbb{E}}[W_{N,\mathrm{str}}^{(\Delta)}] / \left(N^3 \Delta^2 \log(1/\Delta)\right)$"

    payload = {
        "plot_type": "manuscript_main_figure",
        "quantity": "strict",
        "normalization": "N3",
        "selected_N": selected_N,
        "xlabel": r"Grid resolution $M=1/\Delta$",
        "ylabel": ylab,
        "series": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

                                                    
    markers = ["o", "s", "^", "D", "v", "P"]


                                                 
    selected_rows_all = [r for r in results_sorted if r.N in selected_N]    
    

    for idx, N in enumerate(selected_N):
        rows = [r for r in results_sorted if r.N == N]
        x = np.array([r.M for r in rows], dtype=float)
        y = np.array([r.norm_mean_strict_N3 for r in rows], dtype=float)
        yerr = np.array([r.norm_se_strict_N3 for r in rows], dtype=float)

                                                                               
        y_exact = np.array(
            [exact_mean_strict(N, r.M) / ((N ** 3) * r.delta2_log_inv_delta) for r in rows],
            dtype=float,
        )
        z_scores = (y - y_exact) / np.where(yerr > 0, yerr, np.nan)

        eb = ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=markers[idx % len(markers)],
            linestyle="none",
            capsize=2.0,
            elinewidth=0.8,
            markeredgewidth=0.6,
            label=fr"$N={N}$",
            zorder=3,
        )
        ax.plot(x, y_exact, linestyle="--", linewidth=0.9,
                color=eb[0].get_color(), zorder=2)

        payload["series"].append({
            "N": N,
            "M": x.tolist(),
            "y": y.tolist(),
            "yerr": yerr.tolist(),
            "y_exact": y_exact.tolist(),
            "z_scores": z_scores.tolist(),
        })

    ax.set_xscale("log")
    tick_M = sorted({r.M for r in selected_rows_all})
    ax.set_xticks(tick_M)
    ax.set_xticklabels([str(m) for m in tick_M], fontsize=6)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"Grid resolution $M=1/\Delta$")
    ax.set_ylabel(ylab)

                                         
    yvals_all = np.array([r.norm_mean_strict_N3 for r in selected_rows_all], dtype=float)
    ymin = float(np.min(yvals_all))
    ymax = float(np.max(yvals_all))
    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.plot([], [], linestyle="--", color="0.3", linewidth=0.9,
            label=r"exact $(N)_3T_M/M^6$")
    ax.legend(
        frameon=False,
        handlelength=2.0,
        loc="best",
        ncol=2 if len(selected_N) >= 4 else 1,
    )

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(os.path.splitext(output_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

    write_plot_json(payload, output_json)
    

def make_component_fraction_figure(
    results: List[AggregateResult],
    output_png: str,
    output_json: str,
) -> None:
           
    apply_aps_style()
    plt.rcParams["figure.figsize"] = (7.0, 2.8)

    results_sorted = sorted(results, key=lambda r: (r.N, r.M))
    unique_N = sorted({r.N for r in results_sorted})

    fig, axes = plt.subplots(1, len(unique_N), sharey=True)
    if len(unique_N) == 1:
        axes = [axes]

    payload = {
        "plot_type": "component_fraction_vs_M",
        "panels": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    for ax, N in zip(axes, unique_N):
        rows = [r for r in results_sorted if r.N == N]
        x = np.array([r.M for r in rows], dtype=float)

        strict_frac = np.array([r.frac_strict_of_inclusive for r in rows], dtype=float)
        all_same_frac = np.array([r.frac_all_same_of_inclusive for r in rows], dtype=float)
        two_same_frac = np.array([r.frac_exactly_two_same_of_inclusive for r in rows], dtype=float)

        ax.plot(x, strict_frac, marker="o", linestyle="-", label="strict")
        ax.plot(x, all_same_frac, marker="s", linestyle="--", label="all same")
        ax.plot(x, two_same_frac, marker="^", linestyle=":", label="exactly two same")

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"$M=1/\Delta$")
        ax.set_ylabel("Fraction of inclusive mean")
        ax.legend(frameon=False, handlelength=2.0)
        ax.text(0.05, 0.95, fr"$N={N}$", transform=ax.transAxes, va="top", ha="left")

        payload["panels"].append({
            "N": N,
            "M": x.tolist(),
            "strict_fraction": strict_frac.tolist(),
            "all_same_fraction": all_same_frac.tolist(),
            "exactly_two_same_fraction": two_same_frac.tolist(),
        })

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    write_plot_json(payload, output_json)


                                                                       
    
                                                                       

def write_results_csv(results: List[AggregateResult], csv_path: str) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("quantization_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


                                                                       
          
                                                                       

def parse_int_list(s: str) -> List[int]:
    return [int(tok.strip()) for tok in s.split(",") if tok.strip()]


def run_simulation(
    N_values: List[int],
    M_values: List[int],
    num_trials: int,
    seed: int,
    output_dir: str,
    make_component_plot: bool,
    make_dual_plot: bool,
    manuscript_N_values: Optional[List[int]],
) -> None:    
    
    
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(output_dir, f"quantization_validation_{timestamp}.log")
    raw_csv_path = os.path.join(output_dir, f"quantization_validation_raw_{timestamp}.csv")
    summary_csv_path = os.path.join(output_dir, f"quantization_validation_summary_{timestamp}.csv")
    summary_json_path = os.path.join(output_dir, f"quantization_validation_summary_{timestamp}.json")

    strict_png = os.path.join(output_dir, f"quantization_strict_N3_{timestamp}.png")
    strict_json = os.path.join(output_dir, f"quantization_strict_N3_{timestamp}.json")

    inclusive_png = os.path.join(output_dir, f"quantization_inclusive_N3_{timestamp}.png")
    inclusive_json = os.path.join(output_dir, f"quantization_inclusive_N3_{timestamp}.json")

    dual_png = os.path.join(output_dir, f"quantization_dual_N3_{timestamp}.png")
    dual_json = os.path.join(output_dir, f"quantization_dual_N3_{timestamp}.json")

    component_png = os.path.join(output_dir, f"quantization_component_fractions_{timestamp}.png")
    component_json = os.path.join(output_dir, f"quantization_component_fractions_{timestamp}.json")

    manuscript_png = os.path.join(output_dir, f"quantization_manuscript_main_{timestamp}.png")
    manuscript_json = os.path.join(output_dir, f"quantization_manuscript_main_{timestamp}.json")
    
    logger = setup_logger(log_path)
    rng = np.random.default_rng(seed)

    logger.info("Starting stronger validation package")
    logger.info("Seed = %d", seed)
    logger.info("N values = %s", N_values)
    logger.info("M values = %s", M_values)
    logger.info("Trials per (N,M) = %d", num_trials)
    logger.info("Output directory = %s", output_dir)

    results: List[AggregateResult] = []
    total_jobs = len(N_values) * len(M_values)
    job_counter = 0

    for N in N_values:
        for M in M_values:
            job_counter += 1
            logger.info("Running job %d/%d | N=%d, M=%d, Delta=%.8f", job_counter, total_jobs, N, M, 1.0 / M)

            trials: List[TrialCounts] = []
            checkpoint = max(1, num_trials // 10)

            for t in range(num_trials):
                tc = simulate_one_trial(N, M, rng)
                trials.append(tc)

                if (t == 0) or (t == num_trials - 1) or ((t + 1) % checkpoint == 0):
                    strict_so_far = np.array([z.strict_ordered_collinear for z in trials], dtype=float)
                    incl_so_far = np.array([z.inclusive_ordered_collinear for z in trials], dtype=float)
                    logger.info(
                        "  trial %4d/%4d | strict mean so far = %.6g | inclusive mean so far = %.6g",
                        t + 1, num_trials, float(np.mean(strict_so_far)), float(np.mean(incl_so_far))
                    )

            agg = aggregate_trials(N, M, trials)
            results.append(agg)

            logger.info(
                "Completed N=%d, M=%d | "
                "strict=%.6g +/- %.3g | "
                "inclusive=%.6g +/- %.3g | "
                "strict/N3scale=%.6g | inclusive/N3scale=%.6g | "
                "strict fraction of inclusive=%.4f",
                N, M,
                agg.mean_W_strict, agg.se_W_strict,
                agg.mean_W_inclusive, agg.se_W_inclusive,
                agg.norm_mean_strict_N3,
                agg.norm_mean_inclusive_N3,
                agg.frac_strict_of_inclusive,
            )

    write_results_csv(results, raw_csv_path)
    logger.info("Wrote raw CSV: %s", raw_csv_path)

    summary = compute_summary_statistics(results)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_csv(summary, summary_csv_path)
    logger.info("Wrote summary JSON: %s", summary_json_path)
    logger.info("Wrote summary CSV: %s", summary_csv_path)

    make_normalized_figure(results, strict_png, strict_json, quantity="strict", normalization="N3")
    logger.info("Wrote strict figure: %s", strict_png)

    make_normalized_figure(results, inclusive_png, inclusive_json, quantity="inclusive", normalization="N3")
    logger.info("Wrote inclusive figure: %s", inclusive_png)

    if make_dual_plot:
        make_dual_observable_figure(results, dual_png, dual_json, normalization="N3")
        logger.info("Wrote dual observable figure: %s", dual_png)

    if make_component_plot:
        make_component_fraction_figure(results, component_png, component_json)
        logger.info("Wrote component fraction figure: %s", component_png)

    make_manuscript_main_figure(
        results,
        manuscript_png,
        manuscript_json,
        selected_N=manuscript_N_values,
    )    
    
    
    logger.info("Wrote manuscript main figure: %s", manuscript_png)
    
    gs = summary.get("global_summary", {})
    logger.info("Mean fitted N-slope (strict) = %s", gs.get("mean_N_slope_strict"))
    logger.info("Mean fitted N-slope (inclusive) = %s", gs.get("mean_N_slope_inclusive"))
    logger.info("Mean fitted Delta-factor slope (strict) = %s", gs.get("mean_Delta_factor_slope_strict"))
    logger.info("Mean fitted Delta-factor slope (inclusive) = %s", gs.get("mean_Delta_factor_slope_inclusive"))
    logger.info("Validation package complete")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo cross-check of the 2D quantization count against the exact formula."
    )
    parser.add_argument(
        "--N-values",
        type=parse_int_list,
        default=[60, 100, 140, 180, 220, 260],
        help="Comma-separated N values, e.g. 60,100,140,180,220,260",
    )
    parser.add_argument(
        "--M-values",
        type=parse_int_list,
        default=[16, 24, 32, 48, 64, 96, 128, 192, 256],
        help="Comma-separated M values where Delta=1/M",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=300,
        help="Trials per (N,M) pair",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation_output_quantization_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--no-component-plot",
        action="store_true",
        help="Disable component-fraction figure",
    )
    parser.add_argument(
        "--no-dual-plot",
        action="store_true",
        help="Disable dual strict/inclusive figure",
    )
    parser.add_argument(
        "--manuscript-N-values",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated N values to show in the manuscript figure, e.g. 100,140,180,220",
    )
    return parser

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_simulation(
        N_values=args.N_values,
        M_values=args.M_values,
        num_trials=args.num_trials,
        seed=args.seed,
        output_dir=args.output_dir,
        make_component_plot=not args.no_component_plot,
        make_dual_plot=not args.no_dual_plot,
        manuscript_N_values=args.manuscript_N_values,
    )

if __name__ == "__main__":
    main()