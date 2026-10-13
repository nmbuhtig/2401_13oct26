#!/usr/bin/env python3
   

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              

A_LIMIT = 18.0 / math.pi ** 2                             
TESTED_M = [16, 24, 32, 48, 64, 96, 128, 192, 256]                          
TABLE_M = [16, 32, 64, 128, 256, 2 ** 10, 2 ** 12, 2 ** 16, 2 ** 20]
FIT_M = [2 ** 10, 2 ** 11, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20]


                                                                             
                      
                                                                             

def totient_sieve(n: int) -> np.ndarray:
                                                       
    phi = np.arange(n + 1, dtype=np.int64)
    for p in range(2, n + 1):
        if phi[p] == p:              
            phi[p::p] -= phi[p::p] // p
    return phi


def T_exact(M: int, phi: np.ndarray) -> int:
                                                                        
    if M < 3:
        return 0
    total = 0
    for d in range(2, M):
        J = (M - 1) // d
        S = M + 2 * (J * M - d * J * (J + 1) // 2)
        total += int(phi[d]) * (S * S - M * M)
    return 3 * total


def T_bruteforce(M: int) -> int:
                                                                             
    pts = [(x, y) for x in range(M) for y in range(M)]
    count = 0
    for a, b, c in itertools.permutations(pts, 3):
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0:
            count += 1
    return count


def T_gcd_sum(M: int) -> int:
                                                                           
    d = np.arange(-(M - 1), M, dtype=np.int64)
    dx, dy = np.meshgrid(d, d, indexing="ij")
    g = np.gcd(np.abs(dx), np.abs(dy))
    w = (M - np.abs(dx)) * (M - np.abs(dy))
    mask = g > 0
    return 3 * int(np.sum((w * (g - 1))[mask]))


def normalized_ratio(T: int, M: int) -> float:
                            
    return T / (M ** 4 * math.log(M))


def second_order(T: int, M: int) -> float:
                                      
    return T / M ** 4 - A_LIMIT * math.log(M)


                                                                             
              
                                                                             

def run_verification(phi: np.ndarray) -> Dict[str, object]:
    brute = {}
    for M in range(2, 9):
        tb, te = T_bruteforce(M), T_exact(M, phi)
        if tb != te:
            raise AssertionError(f"Brute-force mismatch at M={M}: {tb} vs {te}")
        brute[M] = te
    gcd_check = {}
    for M in (16, 64, 256):
        tg, te = T_gcd_sum(M), T_exact(M, phi)
        if tg != te:
            raise AssertionError(f"gcd-sum mismatch at M={M}: {tg} vs {te}")
        gcd_check[M] = te
    return {
        "bruteforce_M_2_to_8": {str(k): v for k, v in brute.items()},
        "gcd_sum_check": {str(k): v for k, v in gcd_check.items()},
        "status": "passed",
    }


                                                                             
      
                                                                             

def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,                                                                
        "ps.fonttype": 42,
    })


def make_figure(m_grid: List[int], ratio: List[float], second: List[float],
                tested: List[int], ratio_t: List[float], second_t: List[float],
                m_max: int, png: str, pdf: str, dpi: int) -> None:
    apply_style()
    ink = "#1f3b73"
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

                                                 
    ax[0].plot(m_grid, ratio, "-", color=ink, lw=1.6)
    ax[0].plot(tested, ratio_t, "o", ms=5, color=ink, mfc="white", mew=1.4)
    ax[0].axhline(A_LIMIT, ls="--", color="0.35", lw=1.2)
    ax[0].text(12, A_LIMIT - 0.045, r"$18/\pi^2$", color="0.25")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"Grid resolution $M=1/\Delta$")
    ax[0].set_ylabel(r"$T_M/(M^4\log M)$")
    ax[0].set_ylim(1.15, 1.9)
    ax[0].set_title("(a)", loc="left", fontsize=10)

                           
    ax[1].plot(m_grid, second, "-", color=ink, lw=1.6)
    ax[1].plot(tested, second_t, "o", ms=5, color=ink, mfc="white", mew=1.4)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"Grid resolution $M=1/\Delta$")
    ax[1].set_ylabel(r"$T_M/M^4-(18/\pi^2)\log M$")
    ax[1].set_ylim(-1.56, -1.50)
    ax[1].set_title("(b)", loc="left", fontsize=10)

    for a in ax:
        a.grid(True, which="major", color="0.9", lw=0.6)
        a.tick_params(which="both", top=True, right=True)
        a.set_xlim(8, 1.25 * m_max)

    fig.tight_layout()
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)


                                                                             
      
                                                                             

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m-min", type=int, default=8, help="smallest M on the curve")
    p.add_argument("--m-max", type=int, default=2 ** 20, help="largest M on the curve")
    p.add_argument("--n-curve", type=int, default=140,
                   help="number of geometrically spaced M values on the curve")
    p.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    p.add_argument("--output-dir", default=".", help="output directory")
    p.add_argument("--stem", default="quantization_exact_main", help="output file stem")
    p.add_argument("--skip-verification", action="store_true",
                   help="skip brute-force and gcd-sum checks")
    args = p.parse_args()

    m_upper = max(args.m_max, max(TABLE_M), max(FIT_M))
    phi = totient_sieve(m_upper)

    verification = {"status": "skipped"} if args.skip_verification else run_verification(phi)

    m_grid = sorted(set(
        np.unique(np.round(np.geomspace(args.m_min, args.m_max, args.n_curve)).astype(int)).tolist()
    ) | set(TESTED_M))
    T_curve = {M: T_exact(M, phi) for M in m_grid}
    ratio = [normalized_ratio(T_curve[M], M) for M in m_grid]
    second = [second_order(T_curve[M], M) for M in m_grid]
    ratio_t = [normalized_ratio(T_curve[M], M) for M in TESTED_M]
    second_t = [second_order(T_curve[M], M) for M in TESTED_M]

                                                       
    x_fit = np.array([1.0 / math.log(M) for M in FIT_M])
    y_fit = np.array([normalized_ratio(T_exact(M, phi), M) for M in FIT_M])
    B_fit, A_fit = np.polyfit(x_fit, y_fit, 1)

                                         
    table = []
    for M in TABLE_M:
        T = T_exact(M, phi)
        r, s = normalized_ratio(T, M), second_order(T, M)
        table.append({
            "M": M,
            "T_M": str(T),                                                                
            "T_M_over_M4logM": r,                                    
            "second_order_term": s,                                  
            "T_M_over_M4logM_4dp": f"{r:.4f}",                                
            "second_order_term_4dp": f"{s:.4f}",
        })

    os.makedirs(args.output_dir, exist_ok=True)
    png = os.path.join(args.output_dir, args.stem + ".png")
    pdf = os.path.join(args.output_dir, args.stem + ".pdf")
    js = os.path.join(args.output_dir, args.stem + ".json")

    make_figure(m_grid, ratio, second, TESTED_M, ratio_t, second_t,
                args.m_max, png, pdf, args.dpi)

    payload = {
        "description": "Exact two-dimensional quantization count T_M and Fig. 1 data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "parameters": vars(args),
        "limiting_constant_18_over_pi2": A_LIMIT,
        "verification": verification,
        "fit_A_plus_B_over_logM": {
            "M_values": FIT_M,
            "A": float(A_fit),
            "B": float(B_fit),
        },
        "table": table,
        "panel_a": {
            "ylabel": "T_M/(M^4 log M)",
            "curve": {"M": m_grid, "y": ratio},
            "tested_points": {"M": TESTED_M, "y": ratio_t},
        },
        "panel_b": {
            "ylabel": "T_M/M^4 - (18/pi^2) log M",
            "curve": {"M": m_grid, "y": second},
            "tested_points": {"M": TESTED_M, "y": second_t},
        },
        "outputs": {"png": png, "pdf": pdf, "json": js},
    }
    with open(js, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Verification: {verification['status']}")
    print(f"Fit: A = {A_fit:.5f} (18/pi^2 = {A_LIMIT:.5f}), B = {B_fit:.5f}")
    for row in table:
        print(f"  M={row['M']:>8}  T_M/(M^4 log M)={row['T_M_over_M4logM_4dp']}"
              f"  second-order={row['second_order_term_4dp']}")
    print(f"Wrote {png}\nWrote {pdf}\nWrote {js}")


if __name__ == "__main__":
    main()
