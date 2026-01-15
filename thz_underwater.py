#!/usr/bin/env python3
"""
Underwater THz link: FSPL + frequency-dependent absorption.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

C0 = 299792458.0

def thz_underwater_snr_capacity(
    f_hz=300e9,
    alpha_abs_dB_per_m=100.0,
    Pt_dBm=0.0,
    Gt_dB=0.0,
    Gr_dB=0.0,
    B_hz=100e6,
    NF_dB=6.0,
    d_min=0.01, d_max=0.5, num=100
):
    d = np.linspace(d_min, d_max, num)
    FSPL_dB = 20.0*np.log10(4.0*np.pi*f_hz*d/C0)
    L_dB = FSPL_dB + alpha_abs_dB_per_m * d
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = -174.0 + 10.0*np.log10(B_hz) + NF_dB
    SNR_dB = Pr_dBm - N_dBm
    SNR_lin = 10**(SNR_dB/10.0)
    C_bps = B_hz * np.log2(1.0 + SNR_lin)
    df = pd.DataFrame({
        "distance_m": d,
        "FSPL_dB": FSPL_dB,
        "path_loss_dB": L_dB,
        "Pr_dBm": Pr_dBm,
        "SNR_dB": SNR_dB,
        "Capacity_bps": C_bps
    })
    return df

def main():
    df = thz_underwater_snr_capacity()
    df.to_csv("thz_underwater_results.csv", index=False)
    plt.figure()
    plt.plot(df["distance_m"], df["SNR_dB"])
    plt.xlabel("Distance (m)")
    plt.ylabel("SNR (dB)")
    plt.title("Underwater THz: SNR vs Distance")
    plt.grid(True)
    plt.savefig("thz_underwater_snr.png", dpi=150, bbox_inches="tight")
    plt.figure()
    plt.plot(df["distance_m"], df["Capacity_bps"])
    plt.xlabel("Distance (m)")
    plt.ylabel("Capacity (bps)")
    plt.title("Underwater THz: Capacity vs Distance")
    plt.grid(True)
    plt.savefig("thz_underwater_capacity.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
