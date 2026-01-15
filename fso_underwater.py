#!/usr/bin/env python3
"""
Underwater FSO/VLC link: Beer-Lambert attenuation + pointing jitter penalty.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fso_underwater_snr_capacity(
    c_per_m=0.2,
    G_pointing_dB=-1.0,
    Pt_dBm=10.0,
    Gt_dB=0.0,
    Gr_dB=0.0,
    B_hz=10e6,
    NF_dB=5.0,
    d_min=1.0, d_max=50.0, num=100
):
    d = np.linspace(d_min, d_max, num)
    L_dB = 4.343 * c_per_m * d - G_pointing_dB
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = -174.0 + 10.0*np.log10(B_hz) + NF_dB
    SNR_dB = Pr_dBm - N_dBm
    SNR_lin = 10**(SNR_dB/10.0)
    C_bps = B_hz * np.log2(1.0 + SNR_lin)
    df = pd.DataFrame({
        "distance_m": d,
        "extinction_per_m": np.full_like(d, c_per_m),
        "path_loss_dB": L_dB,
        "Pr_dBm": Pr_dBm,
        "SNR_dB": SNR_dB,
        "Capacity_bps": C_bps
    })
    return df

def main():
    df = fso_underwater_snr_capacity()
    df.to_csv("fso_underwater_results.csv", index=False)
    plt.figure()
    plt.plot(df["distance_m"], df["SNR_dB"])
    plt.xlabel("Distance (m)")
    plt.ylabel("SNR (dB)")
    plt.title("Underwater FSO: SNR vs Distance")
    plt.grid(True)
    plt.savefig("fso_underwater_snr.png", dpi=150, bbox_inches="tight")
    plt.figure()
    plt.plot(df["distance_m"], df["Capacity_bps"])
    plt.xlabel("Distance (m)")
    plt.ylabel("Capacity (bps)")
    plt.title("Underwater FSO: Capacity vs Distance")
    plt.grid(True)
    plt.savefig("fso_underwater_capacity.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
