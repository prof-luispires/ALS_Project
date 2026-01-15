#!/usr/bin/env python3
"""
Underwater RF link: attenuation in conductive medium, SNR and capacity vs distance.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rf_underwater_snr_capacity(
    f_hz=100e3,
    sigma=4.0,
    mu=4*np.pi*1e-7,
    Pt_dBm=30.0,
    Gt_dB=0.0,
    Gr_dB=0.0,
    B_hz=1000.0,
    NF_dB=5.0,
    d_min=0.1, d_max=10.0, num=100
):
    d = np.linspace(d_min, d_max, num)
    alpha = np.sqrt(np.pi * f_hz * mu * sigma)
    L_dB = 8.686 * alpha * d
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = -174.0 + 10.0*np.log10(B_hz) + NF_dB
    SNR_dB = Pr_dBm - N_dBm
    SNR_lin = 10**(SNR_dB/10.0)
    C_bps = B_hz * np.log2(1.0 + SNR_lin)
    df = pd.DataFrame({
        "distance_m": d,
        "alpha_Np_per_m": np.full_like(d, alpha),
        "path_loss_dB": L_dB,
        "Pr_dBm": Pr_dBm,
        "SNR_dB": SNR_dB,
        "Capacity_bps": C_bps
    })
    return df

def main():
    df = rf_underwater_snr_capacity()
    df.to_csv("rf_underwater_results.csv", index=False)
    plt.figure()
    plt.plot(df["distance_m"], df["SNR_dB"])
    plt.xlabel("Distance (m)")
    plt.ylabel("SNR (dB)")
    plt.title("Underwater RF: SNR vs Distance")
    plt.grid(True)
    plt.savefig("rf_underwater_snr.png", dpi=150, bbox_inches="tight")
    plt.figure()
    plt.plot(df["distance_m"], df["Capacity_bps"])
    plt.xlabel("Distance (m)")
    plt.ylabel("Capacity (bps)")
    plt.title("Underwater RF: Capacity vs Distance")
    plt.grid(True)
    plt.savefig("rf_underwater_capacity.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
