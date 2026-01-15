"""
channel_metrics.py
------------------
Funções auxiliares para cálculo da energia por bit (Eb) e da probabilidade
de outage a partir dos resultados dos canais RF/FSO/THz.

Este módulo foi pensado para trabalhar em conjunto com os ficheiros:
    rf_channel.py
    rf_underwater.py
    thz_channel.py
    thz_underwater.py
    fso_channel.py
    fso_underwater.py

Ideia geral:
    - A energia por bit mede o custo energético de transmitir 1 bit.
    - A probabilidade de outage mede a fracção de instantes ou condições
      em que o canal não cumpre um requisito mínimo (por exemplo,
      SNR abaixo de um limiar ou capacidade abaixo de um débito-alvo).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Conversões de potência
# ---------------------------------------------------------------------------

def dbm_to_watt(P_dBm: float | np.ndarray) -> np.ndarray:
    """Converte potência em dBm para watt.

    Fórmula:
        P[W] = 10^((P[dBm] - 30)/10)
    """
    P_dBm = np.asarray(P_dBm, dtype=float)
    return 10.0 ** ((P_dBm - 30.0) / 10.0)


def watt_to_dbm(P_W: float | np.ndarray) -> np.ndarray:
    """Converte potência em watt para dBm.

    Fórmula inversa:
        P[dBm] = 10*log10(P[W]) + 30
    """
    P_W = np.asarray(P_W, dtype=float)
    return 10.0 * np.log10(P_W) + 30.0


# ---------------------------------------------------------------------------
# Energia por bit
# ---------------------------------------------------------------------------

def energy_per_bit_joule(
    P_dBm: float | None = None,
    P_W: float | None = None,
    bitrate_bps: float = 1e6,
    efficiency: float = 1.0,
) -> float:
    """Calcula a energia por bit (Eb) em joule.

    Parâmetros
    ----------
    P_dBm : float, opcional
        Potência de transmissão em dBm. Se fornecida, tem prioridade sobre P_W.
    P_W : float, opcional
        Potência de transmissão em watt (valor linear).
    bitrate_bps : float
        Débito binário (bits por segundo) desejado.
    efficiency : float
        Eficiência global do transmissor (0<eff<=1). Por omissão assume-se 1.0,
        ou seja, potência eléctrica ≈ potência radiada.

    Fórmula
    -------
        Eb = P_efectiva / Rb

    onde:
        - Eb é a energia por bit [J/bit]
        - P_efectiva é a potência efectiva consumida (W)
        - Rb é a taxa de bits (bit/s)

    Notas
    -----
    - A inclusão da eficiência permite aproximar consumos reais
      (amplificador de potência, circuitos de RF, etc.).
    """
    if P_dBm is not None:
        P_W_local = dbm_to_watt(P_dBm)
    elif P_W is not None:
        P_W_local = float(P_W)
    else:
        raise ValueError("É necessário especificar P_dBm ou P_W.")

    if bitrate_bps <= 0:
        raise ValueError("bitrate_bps deve ser positivo.")

    if not (0 < efficiency <= 1.0):
        raise ValueError("efficiency deve estar em (0,1].")

    P_effective = P_W_local / efficiency
    Eb = P_effective / float(bitrate_bps)
    return float(Eb)


# ---------------------------------------------------------------------------
# Probabilidade de outage baseada em SNR
# ---------------------------------------------------------------------------

def outage_probability_snr(
    snr_dB: np.ndarray,
    snr_threshold_dB: float,
) -> tuple[float, np.ndarray]:
    """Calcula a probabilidade de outage com base num limiar de SNR.

    Definição
    ---------
    Considera-se que há outage quando:
        SNR[dB] < SNR_threshold[dB]

    A probabilidade de outage é então:
        P_out = N_outage / N_total

    onde:
        - N_outage é o número de amostras que violam o limiar,
        - N_total é o número total de amostras.

    Parâmetros
    ----------
    snr_dB : array-like
        Valores de SNR em dB (por exemplo, ao longo da distância ou de
        diferentes cenários).
    snr_threshold_dB : float
        Limiar mínimo de SNR necessário para garantir BER/qualidade alvo.

    Retorna
    -------
    P_out : float
        Probabilidade de outage (entre 0 e 1).
    outage_flags : np.ndarray (bool)
        Vector booleano indicando, para cada amostra, se está em outage.
    """
    snr_dB = np.asarray(snr_dB, dtype=float)
    outage_flags = snr_dB < float(snr_threshold_dB)
    P_out = float(np.mean(outage_flags.astype(float)))
    return P_out, outage_flags


# ---------------------------------------------------------------------------
# Probabilidade de outage baseada em capacidade
# ---------------------------------------------------------------------------

def outage_probability_capacity(
    snr_dB: np.ndarray,
    bandwidth_hz: float,
    rate_target_bps: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calcula a probabilidade de outage com base na capacidade do canal.

    Definição
    ---------
    Usando a fórmula de Shannon:
        C = B * log2(1 + SNR_linear)

    Considera-se outage quando:
        C < R_target

    Parâmetros
    ----------
    snr_dB : array-like
        Valores de SNR em dB.
    bandwidth_hz : float
        Largura de banda efectiva do canal (Hz).
    rate_target_bps : float
        Débito alvo desejado (bit/s).

    Retorna
    -------
    P_out : float
        Probabilidade de outage (0 a 1).
    outage_flags : np.ndarray (bool)
        Indicador de outage para cada amostra.
    capacity_bps : np.ndarray
        Capacidade instantânea calculada (bit/s) para cada amostra.

    Notas
    -----
    Esta definição é útil quando a métrica principal é a capacidade/
    débito útil, e não apenas a SNR. Está alinhada com a definição
    clássica de outage capacity em teoria de informação.
    """
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz deve ser positivo.")
    if rate_target_bps <= 0:
        raise ValueError("rate_target_bps deve ser positivo.")

    snr_dB = np.asarray(snr_dB, dtype=float)
    snr_lin = 10.0 ** (snr_dB / 10.0)
    capacity_bps = bandwidth_hz * np.log2(1.0 + snr_lin)

    outage_flags = capacity_bps < float(rate_target_bps)
    P_out = float(np.mean(outage_flags.astype(float)))
    return P_out, outage_flags, capacity_bps


# ---------------------------------------------------------------------------
# Funções auxiliares para DataFrames
# ---------------------------------------------------------------------------

def add_metrics_to_df_snr(
    df: pd.DataFrame,
    snr_col: str,
    pt_dBm: float,
    bitrate_bps: float,
    snr_threshold_dB: float | None = None,
    eb_col: str = "Eb_J",
    outage_col: str = "outage_flag",
) -> tuple[pd.DataFrame, dict]:
    """Adiciona Eb e, opcionalmente, flags de outage (SNR) a um DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com, pelo menos, uma coluna de SNR em dB.
    snr_col : str
        Nome da coluna com SNR em dB.
    pt_dBm : float
        Potência de transmissão (dBm) associada a este cenário.
    bitrate_bps : float
        Débito em bit/s.
    snr_threshold_dB : float, opcional
        Se fornecido, calcula e adiciona flags de outage em função da SNR.
    eb_col : str
        Nome da coluna a criar para Eb (J/bit).
    outage_col : str
        Nome da coluna a criar para a flag de outage (0/1).

    Retorna
    -------
    df_out : pd.DataFrame
        DataFrame com as novas colunas.
    summary : dict
        Dicionário com métricas agregadas (por exemplo, P_out).
    """
    if snr_col not in df.columns:
        raise KeyError(f"Coluna '{snr_col}' não encontrada no DataFrame.")

    df = df.copy()
    Eb = energy_per_bit_joule(P_dBm=pt_dBm, bitrate_bps=bitrate_bps)
    df[eb_col] = Eb  # constante ao longo do cenário

    summary: dict = {"Eb_J": Eb}

    if snr_threshold_dB is not None:
        P_out, flags = outage_probability_snr(df[snr_col].values, snr_threshold_dB)
        df[outage_col] = flags.astype(int)
        summary["P_out_snr"] = P_out
        summary["snr_threshold_dB"] = snr_threshold_dB

    return df, summary


def add_metrics_to_df_capacity(
    df: pd.DataFrame,
    snr_col: str,
    pt_dBm: float,
    bitrate_bps: float,
    bandwidth_hz: float,
    rate_target_bps: float,
    eb_col: str = "Eb_J",
    outage_col: str = "outage_flag_cap",
    cap_col: str = "capacity_bps",
) -> tuple[pd.DataFrame, dict]:
    """Adiciona Eb, capacidade e flags de outage (por capacidade) a um DataFrame.

    Esta função é útil quando, além da SNR, se pretende avaliar se o
    canal suporta ou não um débito alvo específico.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com coluna de SNR em dB.
    snr_col : str
        Nome da coluna com SNR em dB.
    pt_dBm : float
        Potência de transmissão em dBm.
    bitrate_bps : float
        Débito em bit/s (usado para Eb).
    bandwidth_hz : float
        Largura de banda do canal (Hz).
    rate_target_bps : float
        Débito alvo que se pretende garantir (bit/s).
    eb_col : str
        Nome da coluna para Eb.
    outage_col : str
        Nome da coluna para a flag de outage (capacidade).
    cap_col : str
        Nome da coluna para a capacidade (bit/s).

    Retorna
    -------
    df_out : pd.DataFrame
        DataFrame com novas colunas.
    summary : dict
        Dicionário com métricas agregadas (P_out, Eb, etc.).
    """
    if snr_col not in df.columns:
        raise KeyError(f"Coluna '{snr_col}' não encontrada no DataFrame.")

    df = df.copy()
    Eb = energy_per_bit_joule(P_dBm=pt_dBm, bitrate_bps=bitrate_bps)
    df[eb_col] = Eb

    P_out, flags, cap = outage_probability_capacity(
        df[snr_col].values,
        bandwidth_hz=bandwidth_hz,
        rate_target_bps=rate_target_bps,
    )
    df[cap_col] = cap
    df[outage_col] = flags.astype(int)

    summary: dict = {
        "Eb_J": Eb,
        "P_out_capacity": P_out,
        "bandwidth_hz": bandwidth_hz,
        "rate_target_bps": rate_target_bps,
    }
    return df, summary
