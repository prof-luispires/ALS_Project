#!/usr/bin/env python3
"""
hybrid_channel_env_multiscenario.py
-----------------------------------
Ambiente Gymnasium para seleção adaptativa de canal (RF, FSO, THz)
num sistema híbrido sob múltiplos cenários ambientais (clear, rain, fog, worst).

Estado s_t:
    [distance_m,
     rain_rate_mmph,
     humidity_pct,
     visibility_km,
     Cn2,
     SNR_RF_dB, SNR_FSO_dB, SNR_THz_dB,
     Eb_RF_J_per_bit, Eb_FSO_J_per_bit, Eb_THz_J_per_bit]

Ação a_t:
    0 -> RF
    1 -> FSO
    2 -> THz

Recompensa (simples, ajustável):
    +1.0  se ação == best_channel (argmax SNR)
    -0.5  caso contrário
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from generate_scenario_dataset import SCENARIOS, snr_capacity_energy
from rf_channel_vectorized import rf_atten_db
from fso_channel_vectorized import fso_atten_db
from thz_channel_vectorized import thz_atten_db


class HybridChannelEnvMulti(gym.Env):
    """
    Ambiente RL multi-cenário para seleção de canal RF/FSO/THz.
    Compatível com Gymnasium e Stable-Baselines3 (DQN).
    """

    metadata = {"render_modes": []}

    def __init__(self, max_distance: float = 4000.0, step_distance: float = 10.0):
        super().__init__()

        # Parâmetros de varrimento de distância
        self.max_distance = float(max_distance)
        self.step_distance = float(step_distance)

        # Cenários disponíveis (chaves de SCENARIOS)
        self.scenario_names = list(SCENARIOS.keys())

        # Espaço de ações: 3 canais (0=RF, 1=FSO, 2=THz)
        self.action_space = spaces.Discrete(3)

        # Espaço de observações (11 features)
        # Valores limites razoavelmente largos (podem ser afinados)
        low = np.array([
            0.0,      # distance_m
            0.0,      # rain_rate_mmph
            0.0,      # humidity_pct
            0.0,      # visibility_km
            0.0,      # Cn2 (valores pequenos, mas aqui 0 até algo)
            -100.0,   # SNR_RF_dB
            -100.0,   # SNR_FSO_dB
            -100.0,   # SNR_THz_dB
            0.0,      # Eb_RF_J_per_bit
            0.0,      # Eb_FSO_J_per_bit
            0.0       # Eb_THz_J_per_bit
        ], dtype=np.float32)

        high = np.array([
            5000.0,   # distance_m
            100.0,    # rain_rate_mmph
            100.0,    # humidity_pct
            25.0,     # visibility_km
            1e-10,    # Cn2
            80.0,     # SNR_RF_dB
            80.0,     # SNR_FSO_dB
            80.0,     # SNR_THz_dB
            1e-10,    # Eb_RF_J_per_bit
            1e-10,    # Eb_FSO_J_per_bit
            1e-10     # Eb_THz_J_per_bit
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(11,),
            dtype=np.float32
        )

        # Estado interno
        self.current_scenario = None
        self.distance_m = None

        # Métricas de canal no estado atual
        self.snr_rf_dB = None
        self.snr_fso_dB = None
        self.snr_thz_dB = None
        self.eb_rf = None
        self.eb_fso = None
        self.eb_thz = None

    # ------------------------------------------------------------------
    # Funções auxiliares internas
    # ------------------------------------------------------------------

    def _sample_scenario(self) -> str:
        """
        Escolhe aleatoriamente um dos cenários definidos em SCENARIOS.
        """
        return np.random.choice(self.scenario_names)

    def _update_channels(self):
        """
        Atualiza SNR e energia por bit (Eb) para RF, FSO e THz,
        com base na distância atual e no cenário escolhido.
        """
        p = SCENARIOS[self.current_scenario]  # p é um dicionário

        d_m = np.array([self.distance_m], dtype=float)
        d_km = d_m / 1000.0

        # RF
        L_RF = rf_atten_db(
            d_m,
            f_Hz=5e9,
            rain_rate_mmph=p["rain_rate_mmph"],
            k=1e-4,
            alpha=0.9,
            shadow_sigma_db=2.0,
            rng_seed=None,   # pode fixar semente se quiser determinismo
        )
        snr_rf_dB, _, eb_rf = snr_capacity_energy(L_RF)

        # FSO
        L_FSO = fso_atten_db(
            d_km,
            visibility_km=p["visibility_km"],
            wavelength_m=1550e-9,
            Cn2=p["Cn2"],
        )
        snr_fso_dB, _, eb_fso = snr_capacity_energy(L_FSO)

        # THz
        L_THz = thz_atten_db(
            d_m,
            f_Hz=300e9,
            humidity_pct=p["humidity_pct"],
            a0_dbkm=0.5,
            a1_dbkm_per_pct=0.03,
        )
        snr_thz_dB, _, eb_thz = snr_capacity_energy(L_THz)

        # Guardar valores escalares
        self.snr_rf_dB = float(snr_rf_dB[0])
        self.snr_fso_dB = float(snr_fso_dB[0])
        self.snr_thz_dB = float(snr_thz_dB[0])

        self.eb_rf = float(eb_rf[0])
        self.eb_fso = float(eb_fso[0])
        self.eb_thz = float(eb_thz[0])

    def _get_observation(self):
        """
        Constrói o vetor de estado s_t com 11 features.
        """
        p = SCENARIOS[self.current_scenario]  # dicionário

        obs = np.array([
            self.distance_m,
            p["rain_rate_mmph"],
            p["humidity_pct"],
            p["visibility_km"],
            p["Cn2"],
            self.snr_rf_dB,
            self.snr_fso_dB,
            self.snr_thz_dB,
            self.eb_rf,
            self.eb_fso,
            self.eb_thz,
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # Métodos obrigatórios Gymnasium
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """
        Reinicia o episódio:
        - escolhe cenário
        - coloca distância no valor inicial (10 m)
        - atualiza canais
        - devolve observação e info
        """
        super().reset(seed=seed)

        self.current_scenario = self._sample_scenario()
        self.distance_m = self.step_distance  # começa em 10 m, 20 m, etc.

        self._update_channels()
        obs = self._get_observation()

        info = {
            "scenario": self.current_scenario,
            "distance_m": self.distance_m,
        }

        return obs, info

    def step(self, action):
        """
        Executa uma ação (escolha de canal) e avança a distância.
        """
        # Garantir que ação é válida
        assert self.action_space.contains(action), "Ação inválida"

        # Determinar canal ótimo (max SNR)
        snrs = np.array([self.snr_rf_dB, self.snr_fso_dB, self.snr_thz_dB])
        best_channel = int(np.argmax(snrs))

        # Recompensa simples (pode ser refinada)
        if int(action) == best_channel:
            reward = 1.0
        else:
            reward = -0.5

        # Atualizar distância
        self.distance_m += self.step_distance

        # Verificar fim de episódio
        terminated = self.distance_m > self.max_distance
        truncated = False  # não usamos truncation aqui

        if not terminated:
            # Atualizar canais e observação para a nova distância
            self._update_channels()
            obs = self._get_observation()
        else:
            # Se terminou, pode devolver o último estado ou um dummy
            obs = self._get_observation()

        info = {
            "scenario": self.current_scenario,
            "distance_m": self.distance_m,
            "snrs": np.array([self.snr_rf_dB, self.snr_fso_dB, self.snr_thz_dB], dtype=float),
            "best_channel": best_channel,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Não usamos renderização neste trabalho.
        Pode ser expandido para debug se necessário.
        """
        pass

    def close(self):
        """
        Nada a fechar explicitamente neste ambiente.
        """
        pass
