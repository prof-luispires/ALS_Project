"""
hybrid_channel_env_multiscenario.py
-----------------------------------
Ambiente Gymnasium para seleção adaptativa de canal RF/FSO/THz
em múltiplos cenários (clear, rain, fog, worst).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rf_channel_vectorized as rf_channel
import fso_channel_vectorized as fso_channel
import thz_channel_vectorized as thz_channel

# Importamos apenas a função de métricas (não o SCENARIOS)
from generate_scenario_dataset import snr_capacity_energy

# ----------------------------------------------------------------------
# SCENARIOS (copiado de generate_scenario_dataset.py)
# ----------------------------------------------------------------------
SCENARIOS = {
    "clear": {
        "rain_rate_mmph": 0,
        "humidity_pct": 40,
        "visibility_km": 20,
        "Cn2": 1e-15,
        "sigma_sub_RF": 3.0,
        "alpha_THz": 20,
        "c_FSO": 0.05,
    },
    "rain": {
        "rain_rate_mmph": 25,
        "humidity_pct": 70,
        "visibility_km": 5,
        "Cn2": 5e-15,
        "sigma_sub_RF": 4.0,
        "alpha_THz": 50,
        "c_FSO": 0.2,
    },
    "fog": {
        "rain_rate_mmph": 50,
        "humidity_pct": 90,
        "visibility_km": 1,
        "Cn2": 1e-14,
        "sigma_sub_RF": 5.0,
        "alpha_THz": 100,
        "c_FSO": 0.5,
    },
    "worst": {
        "rain_rate_mmph": 80,
        "humidity_pct": 95,
        "visibility_km": 0.5,
        "Cn2": 5e-14,
        "sigma_sub_RF": 6.0,
        "alpha_THz": 150,
        "c_FSO": 1.0,
    },
}


class HybridChannelEnvMulti(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_distance_m: float = 4000.0, step_m: float = 10.0):
        super().__init__()

        self.max_distance_m = max_distance_m
        self.step_m = step_m

        # Espaço de ações: 0=RF, 1=FSO, 2=THz
        self.action_space = spaces.Discrete(3)

        # Observação: [distance_norm, rain_norm, hum_norm, vis_norm, Cn2_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.scenario_names = list(SCENARIOS.keys())
        self.current_scenario_name = None
        self.current_params = None
        self.distance_m = None

    # --------------------------------------------------------
    # Normalização do estado
    # --------------------------------------------------------
    def _normalize_state(self, distance_m, rain, humidity, visibility, Cn2):
        d_norm = np.clip(distance_m / self.max_distance_m, 0.0, 1.0)
        rain_norm = np.clip(rain / 100.0, 0.0, 1.0)
        hum_norm = np.clip((humidity - 30.0) / 70.0, 0.0, 1.0)
        vis_norm = np.clip(visibility / 25.0, 0.0, 1.0)
        cn2_norm = np.clip((np.log10(Cn2) + 15) / 3.0, 0.0, 1.0)
        return np.array(
            [d_norm, rain_norm, hum_norm, vis_norm, cn2_norm],
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Reset do ambiente
    # --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Escolher cenário aleatório
        self.current_scenario_name = np.random.choice(self.scenario_names)
        self.current_params = SCENARIOS[self.current_scenario_name]
        assert isinstance(self.current_params, dict), \
            f"current_params não é dict: {type(self.current_params)}"

        # Começar na menor distância
        self.distance_m = self.step_m

        obs = self._get_observation()
        info = {"scenario": self.current_scenario_name}
        return obs, info

    def _get_observation(self):
        p = self.current_params
        assert isinstance(p, dict), f"p não é dict: {type(p)}"
        return self._normalize_state(
            self.distance_m,
            p["rain_rate_mmph"],
            p["humidity_pct"],
            p["visibility_km"],
            p["Cn2"],
        )

    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------
    def step(self, action):
        p = self.current_params
        d_m = np.array([self.distance_m], dtype=float)
        d_km = d_m / 1000.0

        # ---------------- RF ----------------
        L_RF_dB = rf_channel.rf_atten_db(
            d_m,
            f_Hz=5e9,
            rain_rate_mmph=p["rain_rate_mmph"],
            k=1e-4,
            alpha=0.9,
            shadow_sigma_db=2.0,
        )
        snr_RF_dB, cap_RF_bps, Eb_RF_J = snr_capacity_energy(L_RF_dB)

        # ---------------- FSO ----------------
        L_FSO_dB = fso_channel.fso_atten_db(
            d_km,
            visibility_km=p["visibility_km"],
            wavelength_m=1550e-9,
            Cn2=p["Cn2"],
        )
        snr_FSO_dB, cap_FSO_bps, Eb_FSO_J = snr_capacity_energy(L_FSO_dB)

        # ---------------- THz ----------------
# ---------------- THz ----------------
# Chamada compatível com a versão local de thz_atten_db
        L_THz_dB = thz_channel.thz_atten_db(
            d_m,
            f_Hz=300e9,
            humidity_pct=p["humidity_pct"],
        )
        snr_THz_dB, cap_THz_bps, Eb_THz_J = snr_capacity_energy(L_THz_dB)

        # Vetores
        snrs = np.array([snr_RF_dB[0], snr_FSO_dB[0], snr_THz_dB[0]])
        Ebs = np.array([Eb_RF_J[0], Eb_FSO_J[0], Eb_THz_J[0]])

        # Normalizações
        snr_norm = np.clip((snrs - (-20.0)) / (40.0 - (-20.0)), 0.0, 1.0)
        Eb_norm = np.clip(Ebs / np.max(Ebs + 1e-12), 0.0, 1.0)

        chosen_snr = snr_norm[action]
        chosen_Eb = Eb_norm[action]

        # Outage para o canal escolhido
        _, cap_chosen, _ = snr_capacity_energy(
            np.array([
                L_RF_dB[0] if action == 0 else
                L_FSO_dB[0] if action == 1 else
                L_THz_dB[0]
            ])
        )
        outage = (cap_chosen[0] < 10e6) or (snrs[action] < 0.0)

        # Recompensa
        reward = 0.0
        reward += 2.0 * chosen_snr         # premiar boa SNR
        reward -= 2.0 * chosen_Eb          # penalizar energia alta
        if outage:
            reward -= 5.0                  # penalização forte

        best_idx = int(np.argmax(snrs))
        if action == best_idx:
            reward += 2.0                  # bónus se escolhe o melhor canal físico

        # Avançar distância
        self.distance_m += self.step_m
        terminated = self.distance_m > self.max_distance_m

        obs = self._get_observation()
        info = {
            "scenario": self.current_scenario_name,
            "snrs": snrs,
            "Ebs": Ebs,
            "best_channel": best_idx,
        }
        return obs, float(reward), bool(terminated), False, info

    def render(self):
        pass

    def close(self):
        pass
