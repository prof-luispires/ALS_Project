#!/usr/bin/env python3
"""
train_dqn_multiscenario.py
--------------------------
Treino completo de DQN para o ambiente multi-cenário HybridChannelEnvMulti.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from hybrid_channel_env_multiscenario import HybridChannelEnvMulti

def main():
    env = HybridChannelEnvMulti()
    env = Monitor(env)

    log_dir = "logs_dqn_multiscenario"
    os.makedirs(log_dir, exist_ok=True)

    eval_env = HybridChannelEnvMulti()
    eval_env = Monitor(eval_env)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=1,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=2000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        batch_size=32,
        buffer_size=50_000,
        learning_starts=500,
        gamma=0.98,
        tau=0.01,
        target_update_interval=500,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
    )

    model.learn(
        total_timesteps=100_000,
        callback=eval_callback
    )

    model.save(os.path.join(log_dir, "dqn_multiscenario_final"))
    print("\nTreino concluído!")
    print(f"Modelos guardados em: {log_dir}")

    results_file = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(results_file):
        data = np.load(results_file)
        timesteps = data["timesteps"]
        results = data["results"].mean(axis=1)
        stds = data["results"].std(axis=1)

        plt.figure(figsize=(10,5))
        plt.plot(timesteps, results, label="Mean Reward")
        plt.fill_between(timesteps, results-stds, results+stds, alpha=0.3)
        plt.title("DQN Evaluation Reward Over Time")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "reward_curve.png"))
        print("Gráfico guardado: reward_curve.png")
    else:
        print("Aviso: evaluation.npz não encontrado.")

if __name__ == "__main__":
    main()
