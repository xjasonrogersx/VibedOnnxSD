from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigurationError


@dataclass(frozen=True)
class SchedulerConfig:
    num_train_timesteps: int
    beta_start: float
    beta_end: float
    beta_schedule: str
    prediction_type: str = "epsilon"
    clip_sample: bool = False
    set_alpha_to_one: bool = False
    trained_betas: tuple[float, ...] | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "SchedulerConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        trained_betas = payload.get("trained_betas")
        return cls(
            num_train_timesteps=int(payload["num_train_timesteps"]),
            beta_start=float(payload["beta_start"]),
            beta_end=float(payload["beta_end"]),
            beta_schedule=str(payload.get("beta_schedule", "linear")),
            prediction_type=str(payload.get("prediction_type", "epsilon")),
            clip_sample=bool(payload.get("clip_sample", False)),
            set_alpha_to_one=bool(payload.get("set_alpha_to_one", False)),
            trained_betas=tuple(float(value) for value in trained_betas)
            if trained_betas
            else None,
        )


class DDIMScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        if config.prediction_type != "epsilon":
            raise ConfigurationError(
                f"Unsupported prediction_type: {config.prediction_type}"
            )

        self.config = config
        self.betas = self._build_betas(config)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0, dtype=np.float64)
        self.final_alpha_cumprod = (
            1.0 if config.set_alpha_to_one else float(self.alphas_cumprod[0])
        )
        self.timesteps = np.array([], dtype=np.int64)

    @classmethod
    def from_config_file(cls, path: str | Path) -> "DDIMScheduler":
        return cls(SchedulerConfig.from_file(path))

    def set_timesteps(self, num_inference_steps: int) -> np.ndarray:
        if num_inference_steps <= 0:
            raise ConfigurationError("Inference steps must be positive.")
        if num_inference_steps > self.config.num_train_timesteps:
            raise ConfigurationError(
                "Inference steps cannot exceed scheduler training timesteps."
            )

        self.timesteps = np.linspace(
            0,
            self.config.num_train_timesteps - 1,
            num=num_inference_steps,
            dtype=np.int64,
        )[::-1].copy()
        return self.timesteps

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        next_timestep: int | None,
    ) -> np.ndarray:
        alpha_prod_t = float(self.alphas_cumprod[timestep])
        alpha_prod_t_prev = (
            float(self.alphas_cumprod[next_timestep])
            if next_timestep is not None
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1.0 - alpha_prod_t

        pred_original_sample = (
            sample - np.sqrt(beta_prod_t, dtype=np.float32) * model_output
        ) / np.sqrt(alpha_prod_t, dtype=np.float32)
        pred_sample_direction = np.sqrt(
            1.0 - alpha_prod_t_prev, dtype=np.float32
        ) * model_output
        prev_sample = (
            np.sqrt(alpha_prod_t_prev, dtype=np.float32) * pred_original_sample
            + pred_sample_direction
        )
        return prev_sample.astype(np.float32, copy=False)

    @staticmethod
    def _build_betas(config: SchedulerConfig) -> np.ndarray:
        if config.trained_betas is not None:
            betas = np.asarray(config.trained_betas, dtype=np.float64)
        elif config.beta_schedule == "linear":
            betas = np.linspace(
                config.beta_start,
                config.beta_end,
                config.num_train_timesteps,
                dtype=np.float64,
            )
        elif config.beta_schedule == "scaled_linear":
            betas = (
                np.linspace(
                    np.sqrt(config.beta_start),
                    np.sqrt(config.beta_end),
                    config.num_train_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        else:
            raise ConfigurationError(
                f"Unsupported beta_schedule: {config.beta_schedule}"
            )

        if np.any(betas <= 0) or np.any(betas >= 1):
            raise ConfigurationError("Scheduler betas must be within the open interval (0, 1).")
        return betas
