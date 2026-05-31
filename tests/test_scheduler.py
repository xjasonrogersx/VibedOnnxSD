from __future__ import annotations

import unittest

import numpy as np

from vibed_onnx_sd.scheduler import DDIMScheduler, SchedulerConfig


class SchedulerTests(unittest.TestCase):
    def test_set_timesteps_descends(self) -> None:
        scheduler = DDIMScheduler(
            SchedulerConfig(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
            )
        )

        timesteps = scheduler.set_timesteps(5)

        self.assertEqual(timesteps.tolist(), [999, 749, 499, 249, 0])

    def test_step_preserves_shape(self) -> None:
        scheduler = DDIMScheduler(
            SchedulerConfig(
                num_train_timesteps=10,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="linear",
            )
        )
        timesteps = scheduler.set_timesteps(2)
        latents = np.ones((1, 4, 2, 2), dtype=np.float32)
        noise = np.full((1, 4, 2, 2), 0.5, dtype=np.float32)

        updated = scheduler.step(
            model_output=noise,
            timestep=int(timesteps[0]),
            sample=latents,
            next_timestep=int(timesteps[1]),
        )

        self.assertEqual(updated.shape, latents.shape)
        self.assertEqual(updated.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
