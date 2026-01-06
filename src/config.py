import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class PottsConfig:
    """Configuration for the Potts Model Simulation."""
    L: int = 128
    q: int = 4
    n_temp: int = 10
    n_run: int = 100
    temp_min: float = 1.800
    temp_max: float = 1.840
    J: float = 1.0
    k_B: float = 1.0
    thermalization_steps: int = 5000
    production_steps: int = 50000
    save_interval: int = 1000
    initial_seed: int = 42
    output_base: str = "L128_refactored"

    @property
    def temperatures(self):
        return jnp.linspace(self.temp_min, self.temp_max, self.n_temp)

    @property
    def betas(self):
        return 1.0 / (self.k_B * self.temperatures)

    @property
    def total_tasks(self):
        return self.n_temp * self.n_run
