import os
import time
import math
import jax.numpy as jnp
from jax import random
import numpy as onp
from .config import PottsConfig
from .model import PottsModel

class SimulationManager:
    """Manages the simulation execution, batching, and file I/O."""
    
    def __init__(self, config: PottsConfig):
        self.config = config
        self.model = PottsModel(config)
        self.output_base = config.output_base
        self._setup_directories()
        self._setup_initial_state()

    def _setup_directories(self):
        os.makedirs(self.output_base, exist_ok=True)
        print(f"Output directory: {self.output_base}", flush=True)
        for temp in self.config.temperatures:
            temp_dir = os.path.join(self.output_base, f"temp_{temp:.4f}")
            os.makedirs(os.path.join(temp_dir, "configs"), exist_ok=True)

    def _setup_initial_state(self):
        full_key = random.PRNGKey(self.config.initial_seed)
        key, subkey = random.split(full_key)
        self.states = random.randint(
            subkey, 
            (self.config.n_temp, self.config.n_run, self.config.L, self.config.L), 
            0, self.config.q, dtype=jnp.int32
        )
        self.keys = random.split(key, self.config.total_tasks).reshape(self.config.n_temp, self.config.n_run, 2)
        
        # Calculate batch size
        mem_per_task = self.config.L * self.config.L * 8 + 1024
        gpu_mem = 8 * 1024**3
        self.batch_size = min(self.config.total_tasks, max(1, int(gpu_mem / mem_per_task)))
        self.n_batches = math.ceil(self.config.total_tasks / self.batch_size)
        print(f"Auto-selected batch_size: {self.batch_size}, Total batches: {self.n_batches}", flush=True)

        self.magnetization_trajectories = onp.zeros((self.config.n_temp, self.config.n_run, self.config.production_steps), dtype=onp.float64)
        self.signed_magnetization_trajectories = onp.zeros((self.config.n_temp, self.config.n_run, self.config.production_steps), dtype=onp.float64)

    def run(self):
        start_time = time.time()
        task_indices = jnp.arange(self.config.total_tasks)
        
        for b in range(self.n_batches):
            self._process_batch(b, task_indices, start_time)
            
        total_time = time.time() - start_time
        print(f"\nSimulation completed in {total_time/3600:.2f} hours", flush=True)
        self._save_summary(total_time)

    def _process_batch(self, b, task_indices, start_time):
        batch_start = b * self.batch_size
        batch_end = min((b + 1) * self.batch_size, self.config.total_tasks)
        batch_size_actual = batch_end - batch_start
        
        print(f"\nProcessing batch {b + 1}/{self.n_batches} (size={batch_size_actual})", flush=True)
        
        batch_indices = task_indices[batch_start:batch_end]
        temp_indices = batch_indices // self.config.n_run
        run_indices = batch_indices % self.config.n_run
        
        batch_states = jnp.array(self.states[temp_indices, run_indices])
        batch_betas = self.config.betas[temp_indices]
        batch_keys = jnp.array(self.keys[temp_indices, run_indices])
        
        # Thermalization
        print(f"  Batch {b + 1}: Starting thermalization ({self.config.thermalization_steps} steps)...", flush=True)
        batch_states, batch_keys = self._run_phase(
            batch_states, batch_betas, batch_keys, 
            self.config.thermalization_steps, b, is_production=False
        )
        print(f"  Batch {b + 1}: Thermalization completed", flush=True)
        
        # Production
        print(f"  Batch {b + 1}: Starting production phase ({self.config.production_steps} steps)...", flush=True)
        batch_states, batch_keys, batch_mags, batch_signed_mags = self._run_phase(
            batch_states, batch_betas, batch_keys, 
            self.config.production_steps, b, is_production=True,
            temp_indices=temp_indices, run_indices=run_indices
        )
        
        # Store results
        for idx in range(batch_size_actual):
            i_temp = int(temp_indices[idx])
            j_run = int(run_indices[idx])
            self.magnetization_trajectories[i_temp, j_run] = batch_mags[idx]
            if self.config.q == 2:
                self.signed_magnetization_trajectories[i_temp, j_run] = batch_signed_mags[idx]
        
        # Update global state
        self.states = self.states.at[temp_indices, run_indices].set(onp.array(batch_states))
        self.keys = self.keys.at[temp_indices, run_indices].set(onp.array(batch_keys))
        
        self._save_batch_trajectories(batch_size_actual, temp_indices, run_indices)
        print(f"  Batch {b + 1}: Production phase completed", flush=True)

    def _run_phase(self, states, betas, keys, steps, batch_id, is_production=False, temp_indices=None, run_indices=None):
        batch_size = states.shape[0]
        mags = None
        signed_mags = None
        
        if is_production:
            mags = onp.zeros((batch_size, steps), dtype=onp.float64)
            signed_mags = onp.zeros((batch_size, steps), dtype=onp.float64)
        
        last_time = time.time()
        
        for step in range(steps):
            if step % 100 == 0 and step > 0:
                current_time = time.time()
                elapsed = current_time - last_time
                estimated_remaining = elapsed * (steps - step) / 100
                phase_name = "Production" if is_production else "Thermalization"
                print(f"  Batch {batch_id + 1}: {phase_name} step {step}/{steps} "
                      f"(Last 100 steps: {elapsed:.2f}s, Est. remaining: {estimated_remaining/60:.1f} min)", flush=True)
                last_time = current_time
            
            states, keys = self.model.wolff_update(states, betas, keys)
            
            if is_production:
                _, magnetizations, s_mags = self.model.calculate_observables(states)
                mags[:, step] = onp.array(magnetizations)
                if self.config.q == 2:
                    signed_mags[:, step] = onp.array(s_mags)
                
                if step % self.config.save_interval == 0:
                    self._save_configs(states, temp_indices, run_indices, step)
                    
        if is_production:
            return states, keys, mags, signed_mags
        return states, keys

    def _save_configs(self, states, temp_indices, run_indices, step):
        for idx in range(len(states)):
            i_temp = int(temp_indices[idx])
            j_run = int(run_indices[idx])
            temp_val = float(self.config.temperatures[i_temp])
            filename = os.path.join(
                self.output_base, f"temp_{temp_val:.4f}", "configs", 
                f"config_step_{step}_run_{j_run}.txt"
            )
            onp.savetxt(filename, onp.array(states[idx]), fmt="%d")

    def _save_batch_trajectories(self, batch_size, temp_indices, run_indices):
        for idx in range(batch_size):
            i_temp = int(temp_indices[idx])
            j_run = int(run_indices[idx])
            temp_val = float(self.config.temperatures[i_temp])
            
            mag_file = os.path.join(self.output_base, f"temp_{temp_val:.4f}", f"magnetization_run_{j_run}.txt")
            onp.savetxt(mag_file, self.magnetization_trajectories[i_temp, j_run], fmt="%.6f")
            
            if self.config.q == 2:
                signed_mag_file = os.path.join(self.output_base, f"temp_{temp_val:.4f}", f"signed_magnetization_run_{j_run}.txt")
                onp.savetxt(signed_mag_file, self.signed_magnetization_trajectories[i_temp, j_run], fmt="%.6f")

    def _save_summary(self, total_time):
        summary_file = os.path.join(self.output_base, "simulation_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== 2D Potts Model Simulation Summary ===\n\n")
            f.write(f"System size: {self.config.L}x{self.config.L}\n")
            f.write(f"State number (q): {self.config.q}\n")
            f.write(f"Temperature range: {self.config.temp_min} to {self.config.temp_max}\n")
            f.write(f"Number of temperatures: {self.config.n_temp}\n")
            f.write(f"Runs per temperature: {self.config.n_run}\n")
            f.write(f"Thermalization steps: {self.config.thermalization_steps}\n")
            f.write(f"Production steps: {self.config.production_steps}\n")
            f.write(f"Configuration save interval: {self.config.save_interval}\n")
            f.write(f"Total simulation time: {total_time/3600:.2f} hours\n")
            f.write(f"Total measurements per run: {self.config.production_steps}\n")
            f.write(f"Total measurements overall: {self.config.n_temp * self.config.n_run * self.config.production_steps:,}\n")
            f.write("\nOutput files:\n")
            f.write(f"  {self.output_base}/binder_cumulants.txt (Standard definition: U4 = 1 - <M^4>/(3<M^2>^2))\n")
            if self.config.q == 2:
                f.write(f"  {self.output_base}/binder_U2.txt (Paper definition: U2 = <m^2>/<|m|>^2)\n")
            f.write(f"  {self.output_base}/simulation_summary.txt (This file)\n")
        print(f"\nSimulation summary saved to: {summary_file}", flush=True)
