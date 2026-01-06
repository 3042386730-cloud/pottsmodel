import jax
from src.config import PottsConfig
from src.simulation import SimulationManager
from src.analysis import DataAnalyzer

def main():
    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    # Configuration
    config = PottsConfig(
        L=128,
        q=4,
        n_temp=10,
        n_run=100,
        temp_min=1.800,
        temp_max=1.840,
        output_base="L128_refactored"
    )
    
    # Run Simulation
    print("Starting simulation...", flush=True)
    print(f"System size: {config.L}x{config.L}, q = {config.q}", flush=True)
    
    manager = SimulationManager(config)
    manager.run()
    
    # Run Analysis
    analyzer = DataAnalyzer(config)
    analyzer.analyze()
    
    print("\nAll done! Simulation and analysis completed successfully.", flush=True)

if __name__ == "__main__":
    main()
