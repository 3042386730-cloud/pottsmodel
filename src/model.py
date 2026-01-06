import jax.numpy as jnp
from jax import random, vmap, jit, lax
from .config import PottsConfig

class PottsModel:
    """Handles the Physics and JAX-based updates for the Potts Model."""
    
    def __init__(self, config: PottsConfig):
        self.config = config
        self.wolff_update = self._build_wolff_update()
        self.calculate_observables = self._build_observables_calculator()

    def _build_wolff_update(self):
        L_val = self.config.L
        q_val = self.config.q
        J_val = self.config.J

        def wolff_update_single(state, beta, key):
            p_add = 1.0 - jnp.exp(-2.0 * beta * J_val)
            
            key, subkey = random.split(key)
            i0 = random.randint(subkey, (), 0, L_val, dtype=jnp.int32)
            key, subkey = random.split(key)
            j0 = random.randint(subkey, (), 0, L_val, dtype=jnp.int32)
            s_old = state[i0, j0]
            
            key, subkey = random.split(key)
            rand_val = random.randint(subkey, (), 0, q_val - 1, dtype=jnp.int32)
            s_new = jnp.where(rand_val >= s_old, rand_val + 1, rand_val)
            
            visited = jnp.zeros((L_val, L_val), dtype=jnp.bool_)
            max_size = L_val * L_val
            queue = jnp.full((max_size, 2), -1, dtype=jnp.int32)
            queue = queue.at[0].set(jnp.array([i0, j0], dtype=jnp.int32))
            visited = visited.at[i0, j0].set(True)
            size = 1
            head = 0

            def cond_fn(carry):
                _, _, size, head, _ = carry
                return head < size

            def body_fn(carry):
                queue, visited, size, head, key = carry
                i, j = queue[head]
                
                neighbors = jnp.array([
                    [(i - 1) % L_val, j],
                    [(i + 1) % L_val, j],
                    [i, (j - 1) % L_val],
                    [i, (j + 1) % L_val]
                ], dtype=jnp.int32)
                
                spins = state[neighbors[:, 0], neighbors[:, 1]]
                spin_match = (spins == s_old)
                not_visited = ~visited[neighbors[:, 0], neighbors[:, 1]]
                conditions = spin_match & not_visited
                
                key, subkey = random.split(key)
                rand_vals = random.uniform(subkey, shape=(4,))
                to_add = conditions & (rand_vals < p_add)
                
                def add_neighbor(carry, idx):
                    queue, visited, size, key = carry
                    should_add = to_add[idx]
                    coord = neighbors[idx]
                    
                    queue = lax.cond(should_add, lambda q, s, c: q.at[s].set(c), lambda q, s, c: q, queue, size, coord)
                    visited = lax.cond(should_add, lambda v, c: v.at[c[0], c[1]].set(True), lambda v, c: v, visited, coord)
                    size = lax.cond(should_add, lambda s: s + 1, lambda s: s, size)
                    return (queue, visited, size, key), None

                (queue, visited, size, key), _ = lax.scan(add_neighbor, (queue, visited, size, key), jnp.arange(4))
                head += 1
                return queue, visited, size, head, key

            queue, visited, size, head, key = lax.while_loop(cond_fn, body_fn, (queue, visited, size, head, key))
            new_state = jnp.where(visited, s_new, state)
            return new_state, key

        wolff_update_single_jit = jit(wolff_update_single)

        @jit
        def batch_wolff_update(states, betas, keys):
            return vmap(wolff_update_single_jit)(states, betas, keys)
        
        return batch_wolff_update

    def _build_observables_calculator(self):
        L_val = self.config.L
        q_val = self.config.q
        J_val = self.config.J

        @jit
        def calculate_observables(state):
            right = jnp.roll(state, -1, axis=1)
            down = jnp.roll(state, -1, axis=0)
            
            horizontal_match = (state == right).astype(jnp.float32)
            vertical_match = (state == down).astype(jnp.float32)
            total_matches = jnp.sum(horizontal_match) + jnp.sum(vertical_match)
            
            energy = -J_val * total_matches
            
            counts = jnp.zeros(q_val)
            for s in range(q_val):
                counts = counts.at[s].set(jnp.sum(state == s))
            
            N = L_val * L_val
            max_count = jnp.max(counts)
            magnetization = jnp.where(q_val > 1, (q_val * max_count / N - 1) / (q_val - 1), 0.0)
            magnetization = jnp.clip(magnetization, 0.0, 1.0)
            
            signed_m = jnp.zeros(1)
            if q_val == 2:
                ising_spins = jnp.where(state == 0, -1, 1)
                signed_m = jnp.sum(ising_spins) / N
            
            return energy, magnetization, signed_m

        @jit
        def batch_calculate_observables(states):
            return vmap(calculate_observables)(states)
        
        return batch_calculate_observables
