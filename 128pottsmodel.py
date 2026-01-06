import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import numpy as onp
import os
import time
import math
import sys

# =============== 可配置变量 ===============
# 系统尺寸 (论文标准尺寸)
L = 128
# Potts模型的状态数 q (q=2 为 Ising 模型)
q = 4
# 温度数量
n_temp = 10
# 每个温度的独立运行次数
n_run = 100
# 温度范围 (覆盖临界点)
temp_min = 1.800
temp_max = 1.840
# 耦合常数 J
J = 1.0
# 玻尔兹曼常数
k_B = 1.0
# 热化步数 (10% of total)
thermalization_steps = 5000
# 正式测量步数 (production steps)
production_steps = 50000
# 构型保存间隔
save_interval = 1000
# 随机数种子
initial_seed = 42
# 输出文件夹
output_base = f"L{L}"
# =============== 结束配置 ===============

print("Starting simulation...", flush=True)
print(f"System size: {L}x{L}, q = {q}", flush=True)
print(f"Temperature range: {temp_min} to {temp_max} ({n_temp} points)", flush=True)
print(f"Thermalization steps: {thermalization_steps}", flush=True)
print(f"Production steps: {production_steps}", flush=True)
print(f"Runs per temperature: {n_run}", flush=True)

# 设置JAX使用64位精度
jax.config.update("jax_enable_x64", True)

# 生成温度数组 (更密集地覆盖临界区域)
temperatures = jnp.linspace(temp_min, temp_max, n_temp)
betas = 1.0 / (k_B * temperatures)

# 计算批处理大小 (根据GPU内存自动调整)
total_tasks = n_temp * n_run
# 估算单个任务内存占用
mem_per_task = L * L * 8 + 1024  # 保守估计
# 假设可用8GB GPU内存
gpu_mem = 8 * 1024**3
batch_size = min(total_tasks, max(1, int(gpu_mem / mem_per_task)))
n_batches = math.ceil(total_tasks / batch_size)
print(f"Auto-selected batch_size: {batch_size}, Total batches: {n_batches}", flush=True)

# 创建输出文件夹
os.makedirs(output_base, exist_ok=True)
print(f"Output directory: {output_base}", flush=True)

# 为每个温度创建子文件夹
for temp in temperatures:
    temp_dir = os.path.join(output_base, f"temp_{temp:.4f}")
    os.makedirs(os.path.join(temp_dir, "configs"), exist_ok=True)

# 定义Wolff更新函数 (使用闭包捕获静态参数)
def make_wolff_update(q_val, L_val, J_val):
    """创建Wolff更新函数，使用闭包捕获静态参数"""
    
    def wolff_update_single(state, beta, key):
        # 计算实际添加概率 p_add = 1 - exp(-2*beta*J)
        p_add = 1.0 - jnp.exp(-2.0 * beta * J_val)

        # 选择随机种子位置 - 修正: 显式指定dtype=int32
        key, subkey = random.split(key)
        i0 = random.randint(subkey, (), 0, L_val, dtype=jnp.int32)
        key, subkey = random.split(key)
        j0 = random.randint(subkey, (), 0, L_val, dtype=jnp.int32)
        s_old = state[i0, j0]

        # 选择新自旋状态 (排除s_old)
        key, subkey = random.split(key)
        rand_val = random.randint(subkey, (), 0, q_val - 1, dtype=jnp.int32)
        s_new = jnp.where(rand_val >= s_old, rand_val + 1, rand_val)

        # 初始化数据结构
        visited = jnp.zeros((L_val, L_val), dtype=jnp.bool_)
        max_size = L_val * L_val
        queue = jnp.full((max_size, 2), -1, dtype=jnp.int32)
        queue = queue.at[0].set(jnp.array([i0, j0], dtype=jnp.int32))
        visited = visited.at[i0, j0].set(True)
        size = 1
        head = 0

        # Wolff簇生长循环
        def cond_fn(carry):
            _, _, size, head, _ = carry
            return head < size

        def body_fn(carry):
            queue, visited, size, head, key = carry
            i, j = queue[head]

            # 计算邻居位置 (周期性边界条件) - 修正: 显式指定dtype=int32
            neighbors = jnp.array([
                [(i - 1) % L_val, j],
                [(i + 1) % L_val, j],
                [i, (j - 1) % L_val],
                [i, (j + 1) % L_val]
            ], dtype=jnp.int32)

            # 获取邻居自旋
            spins = state[neighbors[:, 0], neighbors[:, 1]]
            spin_match = (spins == s_old)
            not_visited = ~visited[neighbors[:, 0], neighbors[:, 1]]

            # 合并条件
            conditions = spin_match & not_visited

            # 生成随机数决定是否加入
            key, subkey = random.split(key)
            rand_vals = random.uniform(subkey, shape=(4,))
            to_add = conditions & (rand_vals < p_add)

            # 更新队列和visited
            def add_neighbor(carry, idx):
                queue, visited, size, key = carry
                should_add = to_add[idx]
                coord = neighbors[idx]

                queue = lax.cond(
                    should_add,
                    lambda q, s, c: q.at[s].set(c),
                    lambda q, s, c: q,
                    queue, size, coord
                )
                visited = lax.cond(
                    should_add,
                    lambda v, c: v.at[c[0], c[1]].set(True),
                    lambda v, c: v,
                    visited, coord
                )
                size = lax.cond(
                    should_add,
                    lambda s: s + 1,
                    lambda s: s,
                    size
                )
                return (queue, visited, size, key), None

            (queue, visited, size, key), _ = lax.scan(
                add_neighbor,
                (queue, visited, size, key),
                jnp.arange(4)
            )

            head += 1
            return queue, visited, size, head, key

        queue, visited, size, head, key = lax.while_loop(
            cond_fn,
            body_fn,
            (queue, visited, size, head, key)
        )

        # 翻转簇内自旋 (所有被访问的格点)
        new_state = jnp.where(visited, s_new, state)
        return new_state, key

    # JIT编译单个更新函数
    wolff_update_single_jit = jit(wolff_update_single)

    # 创建批处理版本
    @jit
    def batch_wolff_update(states, betas, keys):
        @vmap
        def update(state, beta, key):
            return wolff_update_single_jit(state, beta, key)
        
        return update(states, betas, keys)
    
    return batch_wolff_update


# 创建可观测量计算函数 (使用闭包捕获静态参数)
def make_calculate_observables(L_val, q_val, J_val):
    """创建可观测量计算函数，使用闭包捕获静态参数"""

    @jit
    def calculate_observables(state):
        # 能量计算 (最近邻相互作用，周期性边界条件)
        right = jnp.roll(state, -1, axis=1)
        down = jnp.roll(state, -1, axis=0)

        # 计算最近邻对的匹配
        horizontal_match = (state == right).astype(jnp.float32)
        vertical_match = (state == down).astype(jnp.float32)

        # 总匹配数
        total_matches = jnp.sum(horizontal_match) + jnp.sum(vertical_match)

        # 能量 (E = -J * 匹配数)
        energy = -J_val * total_matches

        # Potts模型序参量计算 (归一化到[0,1])
        # 统计每种状态的数量
        counts = jnp.zeros(q_val)
        for s in range(q_val):
            counts = counts.at[s].set(jnp.sum(state == s))
        
        N = L_val * L_val
        max_count = jnp.max(counts)
        
        # 归一化序参量: m = (q*max_count/N - 1)/(q-1)
        # 当q=2时等价于|<s>|
        magnetization = jnp.where(
            q_val > 1,
            (q_val * max_count / N - 1) / (q_val - 1),
            0.0  # q=1是平凡情况
        )
        
        # 确保磁化强度在[0,1]范围内
        magnetization = jnp.clip(magnetization, 0.0, 1.0)

        # 计算 Ising 带符号磁化强度（如果 q=2）
        signed_m = jnp.zeros(1)
        if q_val == 2:
            ising_spins = jnp.where(state == 0, -1, 1)
            signed_m = jnp.sum(ising_spins) / N
        
        return energy, magnetization, signed_m

    # 批量版本
    @jit
    def batch_calculate_observables(states):
        @vmap
        def calc(state):
            return calculate_observables(state)
        
        return calc(states)
    
    return batch_calculate_observables

# === 初始化数据结构 ===
full_key = random.PRNGKey(initial_seed)
key, subkey = random.split(full_key)
states = random.randint(subkey, (n_temp, n_run, L, L), 0, q, dtype=jnp.int32)  # 修正: 显式指定dtype=int32

# 生成每个任务的随机数种子
keys = random.split(key, n_temp * n_run).reshape(n_temp, n_run, 2)

# 准备任务索引
task_indices = jnp.arange(total_tasks)

# 存储磁化强度轨迹
magnetization_trajectories = onp.zeros((n_temp, n_run, production_steps), dtype=onp.float64)
signed_magnetization_trajectories = onp.zeros((n_temp, n_run, production_steps), dtype=onp.float64)

# 创建Wolff更新器和可观测量计算器
wolff_updater = make_wolff_update(q, L, J)
observables_calculator = make_calculate_observables(L, q, J)

start_time = time.time()
print("\n=== Starting thermalization phase ===", flush=True)

# 按批次处理任务
for b in range(n_batches):
    batch_start = b * batch_size
    batch_end = min((b + 1) * batch_size, total_tasks)
    batch_size_actual = batch_end - batch_start
    
    print(f"\nProcessing batch {b + 1}/{n_batches} (size={batch_size_actual})", flush=True)
    batch_indices = task_indices[batch_start:batch_end]

    # 映射到温度和运行索引
    temp_indices = batch_indices // n_run
    run_indices = batch_indices % n_run

    # 获取当前批次数据
    batch_states = jnp.array(states[temp_indices, run_indices])
    batch_betas = betas[temp_indices]
    batch_keys = jnp.array(keys[temp_indices, run_indices])

    # === 热化阶段 ===
    print(f"  Batch {b + 1}: Starting thermalization ({thermalization_steps} steps)...", flush=True)
    last_time = time.time()
    
    for step in range(thermalization_steps):
        if step % 100 == 0 and step > 0:
            current_time = time.time()
            elapsed = current_time - last_time
            estimated_remaining = elapsed * (thermalization_steps - step) / 100
            print(f"  Batch {b + 1}: Thermalization step {step}/{thermalization_steps} "
                  f"(Last 100 steps: {elapsed:.2f}s, Est. remaining: {estimated_remaining/60:.1f} min)", flush=True)
            last_time = current_time
        
        # 执行Wolff更新
        batch_states, batch_keys = wolff_updater(
            batch_states, batch_betas, batch_keys
        )
    
    print(f"  Batch {b + 1}: Thermalization completed", flush=True)
    
    # === 生产阶段 (正式测量) ===
    print(f"  Batch {b + 1}: Starting production phase ({production_steps} steps)...", flush=True)
    last_time = time.time()
    
    # 为当前批次创建临时存储
    batch_mags = onp.zeros((batch_size_actual, production_steps), dtype=onp.float64)
    batch_signed_mags = onp.zeros((batch_size_actual, production_steps), dtype=onp.float64)
    
    for step in range(production_steps):
        if step % 100 == 0 and step > 0:
            current_time = time.time()
            elapsed = current_time - last_time
            estimated_remaining = elapsed * (production_steps - step) / 100
            print(f"  Batch {b + 1}: Production step {step}/{production_steps} "
                  f"(Last 100 steps: {elapsed:.2f}s, Est. remaining: {estimated_remaining/60:.1f} min)", flush=True)
            last_time = current_time
        
        # 执行Wolff更新
        batch_states, batch_keys = wolff_updater(
            batch_states, batch_betas, batch_keys
        )
        
        # 计算两种可观测量：标准磁化强度和带符号磁化强度
        _, magnetizations, signed_mags = observables_calculator(batch_states)
        
        # 存储两种磁化强度
        batch_mags[:, step] = onp.array(magnetizations)
        if q == 2:
            batch_signed_mags[:, step] = onp.array(signed_mags)
        
        # 每save_interval步保存一次构型
        if step % save_interval == 0:
            # 保存构型
            for idx in range(batch_size_actual):
                i_temp = int(temp_indices[idx])
                j_run = int(run_indices[idx])
                temp_val = float(temperatures[i_temp])
                config = onp.array(batch_states[idx])
                
                # 保存构型
                filename = os.path.join(
                    output_base, 
                    f"temp_{temp_val:.4f}", 
                    "configs", 
                    f"config_step_{step}_run_{j_run}.txt"
                )
                onp.savetxt(filename, config, fmt="%d")
    
    # 将当前批次的数据复制到全局数组
    for idx in range(batch_size_actual):
        i_temp = int(temp_indices[idx])
        j_run = int(run_indices[idx])
        magnetization_trajectories[i_temp, j_run] = batch_mags[idx]
        if q == 2:
            signed_magnetization_trajectories[i_temp, j_run] = batch_signed_mags[idx]
    
    # 保存当前批次的最终状态
    states = states.at[temp_indices, run_indices].set(onp.array(batch_states))
    keys = keys.at[temp_indices, run_indices].set(onp.array(batch_keys))
    
    # 保存每个运行的磁化强度轨迹
    print(f"  Batch {b + 1}: Saving magnetization trajectories...", flush=True)
    for idx in range(batch_size_actual):
        i_temp = int(temp_indices[idx])
        j_run = int(run_indices[idx])
        temp_val = float(temperatures[i_temp])
        
        # 保存标准磁化强度轨迹
        mag_file = os.path.join(
            output_base, 
            f"temp_{temp_val:.4f}", 
            f"magnetization_run_{j_run}.txt"
        )
        onp.savetxt(mag_file, magnetization_trajectories[i_temp, j_run], fmt="%.6f")
        
        # 保存带符号磁化强度轨迹 (仅 q=2)
        if q == 2:
            signed_mag_file = os.path.join(
                output_base, 
                f"temp_{temp_val:.4f}", 
                f"signed_magnetization_run_{j_run}.txt"
            )
            onp.savetxt(signed_mag_file, signed_magnetization_trajectories[i_temp, j_run], fmt="%.6f")
    
    print(f"  Batch {b + 1}: Production phase completed", flush=True)

total_time = time.time() - start_time
print(f"\nSimulation completed in {total_time/3600:.2f} hours", flush=True)

# === 计算并保存 Binder 累积量 ===
print("\n=== Calculating Binder cumulants ===", flush=True)

binder_results = []
error_results = []

# Binning分析函数
def calculate_binder_with_error(mags, num_bins=32):
    """使用binning分析计算Binder累积量及其误差"""
    n_samples = len(mags)
    if n_samples < num_bins:
        num_bins = n_samples // 2
    
    bin_size = n_samples // num_bins
    
    # 重塑数据成bins
    binned_data = mags[:num_bins*bin_size].reshape(num_bins, bin_size)
    
    # 计算每个bin的Binder累积量
    bin_binders = []
    for bin_data in binned_data:
        M2 = onp.mean(bin_data**2)
        M4 = onp.mean(bin_data**4)
        binder = 1 - M4/(3*M2**2) if M2 > 0 else 0.0
        bin_binders.append(binder)
    
    # 计算平均值和标准误差
    binder_mean = onp.mean(bin_binders)
    binder_std = onp.std(bin_binders, ddof=1) if len(bin_binders) > 1 else 0.0
    binder_error = binder_std / onp.sqrt(num_bins) if num_bins > 1 else binder_std
    
    return binder_mean, binder_error

# 为每个温度计算标准Binder累积量 (U4)
for i_temp in range(n_temp):
    temp_val = float(temperatures[i_temp])
    print(f"Processing temperature for standard Binder: {temp_val:.4f}", flush=True)
    
    # 收集该温度下所有运行的磁化强度数据
    all_mags = []
    for j_run in range(n_run):
        # 读取磁化强度轨迹
        mag_file = os.path.join(
            output_base, 
            f"temp_{temp_val:.4f}", 
            f"magnetization_run_{j_run}.txt"
        )
        if os.path.exists(mag_file):
            mags = onp.loadtxt(mag_file)
            all_mags.append(mags)
    
    if len(all_mags) == 0:
        print(f"  Warning: No data found for temperature {temp_val:.4f}", flush=True)
        continue
    
    # 将所有运行的数据合并
    all_mags = onp.concatenate(all_mags)
    
    # 计算Binder累积量
    binder_mean, binder_error = calculate_binder_with_error(all_mags)
    
    binder_results.append((temp_val, binder_mean))
    error_results.append((temp_val, binder_error))
    
    print(f"  Temperature {temp_val:.4f}: Standard Binder U4 = {binder_mean:.6f} ± {binder_error:.6f}", flush=True)

# 保存标准Binder累积量结果
binder_file = os.path.join(output_base, "binder_cumulants.txt")
with open(binder_file, 'w') as f:
    f.write("Temperature BinderCumulant Error\n")
    for (temp, binder), (_, error) in zip(binder_results, error_results):
        f.write(f"{temp:.6f} {binder:.6f} {error:.6f}\n")

print(f"\nStandard Binder cumulants saved to: {binder_file}", flush=True)

# === 计算论文定义的 Binder U2 ===
# 仅适用于 q=2 (Ising 模型)
if q != 2:
    print("\nNote: Paper's U2 = <m^2>/<|m|>^2 is only defined for q=2 (Ising model). Skipping U2 calculation.")
else:
    print("\n=== Calculating Binder U2 (paper method: U2 = <m^2>/<|m|>^2) ===", flush=True)
    
    binder_U2_results = []
    error_U2_results = []

    def calculate_U2_with_error(signed_mags, num_bins=78):  # 论文使用至少78 bins
        """按论文方法计算 U2 及误差（Binning分析）"""
        n_samples = len(signed_mags)
        if n_samples < num_bins:
            num_bins = max(2, n_samples // 2)
        
        bin_size = n_samples // num_bins
        binned_data = signed_mags[:num_bins*bin_size].reshape(num_bins, bin_size)
        
        U2_vals = []
        for bin_data in binned_data:
            m2 = onp.mean(bin_data**2)
            abs_m = onp.mean(onp.abs(bin_data))
            U2 = m2 / (abs_m**2) if abs_m > 1e-10 else 0.0
            U2_vals.append(U2)
        
        U2_mean = onp.mean(U2_vals)
        U2_std = onp.std(U2_vals, ddof=1) if len(U2_vals) > 1 else 0.0
        U2_error = U2_std / onp.sqrt(num_bins) if num_bins > 1 else U2_std
        return U2_mean, U2_error

    for i_temp in range(n_temp):
        temp_val = float(temperatures[i_temp])
        print(f"Processing temperature for paper-style Binder U2: {temp_val:.4f}", flush=True)
        
        # 收集该温度下所有运行的带符号磁化强度数据
        all_signed_mags = []
        for j_run in range(n_run):
            signed_mag_file = os.path.join(
                output_base, 
                f"temp_{temp_val:.4f}", 
                f"signed_magnetization_run_{j_run}.txt"
            )
            if os.path.exists(signed_mag_file):
                signed_mags = onp.loadtxt(signed_mag_file)
                all_signed_mags.append(signed_mags)
        
        if len(all_signed_mags) == 0:
            print(f"  Warning: No signed mag data found for temperature {temp_val:.4f}", flush=True)
            continue
        
        # 将所有运行的数据合并
        all_signed_mags = onp.concatenate(all_signed_mags)
        
        # 计算 U2
        U2_mean, U2_error = calculate_U2_with_error(all_signed_mags, num_bins=78)
        binder_U2_results.append((temp_val, U2_mean))
        error_U2_results.append((temp_val, U2_error))
        
        print(f"  Temperature {temp_val:.4f}: Paper-style Binder U2 = {U2_mean:.6f} ± {U2_error:.6f}", flush=True)

    # 保存 U2 结果
    U2_file = os.path.join(output_base, "binder_U2.txt")
    with open(U2_file, 'w') as f:
        f.write("Temperature BinderU2 Error\n")
        for (temp, U2), (_, err) in zip(binder_U2_results, error_U2_results):
            f.write(f"{temp:.6f} {U2:.6f} {err:.6f}\n")
    
    print(f"\nPaper-style Binder U2 saved to: {U2_file}", flush=True)

# 保存最终统计摘要
summary_file = os.path.join(output_base, "simulation_summary.txt")
with open(summary_file, 'w') as f:
    f.write("=== 2D Potts Model Simulation Summary ===\n\n")
    f.write(f"System size: {L}x{L}\n")
    f.write(f"State number (q): {q}\n")
    f.write(f"Temperature range: {temp_min} to {temp_max}\n")
    f.write(f"Number of temperatures: {n_temp}\n")
    f.write(f"Runs per temperature: {n_run}\n")
    f.write(f"Thermalization steps: {thermalization_steps}\n")
    f.write(f"Production steps: {production_steps}\n")
    f.write(f"Configuration save interval: {save_interval}\n")
    f.write(f"Total simulation time: {total_time/3600:.2f} hours\n")
    f.write(f"Total measurements per run: {production_steps}\n")
    f.write(f"Total measurements overall: {n_temp * n_run * production_steps:,}\n")
    f.write("\nOutput files:\n")
    f.write(f"  {output_base}/binder_cumulants.txt (Standard definition: U4 = 1 - <M^4>/(3<M^2>^2))\n")
    if q == 2:
        f.write(f"  {output_base}/binder_U2.txt (Paper definition: U2 = <m^2>/<|m|>^2)\n")
    f.write(f"  {output_base}/simulation_summary.txt (This file)\n")

print(f"\nSimulation summary saved to: {summary_file}", flush=True)
print("\nAll done! Simulation completed successfully.", flush=True)