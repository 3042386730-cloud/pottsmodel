import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import numpy as onp
import os
import time
import math

import shutil

# =============== 可配置变量 ===============
# 温度数量 (并行任务数)
n_temp = 100
# 每个温度的独立运行次数
n_run = 8
# 系统尺寸 L x L
L = 128
# Potts模型的状态数 q
q = 2
# w温度范围 (最小温度, 最大温度)
temp_range = (2.15, 2.35)
# 耦合常数 J
J = 1.0
# 玻尔兹曼常数 (通常设为1)
k_B = 1.0
# Wolff更新前的预热步数
warmup_steps = 20000
# 批处理大小 (None表示自动选择)
batch_size = None
# 随机数种子
initial_seed = 42
# 输出文件夹 (相对路径)
output_folder = f"Q={q}"
# =============== 结束配置 ===============

print("Starting simulation...", flush=True)

# 自动生成温度数组
temperatures = jnp.linspace(temp_range[0], temp_range[1], n_temp)
betas = 1.0 / (k_B * temperatures)

# 根据A100 40G规格自动选择batch size
if batch_size is None:
    # 估算单个任务的内存占用 (单位: 字节)
    mem_per_task = (L * L * 4) + (L * L * 2 * 4) + 1024  # 保守估计
    # A100 40G GPU内存 (字节)
    gpu_mem = 40 * 1024 ** 3
    # 预留50%内存用于其他操作
    usable_mem = gpu_mem * 0.5
    # 计算最大batch size
    max_batch = int(usable_mem // mem_per_task)
    batch_size = min(n_temp * n_run, max(1, max_batch))
    print(f"Auto-selected batch_size: {batch_size} (based on A100 40G memory)", flush=True)
else:
    print(f"Using user-defined batch_size: {batch_size}", flush=True)

total_tasks = n_temp * n_run
n_batches = math.ceil(total_tasks / batch_size)
print(f"Total tasks: {total_tasks}, Batches: {n_batches}", flush=True)


# 定义Wolff更新函数 (使用闭包捕获静态参数)
def make_wolff_update(q_val, L_val, J_val):
    """创建Wolff更新函数，使用闭包捕获静态参数"""
    p_add_base = 1.0 - jnp.exp(-2.0 * J_val)  # 基础概率

    def wolff_update_single(state, beta, key):
        # 计算实际添加概率 p_add = 1 - exp(-2*beta*J)
        p_add = 1.0 - jnp.exp(-2.0 * beta * J_val)

        # 选择随机种子位置
        key, subkey = random.split(key)
        i0 = random.randint(subkey, (), 0, L_val)
        key, subkey = random.split(key)
        j0 = random.randint(subkey, (), 0, L_val)
        s_old = state[i0, j0]

        # 选择新自旋状态 (排除s_old)
        key, subkey = random.split(key)
        rand_val = random.randint(subkey, (), 0, q_val - 1)
        s_new = jnp.where(rand_val >= s_old, rand_val + 1, rand_val)

        # 初始化数据结构
        visited = jnp.zeros((L_val, L_val), dtype=jnp.bool_)
        max_size = L_val * L_val
        queue = jnp.full((max_size, 2), -1, dtype=jnp.int32)
        queue = queue.at[0].set(jnp.array([i0, j0]))
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

            # 计算邻居位置 (周期性边界条件)
            neighbors = jnp.array([
                [(i - 1) % L_val, j],
                [(i + 1) % L_val, j],
                [i, (j - 1) % L_val],
                [i, (j + 1) % L_val]
            ])

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

        # 翻转簇内自旋
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

        # 对于q-state Potts模型，序参量计算
        # 统计每种状态的数量
        counts = jnp.zeros(q_val)
        for s in range(q_val):
            counts = counts.at[s].set(jnp.sum(state == s))

        # 最大状态占比作为序参量
        max_count = jnp.max(counts)
        magnetization = max_count / (L_val * L_val)

        return energy, magnetization

    # 批量版本
    @jit
    def batch_calculate_observables(states):
        @vmap
        def calc(state):
            return calculate_observables(state)

        return calc(states)

    return batch_calculate_observables


# 初始化所有任务的状态和随机数
full_key = random.PRNGKey(initial_seed)
key, subkey = random.split(full_key)
states = random.randint(subkey, (n_temp, n_run, L, L), 0, q)

# 生成每个任务的随机数种子
keys = random.split(key, n_temp * n_run).reshape(n_temp, n_run, 2)

# 准备任务索引
task_indices = jnp.arange(total_tasks)

# 存储最终构型
final_states = onp.zeros((n_temp, n_run, L, L), dtype=onp.int32)

# 存储宏观参量
# 结构: [温度索引, 运行索引, 观测量]
# 观测量: 0=能量, 1=磁化强度, 2=能量平方, 3=磁化强度平方
observables = onp.zeros((n_temp, n_run, 4), dtype=onp.float32)

start_time = time.time()
print("Starting warmup steps...", flush=True)

# 创建Wolff更新器和可观测量计算器
wolff_updater = make_wolff_update(q, L, J)
observables_calculator = make_calculate_observables(L, q, J)

# 按批次处理任务
for b in range(n_batches):
    batch_start = b * batch_size
    batch_end = min((b + 1) * batch_size, total_tasks)
    batch_size_actual = batch_end - batch_start

    print(f"Processing batch {b + 1}/{n_batches} (size={batch_size_actual})", flush=True)
    batch_indices = task_indices[batch_start:batch_end]

    # 映射到温度和运行索引
    temp_indices = batch_indices // n_run
    run_indices = batch_indices % n_run

    # 获取当前批次数据
    batch_states = jnp.array(states[temp_indices, run_indices])
    batch_betas = betas[temp_indices]
    batch_keys = jnp.array(keys[temp_indices, run_indices])

    # 预热步骤
    last_time = time.time()
    for step in range(warmup_steps):
        if step % 10 == 0:
            current_time = time.time()
            if step == 0:
                print(f"  Batch {b + 1}, Starting warmup (Step 0/{warmup_steps})", flush=True)
            else:
                step_time = current_time - last_time
                print(f"  Batch {b + 1}, Step {step}/{warmup_steps} (Last 10 steps: {step_time:.2f}s)", flush=True)
            last_time = current_time  # 更新last_time

        # 执行Wolff更新
        batch_states, batch_keys = wolff_updater(
            batch_states, batch_betas, batch_keys
        )

        # 每100步输出完整结果（先删除上一次存储的全部内容）
        if step > 0 and step % 100 == 0:
            print(f"  Batch {b + 1}, Step {step}: Deleting previous intermediate results...", flush=True)

            # 删除输出文件夹中所有文件（保留文件夹本身）
            if os.path.exists(output_folder):
                for filename in os.listdir(output_folder):
                    file_path = os.path.join(output_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}', flush=True)
            else:
                os.makedirs(output_folder, exist_ok=True)

            print(f"  Batch {b + 1}, Step {step}: Saving complete intermediate results...", flush=True)

            # 临时存储当前batch的中间结果
            temp_final_states = onp.zeros((n_temp, n_run, L, L), dtype=onp.int32)
            temp_observables = onp.zeros((n_temp, n_run, 4), dtype=onp.float32)

            # 填充当前batch的数据
            for idx in range(batch_size_actual):
                i_temp = int(temp_indices[idx])
                j_run = int(run_indices[idx])
                temp_final_states[i_temp, j_run] = onp.array(batch_states[idx])

            # 计算当前batch的宏观参量
            energies, magnetizations = observables_calculator(batch_states)
            energies_np = onp.array(energies)
            magnetizations_np = onp.array(magnetizations)

            for idx in range(batch_size_actual):
                i_temp = int(temp_indices[idx])
                j_run = int(run_indices[idx])
                temp_observables[i_temp, j_run, 0] = energies_np[idx]
                temp_observables[i_temp, j_run, 1] = magnetizations_np[idx]
                temp_observables[i_temp, j_run, 2] = energies_np[idx] ** 2
                temp_observables[i_temp, j_run, 3] = magnetizations_np[idx] ** 2

            # 创建输出文件夹（确保存在）
            os.makedirs(output_folder, exist_ok=True)

            # 保存构型数据
            for i in range(n_temp):
                temp_data = temp_final_states[i].reshape(n_run, L * L)
                filename = os.path.join(output_folder, f"step{step}_config_temp_{temperatures[i]:.4f}.txt")
                onp.savetxt(filename, temp_data, fmt="%d")

            # 保存宏观参量
            for i in range(n_temp):
                temp_obs = temp_observables[i]
                filename = os.path.join(output_folder, f"step{step}_observables_temp_{temperatures[i]:.4f}.txt")
                header = "energy magnetization energy_squared magnetization_squared"
                onp.savetxt(filename, temp_obs, header=header, comments='', fmt="%.6f")

            # 计算并保存平均宏观参量
            avg_obs = onp.zeros((n_temp, 6))
            for i in range(n_temp):
                T = temperatures[i]
                E = temp_observables[i, :, 0]
                M = temp_observables[i, :, 1]
                E2 = temp_observables[i, :, 2]
                M2 = temp_observables[i, :, 3]

                avg_E = onp.mean(E)
                avg_M = onp.mean(M)
                C_v = (onp.mean(E2) - avg_E ** 2) / (k_B * T ** 2)
                chi = (onp.mean(M2) - avg_M ** 2) / (k_B * T)

                M4 = onp.mean(M ** 4)
                M2_avg = onp.mean(M2)
                binder = 1 - M4 / (3 * M2_avg ** 2) if M2_avg > 0 else 0.0

                avg_obs[i] = [T, avg_E, avg_M, C_v, chi, binder]

            avg_filename = os.path.join(output_folder, f"step{step}_averaged_observables.txt")
            header = "Temperature Energy Magnetization SpecificHeat Susceptibility BinderCumulant"
            onp.savetxt(avg_filename, avg_obs, header=header, comments='', fmt="%.6f")

            print(f"  Batch {b + 1}, Step {step}: Complete results saved (previous results deleted)", flush=True)
        # 计算宏观参量
    print("  Calculating observables...", flush=True)
    energies, magnetizations = observables_calculator(batch_states)

    # 转换为numpy数组
    energies_np = onp.array(energies)
    magnetizations_np = onp.array(magnetizations)

    # 保存最终构型和可观测量
    for idx in range(batch_size_actual):
        i = int(temp_indices[idx])  # 温度索引
        j = int(run_indices[idx])  # 运行索引

        final_states[i, j] = onp.array(batch_states[idx])
        # 存储可观测量
        observables[i, j, 0] = energies_np[idx]  # 能量
        observables[i, j, 1] = magnetizations_np[idx]  # 磁化强度
        observables[i, j, 2] = energies_np[idx] ** 2  # 能量平方 (用于比热计算)
        observables[i, j, 3] = magnetizations_np[idx] ** 2  # 磁化强度平方 (用于磁化率计算)

    # 更新主状态数组
    states = states.at[temp_indices, run_indices].set(onp.array(batch_states))
    keys = keys.at[temp_indices, run_indices].set(onp.array(batch_keys))
total_time = time.time() - start_time
print(f"Warmup completed in {total_time:.2f} seconds", flush=True)

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)
print(f"Saving results to {output_folder}/", flush=True)

# 保存构型数据
for i in range(n_temp):
    temp_data = final_states[i].reshape(n_run, L * L)
    filename = os.path.join(output_folder, f"config_temp_{temperatures[i]:.4f}.txt")
    onp.savetxt(filename, temp_data, fmt="%d")
    print(f"Saved configuration file: {filename}", flush=True)

# 保存宏观参量 - 每个温度一个文件
for i in range(n_temp):
    temp_obs = observables[i]  # 形状: [n_run, 4]
    filename = os.path.join(output_folder, f"observables_temp_{temperatures[i]:.4f}.txt")
    # 添加列标题
    header = "energy magnetization energy_squared magnetization_squared"
    onp.savetxt(filename, temp_obs, header=header, comments='', fmt="%.6f")
    print(f"Saved observables file: {filename}", flush=True)

# 计算并保存每个温度的平均宏观参量
avg_observables = onp.zeros((n_temp, 6))  # T, <E>, <M>, C_v, chi, binder

for i in range(n_temp):
    T = temperatures[i]
    E = observables[i, :, 0]  # 能量
    M = observables[i, :, 1]  # 磁化强度
    E2 = observables[i, :, 2]  # 能量平方
    M2 = observables[i, :, 3]  # 磁化强度平方

    # 计算平均值
    avg_E = onp.mean(E)
    avg_M = onp.mean(M)

    # 比热容 C_v = (<E^2> - <E>^2) / (k_B * T^2)
    C_v = (onp.mean(E2) - avg_E ** 2) / (k_B * T ** 2)

    # 磁化率 chi = (<M^2> - <M>^2) / (k_B * T)
    chi = (onp.mean(M2) - avg_M ** 2) / (k_B * T)

    # Binder累积量 U = 1 - <M^4>/(3<M^2>^2)
    M4 = onp.mean(M ** 4)
    M2_avg = onp.mean(M2)
    binder = 1 - M4 / (3 * M2_avg ** 2) if M2_avg > 0 else 0.0

    avg_observables[i] = [T, avg_E, avg_M, C_v, chi, binder]

# 保存平均宏观参量
avg_filename = os.path.join(output_folder, "averaged_observables.txt")
header = "Temperature Energy Magnetization SpecificHeat Susceptibility BinderCumulant"
onp.savetxt(avg_filename, avg_observables, header=header, comments='', fmt="%.6f")
print(f"Saved averaged observables to: {avg_filename}", flush=True)

print("Simulation completed successfully.", flush=True)