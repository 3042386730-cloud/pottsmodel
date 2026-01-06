import pandas as pd
import matplotlib.pyplot as plt

# 读取数据（假设文件名为 data.txt，空格分隔）
df = pd.read_csv('step8700_averaged_observables.txt', sep='\s+')

# 提取列
T = df['Temperature']
energy = df['Energy']
magnetization = df['Magnetization']
specific_heat = df['SpecificHeat']
susceptibility = df['Susceptibility']
binder = df['BinderCumulant']

# 设置绘图样式
plt.style.use('seaborn-v0_8-poster')  # 可选：更清晰的视觉效果

# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Physical Observables vs Temperature', fontsize=20)

# 绘制各物理量
axs[0, 0].plot(T, energy, 'o-', color='tab:blue')
axs[0, 0].set_xlabel('Temperature')
axs[0, 0].set_ylabel('Energy')
axs[0, 0].grid(True)

axs[0, 1].plot(T, magnetization, 'o-', color='tab:orange')
axs[0, 1].set_xlabel('Temperature')
axs[0, 1].set_ylabel('Magnetization')
axs[0, 1].grid(True)

axs[0, 2].plot(T, specific_heat, 'o-', color='tab:green')
axs[0, 2].set_xlabel('Temperature')
axs[0, 2].set_ylabel('Specific Heat')
axs[0, 2].grid(True)

axs[1, 0].plot(T, susceptibility, 'o-', color='tab:red')
axs[1, 0].set_xlabel('Temperature')
axs[1, 0].set_ylabel('Susceptibility')
axs[1, 0].grid(True)

axs[1, 1].plot(T, binder, 'o-', color='tab:purple')
axs[1, 1].set_xlabel('Temperature')
axs[1, 1].set_ylabel('Binder Cumulant')
axs[1, 1].grid(True)

# 隐藏最后一个子图（因为只有5个量）
axs[1, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留空间
plt.savefig('observables_vs_T.png', dpi=150)
plt.show()