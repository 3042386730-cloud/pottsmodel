# =============================================================================
# Potts模型配置可视化工具 (简化版 + Matplotlib 3.7+ 兼容)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# ==================== 可调参数 ====================
INPUT_FILE = "config_step_43000_run_0_256.txt"  # 输入文件 (L×L 格式)
L = 256  # 体系尺寸 (必须与文件实际尺寸一致)
Q = 4  # 自旋态数量 (1, 2, ..., Q)
OUTPUT_FILE = "256_4.png"  # 输出文件名 (None=不保存)


# ==================== 颜色映射函数 ====================
def get_custom_cmap(Q):
    """获取颜色映射：Q<=5时使用红、蓝、黄、绿、紫，否则使用tab10"""
    if Q <= 5:
        colors = [
            [1.0, 0.0, 0.0],  # 红
            [0.0, 0.0, 1.0],  # 蓝
            [1.0, 1.0, 0.0],  # 黄
            [0.0, 0.8, 0.0],  # 绿
            [0.6, 0.0, 0.8],  # 紫
        ][:Q]
        return ListedColormap(colors)
    else:
        from matplotlib import colormaps
        return colormaps['tab10'].resampled(Q)


# ==================== 主程序 ====================
def main():
    # 1. 读取L×L矩阵
    lattice = np.loadtxt(INPUT_FILE, dtype=np.int32)

    # 验证尺寸
    if lattice.shape != (L, L):
        print(f"警告: 文件尺寸 {lattice.shape} 与设定 L={L} 不符!")

    # 2. 创建离散颜色映射
    cmap = get_custom_cmap(Q)
    bounds = np.arange(-0.5, Q, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 3. 绘制热力图
    plt.figure(figsize=(6, 6))
    im = plt.imshow(lattice, cmap=cmap, norm=norm, origin='lower')

    # 设置标签
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Potts Model (L={L}, Q={Q})')

    # 添加颜色条
    cbar = plt.colorbar(im, ticks=np.arange(Q))
    cbar.set_label('Spin State')

    # 保存或显示
    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {OUTPUT_FILE}")
    else:
        plt.show()

    print("可视化完成!")


if __name__ == "__main__":
    main()
