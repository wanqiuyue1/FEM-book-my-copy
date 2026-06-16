# 导入基础库
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置：解决中文乱码 + 负号显示问题 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 适配Windows
plt.rcParams["axes.unicode_minus"] = False  # 坐标轴负号正常显示

# ===================== 作业要求的标准函数接口 =====================
def alpha_supg(Pe):
    """
    计算SUPG最优参数 alpha_opt = coth(Pe) - 1/Pe
    处理Pe趋近于0，避免除零错误（作业要求）
    :param Pe: 单元Peclet数
    :return: 最优人工扩散系数 alpha_opt
    """
    Pe = np.float64(Pe)
    if Pe < 1e-8:
        return 0.0  # Pe趋近于0，直接置0
    coth_pe = 1.0 / np.tanh(Pe)
    alpha_opt = coth_pe - 1.0 / Pe
    return alpha_opt


def element_matrix(kappa, v, le, alpha):
    """
    构造两节点线性单元矩阵（严格按照作业公式编写）
    等效扩散系数: kappa_bar = kappa + alpha * v * le / 2
    :param kappa: 原始扩散系数
    :param v: 对流速度
    :param le: 单元长度
    :param alpha: 人工扩散参数
    :return: 2×2 单元矩阵 Ke
    """
    # 引入人工扩散后的等效扩散系数
    kappa_bar = kappa + alpha * v * le / 2

    # 扩散项矩阵
    K_diff = (kappa_bar / le) * np.array([[1, -1],
                                         [-1, 1]])
    # 对流项矩阵
    K_conv = (v / 2) * np.array([[-1, 1],
                                 [-1, 1]])
    # 总单元矩阵
    Ke = K_diff + K_conv
    return Ke


def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    求解一维对流扩散方程，完成网格生成、矩阵组装、边界条件、求解
    :param nel: 单元数量
    :param L: 计算域总长
    :param v: 对流速度
    :param kappa: 扩散系数
    :param alpha: 人工扩散参数
    :return: x(节点坐标), theta_num(数值解), theta_exact(精确解), K_global(整体矩阵)
    """
    # 1. 生成均匀网格
    nnodes = nel + 1  # 总节点数
    le = L / nel      # 单元长度
    x = np.linspace(0, L, nnodes)

    # 2. 初始化整体矩阵与右端向量
    K_global = np.zeros((nnodes, nnodes))
    F_global = np.zeros(nnodes)

    # 3. 逐单元组装整体刚度矩阵
    for elem in range(nel):
        n1 = elem
        n2 = elem + 1
        Ke = element_matrix(kappa, v, le, alpha)
        K_global[n1:n2+1, n1:n2+1] += Ke

    # 4. 施加狄利克雷边界条件 theta(0)=0，theta(L)=1
    # 左边界 x=0
    K_global[0, :] = 0.0
    K_global[:, 0] = 0.0
    K_global[0, 0] = 1.0
    F_global[0] = 0.0

    # 右边界 x=L
    K_global[-1, :] = 0.0
    K_global[:, -1] = 0.0
    K_global[-1, -1] = 1.0
    F_global[-1] = 1.0

    # 5. 求解线性方程组
    theta_num = np.linalg.solve(K_global, F_global)

    # 6. 计算精确解（使用expm1避免指数溢出，作业强制要求）
    ratio = v / kappa
    numerator = np.expm1(ratio * x)
    denominator = np.expm1(ratio * L)
    theta_exact = numerator / denominator

    return x, theta_num, theta_exact, K_global


def calculate_max_error(num_sol, exa_sol):
    """计算最大节点绝对误差"""
    return np.max(np.abs(num_sol - exa_sol))

# ===================== 主计算流程（完成全部作业任务） =====================
def main():
    # 全局固定参数（作业规定）
    L = 1.0
    nel_base = 20
    v = 1.0
    le_base = L / nel_base
    Pe_list = [0.1, 3.0]  # 两组计算工况

    print("=" * 60)
    print("        一维对流扩散方程有限元求解程序（作业专用版）")
    print("=" * 60)

    # 遍历两个Peclet数工况
    for Pe in Pe_list:
        print(f"\n【当前工况：单元Peclet数 Pe = {Pe:.2f}】")
        # 由 Pe = v*le/(2*kappa) 反求扩散系数 kappa
        kappa = (v * le_base) / (2 * Pe)
        print(f"计算得到扩散系数 κ = {kappa:.8f}")

        # -------- 1. 标准Galerkin  alpha=0 --------
        alpha_gal = 0.0
        x, theta_gal, theta_exa, K_gal = solve_advection_diffusion(nel_base, L, v, kappa, alpha_gal)
        err_gal = calculate_max_error(theta_gal, theta_exa)

        # -------- 2. 迎风格式  alpha=1 --------
        alpha_up = 1.0
        _, theta_up, _, _ = solve_advection_diffusion(nel_base, L, v, kappa, alpha_up)
        err_up = calculate_max_error(theta_up, theta_exa)

        # -------- 3. SUPG格式  alpha=alpha_opt --------
        alpha_sup = alpha_supg(Pe)
        _, theta_sup, _, _ = solve_advection_diffusion(nel_base, L, v, kappa, alpha_sup)
        err_sup = calculate_max_error(theta_sup, theta_exa)

        # 输出误差与参数
        print(f"标准Galerkin(α=0)   最大误差：{err_gal:.8e}")
        print(f"迎风格式(α=1)       最大误差：{err_up:.8e}")
        print(f"SUPG格式(α_opt)     最大误差：{err_sup:.8e}")
        print(f"SUPG最优参数 α_opt = {alpha_sup:.6f}")

        # -------- 绘图：区分线型与颜色，避免曲线完全重叠 --------
        plt.figure(figsize=(10, 6))
        plt.plot(x, theta_exa, 'k-', linewidth=2.5, label='精确解')
        plt.plot(x, theta_gal, 'r--', linewidth=2, label='标准Galerkin')
        plt.plot(x, theta_up, 'g-.', linewidth=2, label='迎风格式')
        plt.plot(x, theta_sup, 'b:', linewidth=2, label='SUPG/Petrov-Galerkin')

        plt.xlabel("空间坐标 x", fontsize=12)
        plt.ylabel(r"标量场 $\theta(x)$", fontsize=12)
        plt.title(f"一维对流扩散方程数值解对比 (Pe = {Pe:.2f}，单元数 nel=20)", fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"Pe_{Pe}_result.png", dpi=300, bbox_inches="tight")
        plt.show()

        # -------- 任务4：Pe=3.0 矩阵性质分析 --------
        if abs(Pe - 3.0) < 1e-6:
            print("\n" + "-" * 50)
            print("【Pe=3.0 标准Galerkin 整体矩阵分析】")
            K = K_gal
            # 1. 判断矩阵是否对称
            is_sym = np.allclose(K, K.T, atol=1e-10)
            print(f"整体矩阵是否对称：{is_sym}")

            # 2. 判断矩阵是否正定（特征值判定）
            eig_values = np.linalg.eigvals(K)
            real_eig = np.real(eig_values)
            min_eig = np.min(real_eig)
            is_pos_def = min_eig > -1e-10
            print(f"矩阵最小特征值：{min_eig:.8e}")
            print(f"整体矩阵是否正定：{is_pos_def}")
            print("结论：对流项引入非对称分量，矩阵非对称、非正定，对流占优时产生数值振荡")
            print("-" * 50)

    # ===================== 附加题：网格加密收敛性分析 =====================
    print("\n【附加题：网格加密收敛性分析（固定 Pe=3.0）】")
    Pe_test = 3.0
    nel_list = [10, 20, 40, 80]
    err_gal_list = []
    err_supg_list = []

    for nel_t in nel_list:
        le_t = L / nel_t
        kappa_t = (v * le_t) / (2 * Pe_test)
        # 计算Galerkin误差
        _, th_g, th_e, _ = solve_advection_diffusion(nel_t, L, v, kappa_t, alpha=0.0)
        err_g = calculate_max_error(th_g, th_e)
        # 计算SUPG误差
        alpha_t = alpha_supg(Pe_test)
        _, th_s, _, _ = solve_advection_diffusion(nel_t, L, v, kappa_t, alpha_t)
        err_s = calculate_max_error(th_s, th_e)

        err_gal_list.append(err_g)
        err_supg_list.append(err_s)
        print(f"单元数 nel={nel_t:2d} | Galerkin误差={err_g:.8e} | SUPG误差={err_s:.8e}")

    # 绘制对数坐标收敛曲线
    plt.figure(figsize=(8, 5))
    plt.loglog(nel_list, err_gal_list, 'ro-', markersize=7, linewidth=1.5, label='标准Galerkin')
    plt.loglog(nel_list, err_supg_list, 'bo-', markersize=7, linewidth=1.5, label='SUPG/Petrov-Galerkin')
    plt.xlabel("单元数量 nel", fontsize=12)
    plt.ylabel("最大节点误差（对数尺度）", fontsize=12)
    plt.title("网格加密收敛曲线 (Pe=3.0)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4, which="both", linestyle='--')
    plt.tight_layout()
    plt.savefig("convergence_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n程序运行结束！所有图片与数据已生成。")

# 程序入口
if __name__ == "__main__":
    main()