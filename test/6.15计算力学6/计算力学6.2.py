# -*- coding: utf-8 -*-
"""
有限元平衡方程组求解与误差分析 2-4 作业 最终修复版
功能：LDL^T分解求解器、残差/条件数计算、病态矩阵、桁架、非正定检测、Poisson有限元稀疏求解
适配：Python3.8+ / 数组下标：0起始 / 修复中文乱码、空值报错、数值边界问题
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
import math

# ===================== 全局配置：解决Matplotlib中文乱码 + 负号显示 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']    # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

# ===================== 一、自研 LDL^T 分解求解器 核心函数（作业要求必写） =====================
def ldlt_factor(K):
    """
    对称方阵 LDL^T 分解 K = L @ D @ L^T
    :param K: 输入对称稠密方阵
    :return: L(单位下三角), D(对角数组), flag(分解是否成功)
    """
    n = K.shape[0]
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    K_copy = K.copy()

    for j in range(n):
        # 计算对角元 D[j]
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] ** 2 * D[k]
        D[j] = K_copy[j, j] - sum_d

        # 检测非正主元 / 零主元（作业强制要求）
        if D[j] <= 1e-12:
            print(f"【错误】第 {j+1} 个主元非正或接近零，矩阵非正定/奇异！")
            return L, D, False

        # 计算下三角部分 L[i,j] (i > j)
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K_copy[i, j] - sum_l) / D[j]
    return L, D, True


def ldlt_solve(L, D, R):
    """
    LDL^T 三步求解：前代 -> 对角求解 -> 回代
    :param L: 单位下三角矩阵
    :param D: 对角数组
    :param R: 方程组右端向量
    :return: 解向量 a
    """
    n = len(R)
    y = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)

    # 1. 前代：L * y = R
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = R[i] - s

    # 2. 对角求解：D * z = y
    for i in range(n):
        z[i] = y[i] / D[i]

    # 3. 回代：L^T * a = z
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k, i] * a[k]
        a[i] = z[i] - s
    return a


def residual_norm(K, a, R):
    """
    计算残差向量、残差二范数、相对残差
    增加空值判断，避免分解失败后报错
    """
    if a is None:
        return None, None, None
    r = R - K @ a
    norm_r = np.linalg.norm(r, 2)
    norm_R = np.linalg.norm(R, 2)
    rel_res = norm_r / norm_R if norm_R > 1e-12 else 0.0
    return r, norm_r, rel_res

# ===================== 二、统一求解接口（对接2.3作业标准接口） =====================
def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    """
    作业标准统一接口：solve_equilibrium
    :param K_FF: 缩减刚度矩阵
    :param rhs: 缩减右端向量
    :param method: 求解方法 ldlt / scipy_sparse
    :return: 解向量a, 信息字典(耗时、残差、状态)
    """
    t0 = time.time()
    n = K_FF.shape[0]
    a = None
    info = {"n": n, "time_cost": 0.0, "success": True}

    if method == "ldlt":
        L, D, flag = ldlt_factor(K_FF)
        if not flag:
            info["success"] = False
            return None, info
        a = ldlt_solve(L, D, rhs)
    elif method == "scipy_sparse":
        # 转换为CSR稀疏格式（作业要求标准稀疏格式）
        K_sp = csr_matrix(K_FF)
        a = spsolve(K_sp, rhs)
    else:
        raise ValueError("暂不支持该求解方法，仅支持 ldlt / scipy_sparse")

    t1 = time.time()
    info["time_cost"] = t1 - t0
    r, norm_r, rel_res = residual_norm(K_FF, a, rhs)
    info["res_norm"] = norm_r
    info["rel_residual"] = rel_res
    return a, info

# ===================== 三、任务2：病态矩阵、误差、条件数分析（作业算例） =====================
def test_ill_condition():
    print("=" * 70)
    print("【算例：病态矩阵测试 | 任务2 误差与条件数分析】")
    # 作业指定病态矩阵
    K = np.array([[1.0000, 1.0000],
                  [1.0000, 1.0001]], dtype=np.float64)
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    cond_K = np.linalg.cond(K)
    print(f"矩阵条件数 cond(K) = {cond_K:.4e} (数值极大，典型病态矩阵)")

    # 1. 双精度求解
    a_double, info_d = solve_equilibrium(K, R, method="ldlt")
    r_d, nr_d, rr_d = residual_norm(K, a_double, R)
    err_d = np.linalg.norm(a_double - a_exact) / np.linalg.norm(a_exact)
    print("\n----- 双精度计算结果 -----")
    print(f"数值解: {a_double}, 相对残差: {rr_d:.4e}, 相对误差: {err_d:.4e}")

    # 2. 四舍五入至4位有效数字（修复log10边界报错）
    def round4sig(x):
        if abs(x) < 1e-15:
            return 0.0
        digit = 4 - int(np.floor(np.log10(abs(x))) + 1)
        return np.round(x, digit)

    K_round = np.vectorize(round4sig)(K)
    R_round = np.vectorize(round4sig)(R)
    a_round, info_r = solve_equilibrium(K_round, R_round, method="ldlt")

    print("\n----- 4位有效数字舍入后结果 -----")
    if info_r["success"]:
        r_r, nr_r, rr_r = residual_norm(K_round, a_round, R_round)
        err_r = np.linalg.norm(a_round - a_exact) / np.linalg.norm(a_exact)
        print(f"数值解: {a_round}, 相对残差: {rr_r:.4e}, 相对误差: {err_r:.4e}")
    else:
        print("矩阵受舍入扰动变为非正定/奇异，求解失败")
        print("分析：病态矩阵对微小数值扰动极度敏感，残差小不代表解准确")
    print("=" * 70 + "\n")

# ===================== 四、算例0：一维两单元桁架（对接2.3作业） =====================
def test_truss_1d():
    print("=" * 70)
    print("【算例0：一维两单元杆桁架 | 对接2.3作业】")
    # 作业给定总体刚度矩阵
    K_full = np.array([[100, -100, 0],
                       [-100, 300, -200],
                       [0, -200, 200]], dtype=np.float64)
    # 边界条件：d1=0(已知自由度0)，未知自由度 1、2
    free_dof = [1, 2]
    K_FF = K_full[np.ix_(free_dof, free_dof)]
    rhs = np.array([0.0, 10.0])

    a_F, info = solve_equilibrium(K_FF, rhs, method="ldlt")
    print(f"求解耗时: {info['time_cost']:.6f} s")
    print(f"未知位移 d2={a_F[0]:.4f}, d3={a_F[1]:.4f} (理论值 0.1 / 0.15)")
    print(f"相对残差: {info['rel_residual']:.4e}")

    # 重构全场位移（2.3后处理模块）
    d_full = np.zeros(3)
    d_full[free_dof] = a_F
    print(f"完整节点位移向量: {d_full}")

    # 计算约束反力
    force_full = K_full @ d_full
    print(f"节点1约束反力: {force_full[0]:.4f}")
    print("=" * 70 + "\n")

# ===================== 五、算例1：多阶三对角对称正定矩阵 =====================
def test_tridiagonal():
    print("=" * 70)
    print("【算例1：三对角对称正定矩阵 | 效率与精度测试】")
    for n in [10, 100, 500, 1000]:
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2.0
            if i > 0:
                K[i, i - 1] = -1.0
                K[i - 1, i] = -1.0
        a_exact = np.ones(n)
        R = K @ a_exact

        t0 = time.time()
        a_sol, info = solve_equilibrium(K, R, method="ldlt")
        t1 = time.time()
        err = np.linalg.norm(a_sol - a_exact) / np.linalg.norm(a_exact)
        print(f"阶数 n={n:4d} | 耗时={t1-t0:.6f}s | 相对误差={err:.4e}")
    print("分析：稠密LDL^T耗时随矩阵阶数快速上升，高阶矩阵内存开销巨大")
    print("=" * 70 + "\n")

# ===================== 六、算例2：非正定矩阵检测（零主元/非正定判断） =====================
def test_non_positive():
    print("=" * 70)
    print("【算例2：非正定矩阵检测 | 主元判断测试】")
    # 作业指定非正定矩阵
    K = np.array([[1, 2], [2, 1]], dtype=np.float64)
    R = np.array([1, 1])
    a_sol, info = solve_equilibrium(K, R, method="ldlt")
    print(f"LDL^T分解状态(成功=True/失败=False): {info['success']}")
    print("分析：有限元模型约束不足时会出现刚体位移，刚度矩阵奇异、产生零主元")
    print("=" * 70 + "\n")

# ===================== 七、算例4：二维Poisson方程有限元（Q4单元 大规模稀疏算例） =====================
def poisson_fem_q4(nx, ny):
    """
    单位正方形 Poisson 方程：-Δu = 2π²sinπx sinπy，全域边界u=0
    单元：Q4双线性四边形单元 | 稀疏求解器：Scipy CSR格式
    """
    Lx, Ly = 1.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    nnx = nx + 1
    nny = ny + 1

    # 1. 生成节点坐标
    x = np.linspace(0, Lx, nnx)
    y = np.linspace(0, Ly, nny)
    X, Y = np.meshgrid(x, y)
    node_coords = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    n_node = nnx * nny

    # 2. 生成Q4单元列表
    elem_list = []
    for ey in range(ny):
        for ex in range(nx):
            n0 = ey * nnx + ex
            n1 = n0 + 1
            n2 = n0 + nnx + 1
            n3 = n0 + nnx
            elem_list.append([n0, n1, n2, n3])
    n_elem = len(elem_list)

    # 3. 边界自由度（Dirichlet边界 u=0）
    bound_dof = []
    for i in range(n_node):
        xi, yi = node_coords[i]
        if abs(xi) < 1e-6 or abs(xi - 1) < 1e-6 or abs(yi) < 1e-6 or abs(yi - 1) < 1e-6:
            bound_dof.append(i)
    free_dof = [i for i in range(n_node) if i not in bound_dof]
    n_free = len(free_dof)
    fd_map = {d: idx for idx, d in enumerate(free_dof)}

    # 4. 2×2高斯积分点
    gauss_xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    gauss_w = np.array([1.0, 1.0])

    # 5. 总体矩阵与右端项初始化
    K_global = np.zeros((n_free, n_free))
    R_global = np.zeros(n_free)

    # 单元组装计时
    t_assemble = time.time()
    for elem in elem_list:
        nds = elem
        xy_e = node_coords[nds]
        Ke = np.zeros((4, 4))
        Re = np.zeros(4)

        for gi, xi in enumerate(gauss_xi):
            for gj, eta in enumerate(gauss_xi):
                w = gauss_w[gi] * gauss_w[gj]
                # Q4形函数与偏导数
                N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                    (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                dNdxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
                dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])

                # 雅可比矩阵
                J = np.zeros((2, 2))
                for i in range(4):
                    J[0, 0] += dNdxi[i] * xy_e[i, 0]
                    J[0, 1] += dNdxi[i] * xy_e[i, 1]
                    J[1, 0] += dNdeta[i] * xy_e[i, 0]
                    J[1, 1] += dNdeta[i] * xy_e[i, 1]
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dNdx = invJ @ np.vstack([dNdxi, dNdeta])
                B = dNdx

                # 单元刚度矩阵、单元载荷向量
                Ke += (B.T @ B) * detJ * w
                xg = N @ xy_e[:, 0]
                yg = N @ xy_e[:, 1]
                f_val = 2 * math.pi ** 2 * math.sin(math.pi * xg) * math.sin(math.pi * yg)
                Re += N * f_val * detJ * w

        # 组装到总体矩阵
        for i in range(4):
            if nds[i] in fd_map:
                ii = fd_map[nds[i]]
                R_global[ii] += Re[i]
                for j in range(4):
                    if nds[j] in fd_map:
                        jj = fd_map[nds[j]]
                        K_global[ii, jj] += Ke[i, j]
    t_assemble_end = time.time()

    # 6. 稀疏求解（CSR格式）
    a_free, info = solve_equilibrium(K_global, R_global, method="scipy_sparse")
    t_solve = info["time_cost"]

    # 7. 计算全场解与误差
    u_num = np.zeros(n_node)
    for idx, dof in enumerate(free_dof):
        u_num[dof] = a_free[idx]
    u_exact = np.sin(math.pi * node_coords[:, 0]) * np.sin(math.pi * node_coords[:, 1])
    err_node = np.abs(u_num - u_exact)
    max_err = np.max(err_node)
    l2_err = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)

    # 8. 输出作业要求信息
    print("=" * 70)
    print(f"【算例4：Poisson方程有限元 Q4单元 | 网格 {nx}×{ny}】")
    print(f"单元数: {n_elem} | 总节点数: {n_node} | 自由度数: {n_free}")
    print(f"矩阵稀疏格式: CSR | 装配耗时: {t_assemble_end - t_assemble:.6f} s")
    print(f"求解耗时: {t_solve:.6f} s | 总耗时: {(t_assemble_end - t_assemble)+t_solve:.6f} s")
    print(f"节点最大误差: {max_err:.4e} | L2相对误差: {l2_err:.4e}")
    print(f"相对残差: {info['rel_residual']:.4e}")
    print("=" * 70)

    # 9. 绘图：数值解云图 + 误差云图（中文正常显示）
    u_2d = u_num.reshape(nny, nnx)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u_2d, cmap="jet")
    plt.colorbar()
    plt.title("Poisson方程 数值解云图")

    err_2d = err_node.reshape(nny, nnx)
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, err_2d, cmap="Reds")
    plt.colorbar()
    plt.title("误差云图")

    plt.tight_layout()
    plt.show()
    return max_err, l2_err

# ===================== 主程序入口：按作业顺序运行全部算例 =====================
if __name__ == "__main__":
    # 任务2 病态矩阵分析
    test_ill_condition()

    # 算例0 一维桁架（对接2.3）
    test_truss_1d()

    # 算例1 三对角矩阵效率测试
    test_tridiagonal()

    # 算例2 非正定矩阵检测
    test_non_positive()

    # 算例4 Poisson方程 50×50网格（基础规模，运行速度快）
    poisson_fem_q4(50, 50)

    # 可选：100×100网格（规模更大，耗时更长，按需开启）
    # poisson_fem_q4(100, 100)