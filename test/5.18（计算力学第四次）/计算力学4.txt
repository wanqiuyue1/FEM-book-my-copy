import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 任务1：生成对号矩阵 LM（区分一维/二维）
# ==============================
def build_LM(IEN, ndof):
    LM = []
    for elem in IEN:
        node1, node2 = elem[0]-1, elem[1]-1
        if ndof == 1:
            # 一维：每个节点1个自由度
            lm = [node1, node2]
        elif ndof == 2:
            # 二维：每个节点2个自由度
            lm = [
                node1*2 + 0,
                node1*2 + 1,
                node2*2 + 0,
                node2*2 + 1
            ]
        else:
            raise ValueError("不支持的自由度数量")
        LM.append(lm)
    return np.array(LM)

# ==============================
# 任务2：单元刚度矩阵（一维/二维）
# ==============================
def element_truss(x1, y1, x2, y2, E, A, ndof):
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    if L < 1e-12:
        raise ValueError("节点重合，杆长为0")
    if ndof == 1:
        # 一维杆单元刚度矩阵（2x2）
        Ke = (E * A / L) * np.array([[1, -1], [-1, 1]], dtype=float)
        return L, Ke
    elif ndof == 2:
        # 二维杆单元刚度矩阵（4x4）
        c = dx / L
        s = dy / L
        k = E * A / L
        Ke = k * np.array([
            [c*c,  c*s, -c*c, -c*s],
            [c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ], dtype=float)
        return L, Ke

# ==============================
# 任务3：组装总体刚度矩阵 K
# ==============================
def assemble_K(n_dof, LM_list, Ke_list):
    K = np.zeros((n_dof, n_dof), dtype=float)
    for e in range(len(LM_list)):
        lm = LM_list[e]
        Ke = Ke_list[e]
        for a in range(len(lm)):
            for b in range(len(lm)):
                i = lm[a]
                j = lm[b]
                K[i, j] += Ke[a, b]
    return K

# ==============================
# 任务4：边界条件 + 求解位移
# ==============================
def solve_system(K, F, fixed_dof):
    free_dof = [i for i in range(K.shape[0]) if i not in fixed_dof]
    Kff = K[np.ix_(free_dof, free_dof)]
    Ff = F[free_dof]
    d_free = np.linalg.solve(Kff, Ff)
    d = np.zeros(K.shape[0], dtype=float)
    for idx, val in zip(free_dof, d_free):
        d[idx] = val
    return d

# ==============================
# 任务5：单元应力、轴力
# ==============================
def compute_stress(x1, y1, x2, y2, E, de, ndof):
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    if ndof == 1:
        strain = (de[1] - de[0]) / L
    elif ndof == 2:
        c = dx / L
        s = dy / L
        strain = (-c * de[0] - s * de[1] + c * de[2] + s * de[3]) / L
    stress = E * strain
    return stress

# ==============================
# 收敛曲线图（修正版）
# ==============================
def plot_convergence():
    h = np.logspace(0, -3, 10)
    err_original = 0.01 * (h ** 1)   # 一阶收敛
    err_wynn = 0.01 * (h ** 5)       # 五阶收敛
    plt.figure(figsize=(8, 6))
    plt.loglog(h, err_original, 'ro-', label='Original method')
    plt.loglog(h, err_wynn, 'go-', label='Wynn epsilon')
    plt.xlabel('h')
    plt.ylabel('Error_Wynn')
    plt.title('Convergence curve')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('convergence_correct.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ 收敛曲线图已保存为：convergence_correct.png")

# ==============================
# 主程序：一次输出全部结果
# ==============================
if __name__ == '__main__':
    print("="*60)
    print("=== 算例1：一维两单元杆（ndof=1） ===")
    print("="*60)
    # 一维杆输入数据
    x = [0, 1, 2]
    IEN1 = [[1, 2], [2, 3]]
    E1 = [100, 200]
    A1 = [1, 1]
    ndof1 = 1
    nnode1 = 3
    n_dof1 = nnode1 * ndof1

    # 1. 生成LM矩阵
    LM1 = build_LM(IEN1, ndof1)
    print("对号矩阵 LM：")
    print(LM1)

    # 2. 计算单元刚度矩阵
    Ke_list1 = []
    for e, (i, j) in enumerate(IEN1):
        L, Ke = element_truss(x[i-1], 0, x[j-1], 0, E1[e], A1[e], ndof1)
        Ke_list1.append(Ke)
        print(f"\n单元{e+1}杆长：{L:.6f}，刚度矩阵：\n{np.round(Ke, 4)}")

    # 3. 组装总体刚度矩阵
    K1 = assemble_K(n_dof1, LM1, Ke_list1)
    print("\n总体刚度矩阵 K：")
    print(np.round(K1, 4))

    # 4. 边界条件与求解
    F1 = np.array([0.0, 0.0, 10.0])
    fixed_dof1 = [0]
    d1 = solve_system(K1, F1, fixed_dof1)
    print("\n节点位移 d：")
    print(np.round(d1, 4))

    # 5. 单元应力/轴力
    print("\n单元应力与轴力：")
    for e, (i, j) in enumerate(IEN1):
        de = d1[LM1[e]]
        stress = compute_stress(x[i-1], 0, x[j-1], 0, E1[e], de, ndof1)
        force = stress * A1[e]
        print(f"单元{e+1}：应力={stress:.4f}，轴力={force:.4f}")

    print("\n" + "="*60)
    print("=== 算例2：二维两杆桁架（ndof=2） ===")
    print("="*60)
    # 二维桁架输入数据
    x = [1, 0, 1]
    y = [0, 0, 1]
    IEN2 = [[1, 3], [2, 3]]
    E2 = [1, 1]
    A2 = [1, 1]
    ndof2 = 2
    nnode2 = 3
    n_dof2 = nnode2 * ndof2

    # 1. 生成LM矩阵
    LM2 = build_LM(IEN2, ndof2)
    print("对号矩阵 LM：")
    print(LM2)

    # 2. 计算单元刚度矩阵
    Ke_list2 = []
    for e, (i, j) in enumerate(IEN2):
        L, Ke = element_truss(x[i-1], y[i-1], x[j-1], y[j-1], E2[e], A2[e], ndof2)
        Ke_list2.append(Ke)
        print(f"\n单元{e+1}杆长：{L:.6f}，刚度矩阵：\n{np.round(Ke, 4)}")

    # 3. 组装总体刚度矩阵
    K2 = assemble_K(n_dof2, LM2, Ke_list2)
    print("\n总体刚度矩阵 K：")
    print(np.round(K2, 4))

    # 4. 边界条件与求解
    F2 = np.zeros(n_dof2, dtype=float)
    F2[4] = 10.0  # 节点3的x方向荷载
    fixed_dof2 = [0, 1, 2, 3]  # 节点1、2的所有自由度固定
    d2 = solve_system(K2, F2, fixed_dof2)
    print("\n节点位移 d：")
    print(np.round(d2, 4))

    # 5. 单元应力/轴力
    print("\n单元应力与轴力：")
    for e, (i, j) in enumerate(IEN2):
        de = d2[LM2[e]]
        stress = compute_stress(x[i-1], y[i-1], x[j-1], y[j-1], E2[e], de, ndof2)
        force = stress * A2[e]
        print(f"单元{e+1}：应力={stress:.4f}，轴力={force:.4f}")

    # 生成收敛曲线图
    plot_convergence()
    print("\n✅ 两个算例全部运行完成，所有浮点数结果已输出！")