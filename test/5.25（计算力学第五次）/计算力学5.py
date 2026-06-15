import numpy as np
import json

# ===================== 1. 前处理：读取模型、生成IEN、LM矩阵 =====================
def preprocess(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    nsd = model["nsd"]
    ndof = model["ndof"]
    nnp = model["nnp"]
    nel = model["nel"]
    nen = model["nen"]
    E_list = np.array(model["E"], dtype=float)
    A_list = np.array(model["CArea"], dtype=float)
    x = np.array(model["x"], dtype=float)
    y = np.array(model["y"], dtype=float)
    coords = np.column_stack((x, y))
    IEN = np.array(model["IEN"], dtype=int) - 1
    fixed_dof = np.array(model["fixed_dof"], dtype=int) - 1
    fixed_val = np.array(model["fixed_value"], dtype=float)
    force_dof = np.array(model["force_dof"], dtype=int) - 1
    force_val = np.array(model["force_value"], dtype=float)

    # 生成对号矩阵 LM: (局部自由度, 单元号)
    total_dof = nnp * ndof
    LM = np.zeros((nen*ndof, nel), dtype=int)
    for e in range(nel):
        node1, node2 = IEN[e]
        for i in range(nen):
            node = IEN[e, i]
            for d in range(ndof):
                local_idx = i * ndof + d
                global_idx = node * ndof + d
                LM[local_idx, e] = global_idx
    return model, coords, IEN, LM, E_list, A_list, total_dof, fixed_dof, fixed_val, force_dof, force_val

# ===================== 2. 单元分析：计算单元刚度矩阵 =====================
def element_stiffness(e, coords, IEN, E, A, ndof):
    n1, n2 = IEN[e]
    x1, y1 = coords[n1]
    x2, y2 = coords[n2]
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L < 1e-12:
        raise ValueError("单元节点重合，计算失败！")
    c = dx / L
    s = dy / L
    EA_L = E * A / L
    if ndof == 1:
        Ke = EA_L * np.array([[1, -1], [-1, 1]])
    else:
        Ke = EA_L * np.array([
            [c**2, c*s, -c**2, -c*s],
            [c*s, s**2, -c*s, -s**2],
            [-c**2, -c*s, c**2, c*s],
            [-c*s, -s**2, c*s, s**2]
        ])
    return L, c, s, Ke

# ===================== 3. 总体刚度矩阵组装 =====================
def assemble_global(LM, IEN, coords, E_list, A_list, total_dof, ndof):
    nel = IEN.shape[0]
    K = np.zeros((total_dof, total_dof))
    elem_info = []
    for e in range(nel):
        E = E_list[e]
        A = A_list[e]
        L, c, s, Ke = element_stiffness(e, coords, IEN, E, A, ndof)
        elem_info.append([L, c, s, Ke])
        # 对号入座累加
        for a in range(Ke.shape[0]):
            for b in range(Ke.shape[1]):
                ga = LM[a, e]
                gb = LM[b, e]
                K[ga, gb] += Ke[a, b]
    return K, elem_info

# ===================== 4. 缩减法处理边界、求解方程、计算反力 =====================
def solve_reduction(K, total_dof, fixed_dof, fixed_val, force_dof, force_val):
    all_dof = np.arange(total_dof)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    nF = len(free_dof)
    nE = len(fixed_dof)

    # 分块矩阵
    KFF = K[np.ix_(free_dof, free_dof)]
    KFE = K[np.ix_(free_dof, fixed_dof)]
    KEF = K[np.ix_(fixed_dof, free_dof)]
    KEE = K[np.ix_(fixed_dof, fixed_dof)]

    # 荷载向量
    F = np.zeros(total_dof)
    F[force_dof] = force_val
    fF = F[free_dof]
    dE = fixed_val

    # 求解未知位移
    dF = np.linalg.solve(KFF, fF - KFE @ dE)

    # 重构整体位移
    D = np.zeros(total_dof)
    D[free_dof] = dF
    D[fixed_dof] = dE

    # 计算约束反力
    fE = KEF @ dF + KEE @ dE
    return D, fE, free_dof

# ===================== 5. 后处理：单元应力、轴力计算 =====================
def postprocess(D, LM, elem_info, E_list, A_list, ndof):
    nel = len(elem_info)
    elem_res = []
    for e in range(nel):
        L, c, s, Ke = elem_info[e]
        E = E_list[e]
        A = A_list[e]
        # 提取单元局部位移
        local_dof = LM[:, e]
        de = D[local_dof]
        if ndof == 1:
            sigma = E / L * np.array([-1, 1]) @ de
        else:
            sigma = E / L * np.array([-c, -s, c, s]) @ de
        N = sigma * A
        elem_res.append({
            "length": L, "cx": c, "cy": s,
            "disp": de, "stress": sigma, "axial_force": N
        })
    return elem_res

# ===================== 格式化输出矩阵 =====================
def print_mat(mat, name):
    print(f"\n===== {name} =====")
    for row in mat:
        print("  ".join(f"{x:10.6f}" for x in row))

# ===================== 主函数 =====================
def main(json_path):
    # 前处理
    model, coords, IEN, LM, E_list, A_list, total_dof, fixed_dof, fixed_val, force_dof, force_val = preprocess(json_path)
    ndof = model["ndof"]
    print("===== 对号矩阵 LM =====")
    print(LM)
    # 组装总刚
    K_global, elem_info = assemble_global(LM, IEN, coords, E_list, A_list, total_dof, ndof)
    print_mat(K_global, "总体刚度矩阵 K")
    # 求解
    D_total, reaction, free_dof = solve_reduction(K_global, total_dof, fixed_dof, fixed_val, force_dof, force_val)
    print("\n===== 整体节点位移 D =====")
    print(D_total)
    print("\n===== 约束自由度反力 =====")
    print(f"约束自由度编号(0索引): {fixed_dof}")
    print(f"约束反力: {reaction}")
    # 后处理
    elem_result = postprocess(D_total, LM, elem_info, E_list, A_list, ndof)
    # 输出单元结果
    for idx, res in enumerate(elem_result):
        print(f"\n===== 单元 {idx+1} 计算结果 =====")
        print(f"单元长度 L = {res['length']:.6f}")
        print(f"方向余弦 c = {res['cx']:.6f}, s = {res['cy']:.6f}")
        print(f"单元位移 de = {res['disp']}")
        print(f"单元应力 σ = {res['stress']:.6f}")
        print(f"单元轴力 N = {res['axial_force']:.6f}")
    # 检查对称性
    is_sym = np.allclose(K_global, K_global.T)
    print(f"\n===== 矩阵对称性检查 =====")
    print(f"总体刚度矩阵是否对称: {is_sym}")

if __name__ == "__main__":
    # 运行示例：main("case1.json") / main("case2.json")
    pass