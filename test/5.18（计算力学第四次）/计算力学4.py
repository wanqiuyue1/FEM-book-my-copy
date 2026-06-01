import numpy as np


# ========================
# 1. 三维杆单元刚度矩阵（你上周用的原版函数）
# ========================
def truss3d_element_stiffness(x1, x2, E, A):
    x1 = np.array(x1)
    x2 = np.array(x2)
    dx = x2 - x1
    L = np.linalg.norm(dx)

    if L < 1e-12:
        raise ValueError("错误：两个节点重合，无法计算！")

    cx = dx[0] / L
    cy = dx[1] / L
    cz = dx[2] / L

    # 6×6 单元刚度矩阵（原版，你上周用的）
    C = np.array([
        [cx * cx, cx * cy, cx * cz, -cx * cx, -cx * cy, -cx * cz],
        [cx * cy, cy * cy, cy * cz, -cx * cy, -cy * cy, -cy * cz],
        [cx * cz, cy * cz, cz * cz, -cx * cz, -cy * cz, -cz * cz],
        [-cx * cx, -cx * cy, -cx * cz, cx * cx, cx * cy, cx * cz],
        [-cx * cy, -cy * cy, -cy * cz, cx * cy, cy * cy, cy * cz],
        [-cx * cz, -cy * cz, -cz * cz, cx * cz, cy * cz, cz * cz]
    ])
    Ke = E * A / L * C
    return L, (cx, cy, cz), Ke


# ========================
# 2. 应变、应力、轴力（原版）
# ========================
def truss3d_element_stress(x1, x2, E, A, de):
    x1 = np.array(x1)
    x2 = np.array(x2)
    dx = x2 - x1
    L = np.linalg.norm(dx)
    cx, cy, cz = dx / L
    de = np.array(de)

    # 轴向伸长
    delta = cx * (de[3] - de[0]) + cy * (de[4] - de[1]) + cz * (de[5] - de[2])
    eps = delta / L
    sig = E * eps
    N = sig * A
    return eps, sig, N


# ========================
# 3. 矩阵格式化输出（你最想要的！打印 6×6 矩阵）
# ========================
def print_matrix(mat, name="矩阵"):
    print(f"\n========== {name} ==========")
    for row in mat:
        s = " ".join([f"{v:>12.6f}" for v in row])
        print(s)


# ========================
# 4. 算例1（你上周跑的第一个算例）
# ========================
print("========== 算例 1：沿X轴杆 ==========")
x1 = [0, 0, 0]
x2 = [2, 0, 0]
E = 200e9
A = 1.0e-4
de = [0, 0, 0, 1e-3, 0, 0]

L, dircos, Ke = truss3d_element_stiffness(x1, x2, E, A)
eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)

print("长度 L =", L)
print("方向余弦 cx,cy,cz =", dircos)
print_matrix(Ke, "6×6 刚度矩阵 Ke")
print("应变 ε =", eps)
print("应力 σ =", sig)
print("轴力 N =", N)

# ========================
# 5. 算例2（你上周跑的第二个算例）
# ========================
print("\n\n========== 算例 2：空间任意方向杆 ==========")
x1 = [0, 0, 0]
x2 = [1, 2, 2]
E = 210e9
A = 2.0e-4
de = [0, 0, 0, 1e-3, 2e-3, 2e-3]

L, dircos, Ke = truss3d_element_stiffness(x1, x2, E, A)
eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)

print("长度 L =", L)
print("方向余弦 cx,cy,cz =", dircos)
print_matrix(Ke, "6×6 刚度矩阵 Ke")
print("应变 ε =", eps)
print("应力 σ =", sig)
print("轴力 N =", N)