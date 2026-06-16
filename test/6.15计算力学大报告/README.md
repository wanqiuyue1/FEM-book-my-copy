# 一维对流扩散方程有限元求解作业
## 1. 运行环境
- Python 3.7 及以上
- 依赖库：`numpy`、`matplotlib`
- 安装命令：
  pip install numpy matplotlib

## 2. 文件清单
1. advection_diffusion_fem.py ：主程序源代码
2. Pe_0.1_result.png ：Pe=0.1 数值解对比图
3. Pe_3.0_result.png ：Pe=3.0 数值解对比图
4. convergence_curve.png ：网格加密收敛曲线

## 3. 运行方式
1. 将所有文件放在同一文件夹
2. 打开终端/CMD，进入文件夹，执行：
   python advection_diffusion_fem.py

## 4. 计算参数说明
- 计算域长度 L = 1
- 对流速度 v = 1
- 基础单元数 nel = 20
- 计算工况：Pe=0.1（扩散占优）、Pe=3.0（对流占优）
- 格式定义：
  - α = 0 ：标准 Galerkin 有限元
  - α = 1 ：迎风格式（人工扩散）
  - α = α_opt ：SUPG/Petrov-Galerkin 稳定化格式

## 5. 输出内容
1. 控制台：扩散系数、各格式最大误差、矩阵对称性/正定性分析、网格收敛数据
2. 图片：解对比图、对数坐标收敛曲线