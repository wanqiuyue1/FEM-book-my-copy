# 总体刚度矩阵组装与桁架求解程序
## 运行环境
Python 3.8+
依赖库：numpy

## 文件说明
main.py        主程序
input1.json    算例1输入
input2.json    算例2输入
README.md      运行说明

## 运行步骤
1. 确保 input1.json、input2.json 与 main.py 在同一文件夹
2. 运行 main.py
3. 查看终端输出的位移、应力、反力、刚度矩阵等结果

## 输出内容
- 总体刚度矩阵 K
- 节点位移
- 单元应力、轴力
- 约束反力
- 矩阵对称性、奇异性验证