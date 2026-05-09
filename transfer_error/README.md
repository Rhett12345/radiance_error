# FY-4 AGRI 辐射传输模型灵敏度实验

评估 FY-4A/B 卫星通道间辐射亮度转换模型对输入噪声的敏感性。通过蒙特卡洛方法，在源通道辐亮度上施加高斯扰动，传播通过转换模型后统计输出误差。

## 运行方式

**命令行（推荐）：**

```bash
cd ~/transfer_model
python run_sensitivity.py
```

`run_sensitivity.py` 是一个 CLI 入口脚本，负责收集数据文件、解析命令行参数，然后调用 `transfer_error.main()` 执行实验。支持的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `fy4a_fy4b_convolution/*_sat_rad.csv` | 模拟辐亮度 CSV，支持 glob |
| `--coeff` | `transfer_coeff_fy4a_fy4b_v1.csv` | 转换系数 CSV 路径 |
| `--outdir` | `./output` | 输出目录 |

```bash
# 指定自定义路径
python run_sensitivity.py --data "my_data/*.csv" --coeff my_coeffs.csv --outdir ./results
```

**Python 调用：**

```python
from transfer_error import main
main(["data1.csv", "data2.csv"], "coeffs.csv", "./output/")
```

## 输出文件

| 文件 | 内容 |
|------|------|
| `sensitivity_results.csv` | 每个通道 × 扰动幅度的统计量（dy_mean, dy_p95, rel_err_mean, dTb_mean, dTb_p95） |
| `fig_reflective.png` | 反射通道：相对误差均值 vs 扰动百分比 |
| `fig_ir.png` | 红外通道：亮温差均值 vs ΔK |

## 实验方法

1. **单位转换**：辐亮度通过雅可比变换从波数域（W·cm⁻²·sr⁻¹·cm）转到波长域（W·m⁻²·sr⁻¹·μm⁻¹）：L_λ = L_ν × 10⁸ / λ²

2. **加噪**（对每个场景、每个扰动幅度）：
   - 反射通道（ch01–ch06）：σ = 扰动比例 × x，扰动梯度 [0.5%, 1%, 2%, 3%, 4%, 5%]
   - 红外通道（ch07, ch09–ch14/15）：σ = |dL/dTb(Tb_typical)| × ΔK，扰动梯度 [0.5, 1, 2, 3, 4, 5] K

3. **传播**：y_noisy = f(x_noisy)，y_clean = f(x)，计算 δy = |y_noisy − y_clean|

4. **蒙特卡洛**：N=500 次重复，取均值和 p95

5. **红外额外统计**：δTb = |Tb(y_noisy) − Tb(y_clean)|

## 代码结构

```
transfer_error/
├── config.py          # 物理常数、通道波长字典、扰动参数、实验设置
├── physics.py         # 雅可比变换、Planck 函数、亮温反演、dL/dTb
├── io_utils.py        # 辐射数据加载、系数表加载、结果保存
├── sensitivity.py     # 噪声模型、转换模型求值、蒙特卡洛实验
├── plotting.py        # 灵敏度曲线绘图（300 dpi PNG）
├── main.py            # 主控程序，串联完整流程
└── __init__.py        # 包入口
```

各模块依赖关系：`config → physics → sensitivity → main`，`io_utils` 和 `plotting` 分别被 `main` 调用。

## 依赖

Python ≥ 3.10，numpy，scipy，matplotlib。不使用 sklearn。
