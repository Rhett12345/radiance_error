# FY-4 AGRI Transfer Fitting Refactor

把原来一个大脚本拆成：配置、通道表、数据读取、拟合、绘图、导出、运行入口。以后换 AC / BC / AB 不用手改 Python 代码。

## 运行方式

在 `transfer_refactor` 目录下运行：

```bash
python run_fit.py --pair ac
python run_fit.py --pair bc
python run_fit.py --pair ab
```

也可以用 JSON 配置：

```bash
python run_fit.py --config configs/ac.json
python run_fit.py --config configs/bc.json
```

不想画图、只想快速调拟合：

```bash
python run_fit.py --pair ac --no-plots
```

生成一个配置模板：

```bash
python run_fit.py --pair bc --write-template configs/my_bc.json
```

## 文件说明

```text
fy4_transfer_refactor/
├── run_fit.py                     # 命令行入口
├── configs/
│   ├── ac.json                    # FY4A-FY4C 配置
│   ├── bc.json                    # FY4B-FY4C 配置
│   └── ab.json                    # FY4A-FY4B 配置
└── fy4_transfer/
    ├── channel_catalog.py         # 所有卫星通道、波长、默认配对关系
    ├── config.py                  # 配置读取/默认配置
    ├── io.py                      # CSV 读取、Radiance 转换、合并数据
    ├── model.py                   # 线性/二次多项式拟合
    ├── plotting.py                # 画图
    ├── export.py                  # 保存系数 CSV
    └── runner.py                  # 串联完整流程
```

## 需要改哪里？

多数情况下只改 `configs/*.json` 的输入目录：

```json
"input_globs": ["../convolution_result/fy4a_fy4c_convolution/*_sat_rad.csv"]
```

通道配对统一放在 `fy4_transfer/channel_catalog.py` 的 `PAIR_PRESETS`。如果要临时试验某一组通道，不改 Python，也可以在 JSON 里加：

```json
"channel_pair_overrides": [
  ["ch01", "ch01"],
  ["ch02", "ch04"],
  ["ch07", "ch10"]
]
```

## 依赖

和你的原脚本一致，主要是：

```bash
pip install numpy scipy scikit-learn matplotlib
```
