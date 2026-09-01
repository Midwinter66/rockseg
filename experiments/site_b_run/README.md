# 矿区 B 独立运行区

本目录用于在不改动矿区 A 的前提下，执行矿区 B 的冻结流程。所有新结果均写入本目录的 `outputs/`，原始数据不会被修改。

流程含义：

- `P0`：四叉树切片 → 预训练模型分割 → 跨切片融合；不启用三维几何筛查。
- `P1`：同一份切片和分割输出 → 跨切片融合 → 三维几何筛查。
- `volume`：只对 P1 的保留目标进行 2.5D 体积估算与二维代理体积比较。

运行前，请在项目根目录激活已验证的环境：

```powershell
conda activate rock
```

首先仅检查路径、模型、冻结参数与输出目录：

```powershell
python experiments/site_b_run/run_site_b.py --stage prepare
```

建议分阶段运行。每个阶段成功后再执行下一条：

```powershell
python experiments/site_b_run/run_site_b.py --stage slicing
python experiments/site_b_run/run_site_b.py --stage detection
python experiments/site_b_run/run_site_b.py --stage p0
python experiments/site_b_run/run_site_b.py --stage p1
python experiments/site_b_run/run_site_b.py --stage volume
```

若要只做分割的快速通路测试，可使用少量切片；此结果不得用于论文：

```powershell
python experiments/site_b_run/run_site_b.py --stage detection --limit 5
```

完整运行也可使用：

```powershell
python experiments/site_b_run/run_site_b.py --stage all
```

关键结果位置：

- `outputs/slicing/quadtree_dom/`：切片统计和覆盖图。
- `outputs/detection/quadtree_dom/`：分割检测清单与统计。
- `outputs/p0/fusion/quadtree_dom/correlation_clustering/`：P0 融合结果。
- `outputs/p1/fusion/quadtree_dom/correlation_clustering/`：P1 融合和三维筛查结果。
- `outputs/p1/volume/outputs/quadtree_dom/correlation_clustering/`：P1 的体积结果。
- `site_b_run_manifest.json`：输入文件、模型哈希、冻结参数哈希和每一步的状态。

坐标平移仅写在本目录的 `config/scene_b.json` 中；它不会修改 `experiments/common/scene_reference.py`，也不会覆盖矿区 A 结果。B 的坐标质量状态目前仍为“范围核验通过、待人工特征复核”，因此在论文中不要在完成人工残差检查前宣称其为严格的独立验证场景。
