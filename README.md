# 多无人机失观围捕与动态编队重构

硕士论文实验的精简源码备份。保存二维训练、三维 Isaac 场景、控制方法、对比/消融入口、配置和测试；不包含数据集、训练权重、历史实验结果、日志、图片视频、论文 PDF 或对话交接文件。

## 内容与入口

| 内容 | 核心代码 | 运行入口 |
| --- | --- | --- |
| 第三章：二维部分观测围捕 | `envs/`、`algos/`、`belief/`、`safety/`、`utils/` | `train_baseline.py`、`train_belief.py`、`scripts/eval_policy.py` |
| 第四章：固定三机三维围捕、失观搜索与队形恢复 | `experiments/c166_unified_matrix/` | `python -m reproduce.native --task ch4 ...` |
| 第四章：高度自由度、安全层消融 | `experiments/c167_vertical_ablation/`、`c168_safety_ablation/` | `--task ch4-height`、`--task ch4-safety` |
| 第四章扩展：固定三机随机障碍地图 | 随机地图生成器与三维控制链 | `--task ch4-random`（新运行示例，不是历史固定地图矩阵） |
| 第五章：4→3、5→4、5→3 任务退出与编队重构 | `experiments/c171_*` 起的依赖链、`c210_radial_damping/` | `--task ch5` |
| 近期候选方法备份 | `c211_uniformity_handover/`、`c212_predictive_spread/` | `--arm uniformity_sector` / `predictive_spread_sector` |
| Isaac 仿真与低层控制 | `isaac_p5/source/bylw_isaac_p5/` | 由上述原生入口加载 |

C210 是当前保留的开发比较方法。C211、C212 是尚未被采用的候选方法，不代表更新后必然更好。当前第四、五章入口执行确定性控制器评估；二维 MAPPO 与保留的原生 MAPPO 训练入口另列。仓库不声称实现任意随机地图上的成功，也不将历史开发地图当成独立泛化测试。

## 二维环境安装与运行

使用 Python 3.10 的独立环境；先按设备安装适合的 PyTorch，再安装其余依赖。以下命令均在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/isaac_p5/source/bylw_isaac_p5"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q
python -m reproduce.verify
python train_baseline.py --config configs/journal_j12_static_handoff_last_r6.yaml --device cpu --seed 1 --iters 1
```

最后一条仅用于启动检查。完整训练去掉 `--iters 1`；GPU 训练改用 `--device cuda` 并先设置 `CUDA_VISIBLE_DEVICES`。输出写入 `runs/`，不会提交到 Git。主训练配置与无协同恢复控制器的对照分别是：

- `configs/journal_j12_static_handoff_last_r6.yaml`
- `configs/journal_j13_static_plain_last_r6.yaml`

两个配置均可从随机初始化训练。**历史 J12/J13 使用过前期检查点续训；从零开始运行上述配置不等同于恢复历史权重或保证复现历史百分比。** 比较时应统一初始化/续训方式、种子和计算预算。`configs/` 同时保留其他研究配置，部分 belief/anchor 配置需要自行生成预测器或前置策略权重。

训练后使用自己的检查点评估，例如：

```bash
python scripts/eval_policy.py --ckpt runs/YOUR_RUN/final.pt --config configs/eval_j12_static_tracked_reactive.yaml --episodes 100 --eval-seeds 3 --cbf --safety-version dynamic_v2 --reacquisition-controller
```

具体评估控制器参数应与训练配置一致；查看 `--help` 获取覆盖参数。预测器数据与权重可以通过 `scripts/collect_trajectories.py` 和 `scripts/train_predictor.py` 自行生成，不随本仓库发布。

## Isaac 三维实验

需要单独安装与项目兼容的 NVIDIA Isaac Sim / Isaac Lab、GPU 驱动及 Isaac Lab 自带无人机资产。扩展元数据面向 Isaac Lab 2.1；本次检查到本机安装分发版本为 `isaaclab==0.36.21`、`isaacsim==4.5.0.0`、`torch==2.5.1+cu121`、`numpy==1.26.4`（元数据标签不一致，迁移时优先核对实际安装版本与接口）；`requirements.txt` 只覆盖项目 Python 依赖，不能替代 Isaac 环境安装。不要在已有 Isaac 环境中盲目升级其绑定的 PyTorch。设置该环境的 Python 路径：

```bash
export ISAAC_PYTHON=/path/to/isaac/environment/bin/python
python -m reproduce.native --task ch4 --gpu 0 --out outputs/ch4 --dry-run
```

`--gpu` 是物理 GPU 编号，包装入口仅将该卡暴露为子进程的 `cuda:0`。示例编号应改成自己可用的 GPU。移除 `--dry-run` 执行。输出目录必须是新目录。

第四章主对比、两组消融示例（需要分别执行，计算量较大）：

```bash
python -m reproduce.native --task ch4 --gpu 0 --out outputs/ch4_main --methods brake_predict brake_search staged_brake all_search_staged
python -m reproduce.native --task ch4-height --gpu 0 --out outputs/ch4_height_free --methods all_search_staged --vertical-mode free
python -m reproduce.native --task ch4-height --gpu 0 --out outputs/ch4_height_fixed --methods all_search_staged --vertical-mode fixed
python -m reproduce.native --task ch4-safety --gpu 0 --out outputs/ch4_safety_on --methods all_search_staged --safety-layer on
python -m reproduce.native --task ch4-safety --gpu 0 --out outputs/ch4_safety_off --methods all_search_staged --safety-layer off
```

第四章固定三机不进行减员。三维随机地图扩展示例：

```bash
python -m reproduce.native --task ch4-random --gpu 0 --out outputs/ch4_random
```

第五章的配置文件 `reproduce/scenarios.json` 只保存场景种子、机数、退出时间/机号、运行时长和评估窗口，共 24+48 个历史开发回合配置，不含实验结果。退出机号在不同分组间变化，在同组对比方法间保持一致。

```bash
python -m reproduce.native --list-cases
python -m reproduce.native --task ch5 --case recent_4to3_g0 --arm radial_damped_sector --gpu 0 --out outputs/ch5_c210
python -m reproduce.native --task ch5 --case recent_4to3_g0 --arm sector_bypass --gpu 0 --out outputs/ch5_sector
python -m reproduce.score --case recent_4to3_g0 --run radial_damped_sector=outputs/ch5_c210 --run sector_bypass=outputs/ch5_sector --out outputs/ch5_comparison.json
```

其他机数使用 `--list-cases` 中对应 `5to4` 或 `5to3` 条目。默认保持历史配置；`--n`、`--seed-base`、`--duration` 的覆盖仅用于新的实验/启动检查，覆盖后不能混称原历史批次。评分器要求完整配置时长，不应用于缩短的启动检查。所有对比应使用相同场景、退出安排与评估窗口。

无需策略权重即可执行这些三维控制器入口。保留的 `scripts/train_p5_2_native_mappo.py` 用于研究原生 MAPPO 训练，其课程和奖励口径独立于 C166/C210；请先查看该入口的 `--help`，不要把一次训练启动等同于第四、五章方法的复现。

## 备份边界与验证

- 地图由随机种子和程序生成，不需要上传地图数据集；Isaac 资产由外部环境提供。
- 按原模块路径保留依赖链，避免整理备份时改动控制算法。仅将 C166/C167/C168 的历史快照登记替换为当前源码登记；原实验工作目录没有修改。
- `SOURCE_ORIGINS.json` 记录导出前源码的哈希；`SOURCE_MANIFEST.json` 记录本仓库导出文件的哈希。运行时会另外冻结当次源代码，便于核对。
- 3 项测试因旧录制/零样本管线或历史观测数据未上传而明确跳过；其余测试检验当前导出代码。CPU 测试不能证明三维物理成功率。
- 克隆此仓库可以重新训练和运行仿真；不能直接恢复未备份的历史权重和结果。完整数值复现实验还需匹配 Isaac/驱动/依赖版本和原训练检查点。

本次整理没有自动为项目添加开源许可证或转授第三方代码/资产的许可。源码注释与原有来源标识保留，外部 Isaac 软件及资产按各自许可安装使用。

本次备份验证：**541 项通过、3 项明确跳过**，二维 CPU 单次训练迭代通过；未重跑完整三维物理矩阵。详见 [导出说明](docs/EXPORT_NOTES.md)。
