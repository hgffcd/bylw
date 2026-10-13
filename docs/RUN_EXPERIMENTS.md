# 运行二维与原生Isaac实验

源码包含训练、控制器、地图生成、几何/视场、安全检查和评分入口。训练权重、轨迹、论文结果、Isaac二进制和历史调度目录不在仓库中。先配置与requirements-tested.txt相符的Python依赖；Isaac入口须指定已安装且能导入isaaclab的Python。

完整性与CPU检查：

```bash
python -m reproduce.verify
python -m pytest -q
```

二维J13主线沿用configs/journal_j12_static_handoff_last_r6.yaml的完整方法和configs/journal_j13_static_plain_last_r6.yaml的Plain对照，训练入口train_belief.py、train_baseline.py，评价入口scripts/eval_policy.py。先用各入口的--help核对训练参数。这里从新训练复现方法，不提供旧检查点；不得把一次新训练或单步二维成功率直接当成三维持续围捕成绩。

第四章最终系统为ordinary_all；同版本比较ordinary_predict、ordinary_single、budget_all。示例先预览再去掉--dry-run运行；GPU编号按本机选择：

```bash
python -m reproduce.closing_native --chapter 4 --case ch4_final3_g0 --arm ordinary_all --isaac-python /path/to/isaac/python --gpu 0 --out runs/ch4_final3_g0_all --dry-run
```

第五章冻结C210，活动重构active与保留退出前槽位retained比较：

```bash
python -m reproduce.closing_native --chapter 5 --case ch5_5to3_g0 --arm active --isaac-python /path/to/isaac/python --gpu 0 --out runs/ch5_5to3_g0_active --dry-run
python -m reproduce.closing_native --chapter 5 --case ch5_5to3_g0 --arm retained --isaac-python /path/to/isaac/python --gpu 0 --out runs/ch5_5to3_g0_retained --dry-run
```

reproduce/closing_cases.json含第四章8组及第五章12组场景配置；全部组和全部方法都要保留，不能只运行已知易成功的种子。每个输出目录必须是新目录，部分失败输出保留排查。运行会冻结源码和调用参数。跨驱动/硬件数值逐位一致不作保证。

完成同一case的各方法后评分：

```bash
python -m reproduce.closing_score --case ch5_5to3_g0 --run active=runs/ch5_5to3_g0_active --run retained=runs/ch5_5to3_g0_retained --out runs/ch5_5to3_g0_scores.json
```

第四章相应使用ch4_final3_g0以及ordinary_all等ARM名称；评分入口记录实际n/seed/duration，默认按30/45/60秒。第五章从最后退出计时，第四章从任务开始计时，分母不能混合。第五章保留原几何保持主指标并额外报告连续可见的严格指标；第四章主判据已要求连续可见。

多GPU Isaac工作站可加`--physical-device`，让`--gpu 2`直接使用`cuda:2`，并清除子进程的CUDA_VISIBLE_DEVICES重映射。默认仍采用仅暴露所选GPU的模式。需要本地资产或独立Kit缓存时，使用`--kit-args="--portable-root /path/to/runtime --/renderer/multiGpu/enabled=false --/persistent/isaac/asset_root/cloud=/path/to/installed/assets"`。资产须为已安装的Isaac资产；仓库不包含第三方二进制资产。这两个选项仅配置运行环境，不修改场景或控制参数。

matched_all和smooth_all仅为第四章开发候选，入口保留以便复查，不是最终方法或已证明有效的改进。旧固定布局、垂向自由度和安全层实验继续使用reproduce/native.py。退出机是保持悬停的任务退出，不是电机失效坠落；地图限六混合障碍的通行约束分布。
