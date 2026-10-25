# cloud_s34 —— 云端会话产出索引

本目录由没有本机访问权限的云端会话产出（原因与时间线见 `LOG.md`）。全部内容自包含、可复现，不依赖 `verichrono/` 的任何文件；复制到本机后按 `LOG.md` §三接入。

| 路径 | 内容 | 怎么验 |
|---|---|---|
| `LOG.md` | 时间线、每步结果、未做与原因、本机接入步骤 | — |
| `方法批判与创新备忘.md` | 方法与理论批判；哲学透镜 → 机制 → 模拟证据；token 经济学；本机实验清单 | 对照 `sim/results/RESULTS.md` |
| `reviews/十位评委汇总.md` | 十位顶会评委模拟的去重待办（18 条，按致命度 × 成本排序，标出模拟已回答的） | 原文 `reviews/panel_A–D.md` |
| `reviews/panel_D_philosophy_related_work.md` 末节 | 相关工作检索，12 条「构成威胁」的文献 | 作者复核标「待核」者 |
| `sim/grpo_sim.py` | 决策级 GRPO 模拟器（四行为 / 六方案 / 三估计器），`python grpo_sim.py --selftest` | 10 项自检全过 |
| `sim/run_experiments.py` | E1–E10 全套，`python run_experiments.py`（约 5 分钟）或 `python run_experiments.py E3 E4` | 输出 `sim/results/`、`figures/` |
| `sim/results/RESULTS.md` | 所有表与结论（自动生成） | 与 CSV 一致 |
| `figures/S1–S10.{png,pdf}` | 相图 ×2、token-matched Pareto、自适应控制器、数据配比、滞后、弃权陷阱、估计器对比、成本色盲、Becker 杠杆 | 300 dpi + 矢量 |
| `data/nasa_canon/` | NASA 五千年食典 −699..−300 与 1701–1900 的 12 页（本机抓不到的），`MANIFEST.json` 含 sha256 与逐页行数 | `python fetch_nasa_canon.py --verify` |
| `data/dvsktt/` | 《大越史記全書》30 卷全文（维基文库，公有领域底本）、天象词频与可定到日计数 `probe.md`、接入说明 `INTEGRATION.md` | `python fetch_dvsktt.py --skip-fetch` |
| `logs/` | 抓取、自检、模拟运行的原始日志 | — |

只依赖 `numpy`、`matplotlib`（模拟）与标准库（抓取）。本机 `C:\Compiler\python.exe` 可直接运行。
