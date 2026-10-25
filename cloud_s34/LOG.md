# S34（云端会话）工作日志

> 本目录 `cloud_s34/` 由云端 agent 在 **没有本机访问权限** 的情况下产出：本机桥接（MFMCP）在本窗口无法连通，GitHub 仓库 `ZhongQuji/newo` 只有一个空的 Initial commit，因此 **本机 `verichrono/` 的代码、数据、API 队列、GPU 都碰不到**。这里只做三类不依赖本机的事：① 本机抓不到的数据（NASA 食典补页、《大越史記全書》全文）；② 独立实现的决策级 GRPO 小模拟实验（方法变体与 token 效率）；③ 十位评委模拟与方法批判。全部可复现，每一步有日志。
>
> 时间均为 **北京时间（UTC+8）**。本地机器时钟是 UTC，文件 mtime 需 +8h。

## 一、时间线

| 时间 | 做了什么 | 结果 / 证据 |
|---|---|---|
| 16:50 | 收到任务；尝试连本机桥接 MFMCP | 工具发现失败、`mcp_auth` 两次超时；运行事件 0 条 → 不是鉴权被拒，是云端连不上本机服务 |
| 16:59 | 读完 2,892 行对话记录（S30ae–S33 全部阶段汇报），核对 GitHub 远端 | 远端只有 `5c9e5bd Initial commit`（7 字节 README）；本机项目从未推送 |
| 17:04 | 建分支 `cursor/s34-cloud-review-sim-data-fc90`，装 numpy/matplotlib/scipy | 见 `logs/` |
| 17:05 | 并行开 4 个评委子代理（A 理论/机制/效率，B 评测/统计/AC，C 数据/抽取/历史天文，D 科学哲学+相关工作检索） | 产出 → `reviews/panel_*.md`（完成后汇总到 `reviews/十位评委汇总.md`） |
| 17:08 | **NASA 食典补页**：S33 记录「本地 NASA 典范只到前 299 年，补抓前 700–300 年八页四次都失败（本机 SSL 断开）」。云端可达，抓 SE/LE 各 −699..−300 四个世纪（8 页）+ 1701–1900（4 页，解 《清實錄》1700 边界） | 12/12 页 200 OK；逐页解析行数 225–256（每世纪正常 220–260）、年份全在范围内、范围外 0 行；泰勒斯日食 −584 May 28、《至大论》冈比西斯七年月食 −522 Jul 17 01:23 TD 均命中。`data/nasa_canon/MANIFEST.json`（sha256）、`logs/nasa_canon_fetch.log` |
| 17:12 | **《大越史記全書》全文**：S33 记录「大越史记（网站动态加载、拿不到正文）」。改走中文维基文库 MediaWiki API（静态 wikitext），30 卷全部取到 | 550,688 字；关键词：日食 90、月食 52、行星犯 17、月犯 8、太白晝見 13、彗孛流星 64；**可定到日**：干支+朔+日食 27、月+朔+日食 33、晦+日食 20（外紀转录汉代）、望+月食 23、十五/十六日+月食 14。`data/dvsktt/{text,raw,dvsktt_all.txt,MANIFEST.json,probe.md}`，`logs/dvsktt_fetch.log` |
| 17:15 | 史料里现成的「核验者」片段（可进引言，与扎哈比驳斥阿布·沙马并列）：黎朝太史令裴時亨「妄奏是月十六日卯時月食，詔百官詣承天門救月，不見食」被劾；另有「密奏隐之，不救」的月食与两次密奏日食（五月丁丑朔、十一月戊辰朔）——四条都可用神谕判真伪 | `data/dvsktt/probe.md` 末节 |

| 17:20 | **模拟器** `sim/grpo_sim.py`：独立实现决策级 GRPO（四种行为 S0 无账本盖章 / S1 假账本盖章 / C 查证 / A 弃权；方案 outcome / balanced / tapo / ra / ra_ledger / adaptive；估计器 grpo / drgrpo / reinforce；向量化到 格子 × 种子）。自检 10 项全过（含：c>c* 锁死盖章、c<c* 锁死查证、q* 两侧、同组同动作梯度恰为 0、REINFORCE 基线仍动、**全对组里带成本样本的归一化优势 = −√((G−k)/k) 与 c 无关**、r_a=0 时弃权不出现） | `logs/sim_selftest.log` |
| 17:30–17:45 | 第一轮跑 E1–E8，发现两件事：① GRPO 的经验相边界整体偏向盖章侧（Δc ≈ −0.04），Dr. GRPO 回到解析线 → 加入估计器维度；② 自适应控制器把 q 推到 1 不松——目标失败率 τ=0.05 低于诚实查证的错误率 1−a=0.10 → τ 改 0.15；③ 接种复现在 TAPO 150 步切换时 100 步后 0.981、500 步后 0.007 → 加三档接种时长 | 中间版本的图已被覆盖，过程见 `logs/sim_run_console_*.log` |
| 17:50 | 新增 E9（锁死速度 vs 成本：GRPO 20 步恒定、Dr. GRPO 648→133 步）、E10（Becker 杠杆：GRPO 下 κ 0.5→20 几乎无效，Dr. GRPO 下 κ≥2 即保住） | |
| 17:58 | **全套 E1–E10 跑完**（300 s，CPU 单核），10 张图 `figures/S1–S10`，表 `sim/results/RESULTS.md` + 14 个 CSV | `logs/sim_run.log`、`logs/sim_run_console.log` |
| 17:40–18:00 | 四个评委小组交稿（`reviews/panel_A–D.md`，共 10 位评委 + 相关工作检索 29 条查询），汇总去重成 18 条待办 `reviews/十位评委汇总.md`；写 `方法批判与创新备忘.md`（哲学透镜 → 机制 → 模拟证据；token 经济学；本机实验清单） | |

### 模拟实验一句话结论（细节见 `sim/results/RESULTS.md`）
1. GRPO 的 std 归一化对查证成本「色盲」：全对混合组里查证样本的归一化优势恒为 −√((G−k)/k)，c=0.002 与 0.2 的锁死步数都是 ~20 步；Dr. GRPO 正比于 c（648 → 133 步）。
2. GRPO 有效 c* 比 a−π 低 0.04（符合率 91.8% vs Dr. GRPO 99.5%）；有效 q* 比命题 3 解析值高 4–5 倍（π=0.9：解析 0.056，GRPO 要 q≈0.3–0.5）。
3. 只罚「没账本」+ 审计不够（q=0.1）→ P(假账本)=0.99：伪推导的最小模型。
4. 账本感知 + 自适应审计：1.3 次神谕/样本达到 TAPO（3 次）的查证水平，控制器自动停在优化器的真实相边界；τ 必须 > 1−a；下限低于有效 q* 则复发。
5. 滞后长度 ∝ 1/(1−p0)：接种「保住」是吸收态几何，不是习惯（TAPO 150 步 → 切回 100 步后 0.981 → 500 步后 0.007）。
6. 弃权成为吸收态的条件 r_a > a−c−qκ(1−a)；r_a=0 永远安全。
7. Becker 杠杆（加罚少审）在 GRPO 下失效。
8. 数据配比替代审计：GRPO 下翻转点 f≈0.65（理论 0.22 的三倍），低于它两层一起锁死；Dr. GRPO 下 ≈0.22。
9. 精确零梯度吸收态属于 group-relative 基线族（GRPO、Dr. GRPO），REINFORCE+EMA 没有；但锁死快慢由 std 归一化决定。

## 二、未做与原因

- 本机 `verichrono/` 流水线接入（`build_sky_vn.py`、冻结、审计、建包）：**做不了**——看不到 `chrono.py / corpus.py / skyx.py / harness_v2.py` 的真实接口，盲写会重蹈 S30ae/S32 的「每段三轮修」。接入要点已写在 `data/dvsktt/INTEGRATION.md`，等本机会话按它写。
- 任何 API 小批量、任何 RL：无密钥、无 GPU、无数据集。
- OpenAlex 7,475 DOI 补查：deferred.json 在本机。
- 人工核验：仍是 0 条，只有人能做。

## 三、本机接入步骤（给下一段有本机权限的会话）

1. `git fetch origin cursor/s34-cloud-review-sim-data-fc90 && git checkout origin/cursor/s34-cloud-review-sim-data-fc90 -- cloud_s34/`（或在 GitHub 上下载该目录）。
2. NASA 补页：把 `cloud_s34/data/nasa_canon/pages/*.html` 复制到本机食典目录（`skyx.py`/`nasa_maps.py` 读 `SEcat5`/`LEcat5` 页的位置），把 `build_sky_bab.py` 里「前 299 年」的常量改到 −699，重建 BA 层（S33 记录「换网络抓到后改一个常量重建即可」）。1701–1900 页可选，用于把 `build_sky_shilu.py --qing` 的 1700 边界放宽（PL/MO/VD 仍受逐日星历 1645 限制，SE/LE 不受）。
3. 《大越史記全書》：按 `cloud_s34/data/dvsktt/INTEGRATION.md` 写 `build_sky_vn.py`（复用高麗/日本流水线：年号表 + 干支 + 站点）。
4. 模拟器与图：`cloud_s34/sim/` 可直接在本机 `C:\Compiler\python.exe` 运行（只依赖 numpy/matplotlib）。
