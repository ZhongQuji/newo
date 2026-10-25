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

（以下随工作追加）

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
