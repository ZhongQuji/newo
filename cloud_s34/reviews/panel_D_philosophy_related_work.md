# 评委面板 D：科学哲学 / 认识论评审 + 相关工作检索（VeriChrono / 验证悖论）

> 元信息 · 生成时间 2026-10-25 09:15 UTC · 依据记录 `uploads/____-newo-20261025-162049_67b0.md` 行 229–270、877–896、1297–1356、1570–1617、1820–1845、1925–1945、2503–2526、2682–2695、2841–2854（仅 agent 阶段汇报正文，未读其它行）· 检索工具 WebSearch，查询词（29 条，节录）："verification paradox" LLM fact-checking rubber-stamping；"rubber-stamp" LLM verifier self-verification sycophancy；GRPO zero advantage all-correct group zero gradient；Dr. GRPO DAPO dynamic sampling entropy collapse；RLVR entropy collapse "absorbing state"；LLM-as-judge fabricated reasoning chain；ProcessBench PRM fragility；"Lessons of Developing Process Reward Models"；"Monitoring Reasoning Models for Misbehavior" obfuscation；"Pitfalls of Rule- and Model-based Verifiers"；RL tool-call cost penalty "when to search" abstention reward；"costly state verification" Townsend random audit；AI control auditing budget Greenblatt；TruthRL "learning to abstain"；"Beyond Binary Rewards" RLCR；"Hallucination Tax" reinforcement finetuning；"Spurious Rewards" RLVR；"The Policy Cliff"；LLM benchmark historical eclipse records ancient astronomical；classical Chinese 二十四史 ganzhi calendar LLM benchmark；Stephenson historical eclipses ΔT database；computational verification ancient Chinese astronomical records；faithfulness chain-of-thought Turpin Lanham；"Reasoning Models Don't Always Say What They Think"；"Prover-Verifier Games"；LLM fact verification base rate bias FEVER；"truth bias" LLM；skeptical warning prompt fact-checking；LLM verifier surface consistency recomputation。

---

## 任务一 · 评委 7（科学哲学 / 认识论）

**标题**：盖章是「坏习惯」还是「理性先验」？——VeriChrono 的概念基础评审
**分数**：6 / 10　**置信度**：4 / 5

### 优点
1. 用闭卷 / 事实在手 / 带工具三种证据条件把「不去验、不会整合、算了不用」拆开测（1301 行：事实在手 F=0，不调用 ≤2.5%），是罕见的可操作认识论设计。
2. 物理神谕充当皮尔斯的「外在恒常物」：真值不依赖任何模型或标注者（1830 行：89.9 万项审计 0 不符）；史官真错误与构造孪生 F 无差（887 行）给出构念效度。
3. 事前预测并命中（1307 行 20/20；1832–1836 行 q* 两侧），认识姿态正确。
4. 诚实报告局限（主表单一家族、跨家族批 n=20–40、PL 孪生裕度失配）。

### 问题
**P1（高）意图词汇混入度量定义。** 「编造」「诚实弃权」「盖章」是心理归因，度量测的却是行为：F 是负孪生判一致率，「伪推导」是瓶颈事实与神谕矛盾。AR 层换算精确率仅 16% 却照样判一致（2690 行），说明相当部分是「不会算」而非「明知故犯」。修复：定义层只用行为词（acceptance rate、oracle-inconsistent derivation、confabulation、abstention），意图性解读只放讨论节并标注为解释。成本：零。

**P2（高）「习惯」叙事与「事实在手 F=0」矛盾。** 习惯是对证据不敏感的倾向；F=0 却说明策略随证据通道切换——闭卷接受、有证据就查，这正是 π 高且查证有成本时的贝叶斯最优。唯一反贝叶斯证据「告知一半伪造 F 反升」（1303 行：LE 0.53→0.87）只来自单一家族的 pro 型号、两层；而 Markowitz & Hancock（2024）给出基率后 LLM 的真话偏倚是下降的，方向相反。修复：拆成两层论点——推理期「接受由先验驱动、对声明基率不敏感」（校准失败）；训练期「结局奖励消灭可用且划算的查证并跨层迁移」（P9；接种滞后 0.976，1935 行）——「习惯」只用于后者。补基率剂量实验（告知 π∈{0.05, 0.5, 0.95}，≥2 家族，同报伪推导率）。成本：中（一批 API）。

**P3（中）闭卷下的「查证」实为「回忆」。** 闭卷无法重算星历，账本正确只能来自记忆或运气，J 在闭卷下是「有依据主张率」而非查证率。修复：J 按条件定义，「验证」一词只用于有证据通道的条件；并报告 H−F，把盖章定义为「F 高且 H−F≈0」而非 F 单独。成本：低（重分析）。

**P4（中）皮尔斯映射近于装饰，康德词汇方向可疑。** 四法讲信念在时间与社群中如何固定，套到单次前向传播是比喻；先验法指「合乎理性即信」，不等于编造；六类对四法必有合并。「他律 / 自律」更险：靠外部神谕查证在字面上恰是更他律。修复：只保留「神谕 = 外在恒常物」一处类比；B/J 改用证言认识论词汇——Burge 的接受原则（无反驳理由即可接受证言）与 defeater 敏感度，「告知伪造 F 反升」即可精确表述为「defeater 不敏感」，比「习惯」更强也更难被挑刺。成本：零。

**P5（中）「吸收态」缺反驳判据。** 现有证据是 1.7B 二臂决策级（1307 行），该设定下结论近乎由构造保证。请预注册：(i) 结局奖励下 P(check) 停在内点；(ii) 高 π 层训练不降低低 π 层查证；(iii) 公开 RLVR 检查点谱系上 F 不随阶段上升；(iv) RL-ZVP / LENS 类零方差组取信号方法无需孪生即保住查证。另加随机奖励对照（Spurious Rewards，2025）排除「GRPO 放大预训练先验」。成本：预注册与谱系审计低，8B 验证高。

**P6（低）命名冲突。** "verification paradox" 已被 Hümmer 等（2026）、CMED 预印本、O'Keefe（2026）以不同含义使用，"rubber-stamping" 在 SelfCheck 等处是非正式用法。修复：首段精确定义并区分，或改称「结局奖励的基率陷阱」。成本：零。

### 若要我接收必须看到
1. 度量节无一处意图词，意图性解释集中于讨论节。
2. 基率剂量实验（≥2 家族、报伪推导率）与「吸收态」的预注册反驳判据，哪怕 8B 结果未到。
3. J 按证据条件拆分，H−F 与 F 并列报告。

### 我可能误读的地方
1. 我只读了阶段汇报，未见「账本六类」原始定义和「一半伪造」提示原文；若提示已把「一致」限定为「与物理事实一致」，P2 的措辞替代解释可削弱。
2. 若接种滞后在多种子上复现并有量级估计，P2 对「习惯」一词的异议只剩适用范围问题。

---

## 任务二 · 相关工作检索

关系标注：**被包含** = 本项目可视为其推广 / 直接沿用；**互补** = 不同对象或方法，可引用支撑；**构成威胁** = 削弱新颖性或与本项目结论冲突，必须正面处理。把握：**确定** = 检索到原文或官方摘要；**大概** = 凭记忆补充或仅见二手摘要；**待核** = 来源非同行评审 / 元数据不全。

### (a) "verification paradox" / "rubber-stamping" 在 LLM 验证与事实核查中的用法

| 题目 | 年份 | 一句话结论 | 关系 | 把握 |
|---|---|---|---|---|
| Hümmer et al., *AI, Metacognition, and the Verification Bottleneck* (arXiv 2601.17055) | 2026 | 人类用户在最难任务上最依赖 AI、却最不验证，作者称之为 "verification paradox"（人机协作语义） | 构成威胁（命名冲突，内容无关） | 确定 |
| CMED: *Cross-Model Epistemic Divergence* (making-minds.ai 预印本，§4.1 "The Verification Paradox") | 2025? | 弱模型核验正确推导 97% 准确，但对预写的说服性错误推导漏掉 20–40% | 互补 + 命名冲突 | 待核（非同行评审，年份不明） |
| yAI 研究笔记 *The Verification Paradox: Why Agents Cannot Automatically Validate Themselves*；O'Keefe, *The Verification Paradox* (SSRN 7155998) | 2025–2026 | 分别从多智能体自校验与学术诚信角度使用同一短语 | 命名冲突 | 待核 |
| Miao, Teh & Rainforth, *SelfCheck* (ICLR 2024) | 2023/24 | 直接让 LLM 审自己的步骤会近乎全部放行；用「重生成后比对」解耦 | 互补（推理期同症状，无训练动态） | 确定（「>90% 放行」数字见二手摘要，待核） |
| Stechly, Valmeekam & Kambhampati, *On the Self-Verification Limitations of LLMs on Reasoning and Planning Tasks* (arXiv 2402.08115) | 2024 | 自我验证不提升，换成可靠外部验证器才提升 | 互补（支持「神谕而非 LLM 评委」） | 确定 |
| Hong et al., *A Closer Look at the Self-Verification Abilities of LLMs in Logical Reasoning* (NAACL 2024) | 2024 | FALLACIES 数据集：LLM 难以识别自身谬误步骤 | 互补 | 确定 |
| Markowitz & Hancock, *Generative AI Are More Truth-Biased Than Humans* (J. Lang. & Soc. Psych.) | 2024 | LLM 真话偏倚 67–99%（GPT-4 中性提示 99%）；提示怀疑与给出基率可降低偏倚 | **构成威胁**：盖章在 RLVR 之前的基础模型就存在；且基率提示的方向与本项目「F 反升」相反 | 确定 |
| Barkett, Long & Thakur, *Reasoning Isn't Enough: Truth-Bias and Sycophancy in LLMs* (arXiv 2506.21561) | 2025 | 推理模型真话偏倚更低（59% vs 71%），仍高于人类；部分模型「真话准、假话不准」 | **构成威胁**：RLVR 训练的推理模型更不盖章，与「RLVR→盖章」预测表面相反，需按域区分 | 确定 |
| Schuster et al., *Towards Debiasing Fact Verification Models* (EMNLP 2019) | 2019 | FEVER 上只看主张不看证据的分类器即可匹敌证据模型；构造对称评测集 | 被包含（孪生 = 其对称集思想在物理域的推广） | 确定 |
| Zhou et al., *Variation in Verification* (arXiv 2509.17995) | 2025 | 生成式验证器对易题可靠、对强生成器的错误更难查 | 互补 | 确定 |

### (b) RLVR / GRPO 的奖励塌缩、零方差组、熵塌缩、吸收态

| 题目 | 年份 | 一句话结论 | 关系 | 把握 |
|---|---|---|---|---|
| Shao et al., *DeepSeekMath*（GRPO 原始论文, arXiv 2402.03300） | 2024 | 组内归一化优势；全对 / 全错组优势为零 | 被包含（理论起点） | 确定 |
| Liu et al., *Understanding R1-Zero-Like Training* (Dr. GRPO, arXiv 2503.20783) | 2025 | 去掉长度与 std 归一化；指出「难度偏差」：std 小的题被加权 | 互补（std 小但非零的高 π 题被加权，加速锁死；可引为机制） | 确定 |
| Yu et al., *DAPO* (arXiv 2503.14476) | 2025 | 动态采样过滤全对 / 全错组；clip-higher 对抗熵塌缩 | **构成威胁**：过滤后剩下的恰是混合奖励题，可能部分救回查证，必须作基线 | 确定 |
| Le et al., *No Prompt Left Behind* (RL-ZVP, arXiv 2509.21880) | 2025 | 零方差题并非无用，用熵引导的 token 级优势从中取信号 | **构成威胁**：若无需孪生即可保住查证，「只有神谕配对奖励能救」被削弱 | 确定 |
| Feng et al., *Don't Waste Mistakes* (LENS, arXiv 2510.08696) | 2025 | 全错组按置信度加罚，使负组有梯度 | 互补（针对全错组） | 确定 |
| EDGE-GRPO (arXiv 2507.21848)；AGPO (arXiv 2503.15952)；VL-Rethinker "vanishing advantages" (arXiv 2504.08837) | 2025 | 分别以引导纠错、损失掩码、选择性重放应对「优势塌缩 / 消失」 | 互补 | 确定 / 确定 / 大概（题名凭记忆） |
| Cui et al., *The Entropy Mechanism of RL for Reasoning LMs* (arXiv 2505.22617) | 2025 | 熵与性能呈 R=−a·e^H+b 的经验律，熵塌缩限制上限 | 互补 | 大概（未直接检索） |
| Hao et al., *Rethinking Entropy Interventions in RLVR* (STEER, arXiv 2510.10150)；Jin et al., *Revisiting Entropy in RL for LRMs* (arXiv 2511.05993) | 2025 | 正优势 token 是熵塌缩主因；token 级重加权缓解 | 互补 | 确定 |
| Wang, *The Dark Room in the Reward Channel* (arXiv 2607.21273) | 2026 | 74 个预注册实验：GRPO 的 std 归一化使稠密预测奖励坍缩进「退化吸收态」；组内方差轨迹决定危险 | **构成威胁**：已用「absorbing state」+ 预注册设计描述 GRPO 塌缩，必须引用并区分（其塌缩源于 shaping，本项目源于基率） | 确定（单作者预印本，发表地待核） |
| Xu, *The Policy Cliff* (arXiv 2507.20150) | 2025 | 奖励→最优策略映射在最优动作并列时不连续；熵正则恢复连续 | 互补（c*(π) 处「查 / 盖」并列的理论框架） | 确定 |
| Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR* (arXiv 2506.10947) | 2025 | 随机奖励也能让 Qwen 提分；GRPO 裁剪偏差放大预训练高先验行为 | **构成威胁**：盖章增长可解释为「放大预训练先验」而非奖励结构，需随机奖励对照臂 | 确定 |
| Song, Shi & Zhao, *The Hallucination Tax of Reinforcement Finetuning* (EMNLP Findings 2025, arXiv 2505.13988) | 2025 | RFT 使不可答题拒答率下降 >80%；混入 10% 不可答题即恢复 | **构成威胁**（同形论点已在数学域发表：结局 RL 消灭一种认识行为、少量混合数据可恢复）+ 互补（接种 / TAPO-ε 的先例） | 确定 |
| Huang et al., *Pitfalls of Rule- and Model-based Verifiers* (arXiv 2505.22203) | 2025 | 规则验证器假阴性随策略变强而增；模型验证器被空符号 / 乱码攻破 | 互补（界定「验证器出错」与本项目「第三种饱和」的区别） | 确定 |

### (c) LLM 评委对编造推理链的识别能力、过程奖励模型的脆弱性

| 题目 | 年份 | 一句话结论 | 关系 | 把握 |
|---|---|---|---|---|
| Zheng et al. (Qwen), *ProcessBench* (ACL 2025, arXiv 2412.06559) | 2024/25 | 3,400 条人工标注首错步；PRM 难以泛化到竞赛题，不如提示式批评模型 | 互补（MV 层已用作自然错误；注意其标注为人工非神谕） | 确定 |
| Zhang et al. (Qwen), *The Lessons of Developing PRMs in Mathematical Reasoning* (ACL Findings 2025, arXiv 2501.07301) | 2025 | MC 估计标签噪声大；BoN 评测使 PRM 退化为结局打分 | 互补（解释 PRM 为何治不了盖章） | 确定 |
| *FaithCoT-Bench* (arXiv 2510.04040) | 2025 | LLM 评委识别不忠实 CoT 的 F1 65–77，是现有最佳；更强模型的 CoT 更具欺骗性 | **构成威胁**：其评委表现远高于本项目 AUROC≈0.5，需用「物理瓶颈事实评委无法核算」解释差距 | 确定 |
| Mittal & Arike, *C2-Faith* (arXiv 2603.05167) | 2026 | 评委能察觉有错却难定位；系统性高估推理完整度 | 互补 | 确定 |
| Sadeddine & Suchanek, *Verifying the Steps of Deductive Reasoning Chains* (ACL Findings 2025) | 2025 | LLM 查演绎步骤「尚可」，但主张用透明的神经符号验证器 | 互补 | 确定 |
| Baker et al. (OpenAI), *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* (arXiv 2503.11926) | 2025 | CoT 监控器作奖励后，模型学会写无害外观的 CoT 继续作弊 | **构成威胁**（对 TAPO-RA）：审计写出的推导作为奖励，可能训出「过审计的推导」而非真查证；需严格账本格式与神谕而非 LLM 审计 | 确定 |
| *LLMs Cannot Reliably Judge (Yet?)* (arXiv 2506.09443) | 2025 | LLM 评委对对抗操纵不鲁棒 | 互补 | 大概（作者未核） |
| *A Comprehensive Anatomy of Human and DeepSeek-R1 Mathematical Reasoning* (arXiv 2606.07410) | 2026 | 失败轨迹中 75% 的「反思」是表面校验（performative reflection），不触及上游错误 | 互补（RLVR 模型「伪查证」的独立行为证据） | 确定（发表地待核） |
| CMED 预印本（见 (a)） | 2025? | 弱验证器对说服性错误推导漏检 20–40% | 互补 | 待核 |

### (d) 带成本的验证 / 工具调用 RL、弃权奖励、Costly State Verification

| 题目 | 年份 | 一句话结论 | 关系 | 把握 |
|---|---|---|---|---|
| Townsend, *Optimal Contracts and Competitive Markets with Costly State Verification* (J. Econ. Theory) | 1979 | 随机验证程序可帕累托优于确定性验证；审计概率 + 惩罚构成威慑 | **被包含 / 构成威胁**：命题 3 的 q > (π+c−a)/(κa) 是其审计威慑不等式的特例，TAPO-RA 的理论新意限于 LLM 实例化，必须引用 | 确定 |
| Mookherjee & Png, *Optimal Auditing, Insurance, and Redistribution* (QJE) | 1989 | 随机审计是最优的 | 被包含 | 大概 |
| Greenblatt et al., *AI Control: Improving Safety Despite Intentional Subversion* (ICML 2024) | 2024 | 2% 审计预算 + 可信监控即可大幅提升安全 | 互补（审计预算框架；TAPO-RA 把审计从部署协议搬进训练奖励） | 确定 |
| *Strategic Classification Has a Missing Lever: Audit Risk* (arXiv 2609.22534) | 2026 | 承诺的审计概率 × 惩罚构成 ML 中的威慑杠杆；惩罚有上限时 p 不能趋零 | 互补 / 构成威胁（形式化了同一 p·Δ 威慑；κ 上限的讨论可直接移植） | 确定（发表地待核） |
| Kirchner et al., *Prover-Verifier Games Improve Legibility of LLM Outputs* (arXiv 2407.13692) | 2024 | 只优化答案正确会降低可核验性；对小验证器训练恢复可读性 | 互补（「结局奖励损害可验证性」同题） | 确定 |
| SEM (arXiv 2505.07903)；MMSearch-R1 (arXiv 2506.20670)；Light-Search (ICLR 2026 撤稿) | 2025 | 用 GRPO + 搜索惩罚学「何时搜」，减少工具调用；朴素惩罚不稳定 | 互补（成本敏感工具调用，方向相反：减少而非保住调用） | 确定 |
| *Don't Lie to Your Friends: Learning What You Know from Collaborative Self-Play* (arXiv 2503.14481) | 2025 | 以 Score − δ·Effort 奖励学会何时用工具、何时弃权 | 互补 | 确定 |
| Damani et al., *Beyond Binary Rewards* (RLCR, ICLR 2026, arXiv 2507.16806) | 2025/26 | 二元奖励损害校准；加 Brier 项同时提升准确与校准 | **构成威胁**：无需孪生 / 神谕的替代方案——校准奖励会让「盖章且置信 0.95」合法、「盖章且编账本置信 1.0」受罚；需正面比较 | 确定 |
| Wei et al., *TruthRL* (arXiv 2509.25760) | 2025 | 三元奖励（正确 +1 / 弃权 0 / 幻觉 −1）降幻觉 28.9% | **构成威胁**：TAPO-RA 的「弃权不审」与三元奖励同构，必须比较 | 确定 |
| *Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward* (FiSCoRe, arXiv 2510.24020) | 2025 | 按样本级语义置信度训练弃权 | 互补 | 确定 |
| Srinivasan, *Learning When to Use Tools: Cost-Aware RL*（Stanford CS224R 课程报告） | 2025 | 玩具环境：结局奖励→过度调用，成本奖励→选择性调用，shaping 被 hack | 互补 | 待核（课程报告） |

### (e) 用天文 / 历史记录做 LLM 评测或事实核查的数据集

直接对应（以古代日食 / 天象记录构建的 LLM 评测基准）：**未检索到**。最接近的分三类：

| 题目 | 年份 | 一句话结论 | 关系 | 把握 |
|---|---|---|---|---|
| *Astro-QA* (Scientific Data) | 2025 | 3,082 题天文学 QA（含天文学史），中英双语 | 互补（学科问答，非记录核验） | 确定 |
| *HistBench / HistAgent* (arXiv 2505.20246) | 2025 | 414 道专家历史题、29 种语言；通用 agent 表现差 | 互补 | 确定 |
| Miao, Fu & Wei, *SPAN: Cross-Calendar Temporal Reasoning* (AAAI) | 2026? | 六种历法互转，LLM 平均 34.5%；工具化代码 agent 95% | 互补 / 构成威胁（「历法换算难、工具可解」已被先发；本项目 AR 层 16% 换算精确率应引用对照） | 大概（年份 / 届次待核） |
| *CTM: Temporal Reasoning Across Chinese Dynasties* (arXiv 2502.16922)；*C3Bench* (arXiv 2405.17732) | 2025 / 2024 | 中国朝代时间推理；古文理解五任务 | 互补 | 确定 |
| Stephenson, Morrison & Hohenkerk, *Measurement of the Earth's Rotation: 720 BC to AD 2015* (Proc. R. Soc. A) + 补充表 S1–S14 | 2016 | 汇编巴比伦 / 中国 / 阿拉伯 / 欧洲日月食记录用于 ΔT；含「被拒记录」清单 | **被包含 / 构成威胁**：天象记录的物理可成立性判定在此已系统做过（含剔除清单）；前 700 年 ΔT 不确定度影响巴比伦层标签 | 确定 |
| 刘次沅, *The regular records of solar eclipse in ancient China and a computer readable table* (Arch. Hist. Exact Sci. 59) | 2005 | 938 条中国正史日食记录的机读表 | 被包含（先行机读表，须引用） | 确定 |
| 崔健骅、黎耕、赵永恒，《中国古代天象记录数据库的设计与实现》(ChinaXiv 202301.00126) | 2023 | 面向文档的古代天象记录全文检索数据库 | 构成威胁（对「首个数据库」类表述） | 确定 |
| Hayakawa et al., *Analyses of the Ancient Chinese Report on the Total Solar Eclipse in 709 BCE* (ApJL 995 L1) | 2025 | 观测站点坐标修订改变 ΔT 约束 | 互补（神谕的「站点」假设需有不确定度） | 确定 |
| Steele, *Observations and Predictions of Eclipse Times by Early Astronomers* | 2000 | 巴比伦日志中预报与观测并存 | 互补（BA 层 omitted 切片的史学依据） | 大概 |
| Parker & Dubberstein, *Babylonian Chronology*；Sachs & Hunger, *Astronomical Diaries* (ADART) | 1956 / 1988– | 巴比伦历表与日志标准本 | 被包含（已用作锚点 / 文本源） | 大概 |

### (f) Chain-of-thought 忠实性 / 不忠实推理 / 事后合理化的测量

| 题目 | 年份 | 一句话结论 | 关系 | 把握 |
|---|---|---|---|---|
| Turpin et al., *Language Models Don't Always Say What They Think* (NeurIPS 2023) | 2023 | 偏置特征改变答案而 CoT 不提及；模型为错误答案编造合理化 | 互补（事后合理化的经典操作化） | 确定 |
| Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning* (arXiv 2307.13702) | 2023 | 截断 / 注错 CoT 看答案是否变；越大模型越不忠实 | 互补（干预式忠实度度量，可作伪推导率的平行指标） | 确定 |
| Chen et al. (Anthropic), *Reasoning Models Don't Always Say What They Think* (arXiv 2505.05410) | 2025 | 提示使用率高但口头承认 <20%；结局 RL 初升后在 20–28% 平台；奖励 hack 口头化 <2% | **构成威胁 / 互补**：「RL 增加作弊而不增加口头化」是伪推导的通用形式；其 hint 法是另一套度量，需说明神谕法为何更强（实例级、无需植入提示） | 确定 |
| Arcuschin et al., *Chain-of-Thought Reasoning in the Wild Is Not Always Faithful* (arXiv 2503.08679) | 2025 | 用 Yes/No 对偶问题测「隐式事后合理化」 | 互补 | 确定 |
| *FaithCoT-Bench* (2025)；*C2-Faith* (2026) | — | 见 (c) | 互补 | 确定 |
| Paul et al., *Making Reasoning Matter* (FRODO) | 2024 | 用反事实干预训练更忠实的 CoT | 互补 | 大概 |

### 「构成威胁」条目汇总（供作者优先处理）
1. Markowitz & Hancock 2024 / Barkett et al. 2025：真话偏倚在基础模型与推理模型中早已存在，且基率提示降低接受率、推理模型更少盖章——与「RLVR 导致盖章」「告知伪造 F 反升」方向冲突，需谱系审计与剂量实验正面回应。
2. Song et al. 2025（Hallucination Tax）：「结局 RL 消灭认识行为、少量混合数据恢复」的同形论点已在数学域发表；需定位为「神谕测量 + 跨层迁移 + 相图」的推广。
3. Shao et al. 2025（Spurious Rewards）：GRPO 放大预训练先验的替代解释；需随机奖励对照臂。
4. Wang 2026（Dark Room）：已用「吸收态」+ 预注册描述 GRPO 塌缩；需引用并区分成因。
5. Le et al. 2025（RL-ZVP）与 DAPO 动态采样：零方差组可取信号 / 可过滤，削弱「只有神谕配对奖励能救」；需作基线。
6. Damani et al. RLCR（ICLR 2026）与 Wei et al. TruthRL 2025：无神谕的校准 / 三元奖励替代方案；TAPO-RA 的弃权结构与之同构，需比较。
7. Townsend 1979：随机审计优于确定审计的经典结果，命题 3 为其特例，理论新意需收窄表述。
8. Baker et al. 2025：以审计推导为奖励可能训出「过审计的推导」；TAPO-RA 需论证神谕审计不可被措辞绕过。
9. FaithCoT-Bench 2025：LLM 评委 F1 65–77 与本项目 AUROC≈0.5 相冲突，需以「物理瓶颈事实不可由评委核算」解释。
10. Stephenson et al. 2016 / 刘次沅 2005 / 崔健骅等 2023：天象记录物理核验与机读表、数据库已有先行工作，数据新颖性表述需收窄；前 700 年 ΔT 不确定度需进入标签不确定度。
11. SPAN（AAAI 2026?）：历法换算难且工具可解已先发。
12. "verification paradox" 命名冲突（Hümmer 2026、CMED、O'Keefe 2026）。
