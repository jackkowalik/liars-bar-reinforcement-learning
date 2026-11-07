# Liar's Bar 排位机器人

[English](README.md) | 中文 | [Русский](README.ru.md)

一个在 Liar's Bar 的排位队列中玩 Liar's Deck（恶魔模式）的机器人，只使用桌上任何真人玩家都能看到的信息。它在大约 24
小时内从 0 分爬到了排行榜（约 3000 分）。

## 快速开始

要在游戏里运行机器人（而不是只跑模拟器），请看 [INSTALL.md](INSTALL.md)；预编译的 `liars.exe`、mod DLL 和权重文件在
Releases 页面。

```
git clone https://github.com/jackkowalik/liars-bar-deck-reinforcement-learning.git
cd liars-bar-deck-reinforcement-learning/rust
cargo build --release
target/release/liars fiteval --net ../weights/devil_mlp --opp rankedfit --players 4 --games 3000
```

这会让实际打过排位的网络对阵根据真人日志拟合出的对手群体，并输出胜率、每局按原因分类的扣扳机次数以及出牌构成。
`--players 3` 或 `2` 对应更小的桌子，`--seed N` 改变发牌顺序。

```
python -c "import json; r=json.loads(open('data/plays_public.jsonl').readline()); print(r['table'], r['claimer_hand'], r['thrown'], r['was_bluff'])"
```

环境要求：模拟器和顾问程序需要 stable 版 Rust。`nn/` 里的训练脚本需要 Python 3.11、PyTorch 和 `maturin`（用来编译
`pyliars/`）。`mod/` 里的 mod 需要 .NET 6 SDK、MelonLoader 和 Steam 版 Liar's Bar，项目文件会从默认的 Steam 路径解析游戏程序集。
实时工具（顾问、mod、看门狗）仅支持 Windows，并以 `C:\liars` 作为工作目录；模拟器、评测工具和数据集没有这些依赖。

## 组成部分

- **游戏引擎与模拟器**（`rust/src/engine.rs`）：带恶魔模式的 Liar's Deck，六膛左轮、强制质疑、恶魔契约，2 到 4 个座位。
  桌面机上每小时可跑数百万局。
- **策略网络**（`rust/src/nn.rs`、`nn/`）：一个用 PPO 在自博弈中对阵脚本对手联赛训练出的 MLP，以及一个读取整回合历史
  （含座位身份）的 transformer，先从 MLP 蒸馏再用 PPO 训练。两者都导出为扁平权重文件，由 Rust 代码无依赖地推理。
- **游戏端 mod**（`mod/`，MelonLoader，C#）：为每局游戏写出公开事件日志（座位、段位、宣称、思考时间、质疑、扣扳机、
  我们自己的手牌），并通过游戏自身的输入方法执行顾问选定的动作。
- **顾问**（`rust/src/watch.rs`）：跟踪日志，重建对局状态，运行策略，打印推理过程，并把动作写给 mod。每个策略开关
  都是一个环境变量。
- **数据流水线**（`nn/reveals.py`、`data/`）：在对局结束后把每一次对手出牌标注为真话或虚张，从而为所有对手出牌提供
  真实标签（约 10,000 次出牌和 4,400 次被质疑的宣称）。标签只离线使用，实时顾问从不使用。
- **拟合的对手群体**（`research/fit_population.py`、`agents::FitHuman`）：根据带标签的日志拟合虚张率、质疑率、出牌张数
  和清手习惯，并带有每局的随机变化。
- **评测工具**（`liars fiteval`）：任意策略变体对阵拟合群体，几秒钟内跑完数千局，A/B 使用相同随机种子。每个改动上线前
  都要先过这一关。
- **实时测量**（`research/scoreboard.py`）：从游戏自己的分数计数器读取胜率，按桌子人数和对手段位拆分，每个版本的对局分开
  统计，样本量预先登记。

## 方法

1. 在模拟器中通过自博弈训练策略。
2. 上线实战，记录一切，从日志中标注对手的出牌。
3. 用标签拟合对手群体，放进模拟器。
4. 每个候选改动在触碰游戏之前，先用固定随机种子对阵该群体评测。
5. 通过评测的改动上线实战，一次只改一个变量，预先登记样本量，设定停止规则，用记分板把每个版本的对局分开。

## 结果

从游戏分数计数器得到的实战胜率：在 2、3、4 人桌共 130 局排位中为 47-50%。在观察到的桌型分布下，随机出牌的胜率是 31%。
在评测工具里对阵拟合群体，同一策略的胜率为 48%（4 人）、51%（3 人）和 67%（1v1）。

机器人死亡原因（日志中的 240 次死亡）：

| 原因 | 次数 |
|---|---|
| 手里有匹配牌却虚张，被质疑（理论上可避免） | 74 |
| 部分虚张（真牌假牌混出），被质疑 | 60 |
| 主动质疑但判断错误 | 44 |
| 对方清手且为真话时的强制质疑（1v1） | 31 |
| 没有匹配牌只能虚张，被质疑 | 20 |

日志中的其他测量：

- 人类大约质疑 22% 的宣称，与出牌者的思考时间无关；他们质疑机器人真话出牌（37%）的频率高于质疑其虚张（24%）。
- 对手段位不影响机器人的虚张被质疑的频率。
- 匹配不按分数进行：对手段位分布是平的，所以在固定奖惩下，分数随对局数线性上升。
- 机器人自己的质疑有 37% 正确：对回合首次宣称的质疑 62%，对回合第 3 到第 5 次宣称的质疑 17-28%，强制质疑 0%。

## 相关工作

- 斯坦福 CS224R 项目 *Bluff and Learn*：Liar's Bar 上的 CFR 与 NFSP 智能体；表格式 CFR 在简化版本中有效，带稠密每回合奖励
  的 NFSP 在完整规模下泛化更好。
- *Outbidding and Outbluffing Elite Humans*（arXiv 2511.03724，2025）：Solly，一个自博弈 R-NaD 智能体，在简化版 Liar's Poker
  中达到顶级人类水平。它的虚张频率远高于人类，这与"不可被利用时大量虚张是正确的"一致，也与我们的发现一致：问题不在于
  虚张率，而在于虚张的对象。
- *Social Behaviour and Strategic Adaptation of LLMs in Multiplayer Sequential Games*（NeurIPS 2025）：用一个基于提示词的
  Liar's Bar 变体研究 LLM 智能体。
- Neller 与 Lanctot，*An Introduction to Counterfactual Regret Minimization*；Neller 与 Hnath 关于 Dudo 的不完美记忆抽象：
  虚张类游戏的经典处理，以及精确 CFR 为何无法扩展到这个游戏。

## 范围与行为

机器人只依据公开信息行动：任何玩家都能看到的宣称、张数、时间和结果，加上自己的手牌。游戏客户端会从服务器收到每个
玩家的手牌；这个泄漏被用来在事后标注日志数据，并被刻意排除在实时顾问之外；带标签的手牌在 `data/` 中发布，读取它们的代码
没有发布。机器人在排位中与不知道自己在和机器对局的真人对战，这几乎肯定违反游戏规则。

## 数据集与权重

`data/plays_public.jsonl` 包含来自 522 局的 15,314 次宣称，每条都标注了真话或虚张，并带有宣称发生时每个座位的隐藏手牌、
宣称者的手牌和实际打出的牌（来自更早 194 局的 5,616 条未标注宣称在 `data/plays_unlabelled.jsonl`）。`data/calls_public.jsonl`
包含 4,439 次被质疑的宣称，带有翻开的牌和同样的隐藏手牌。玩家名在两个文件中用一致的化名替代；见 `data/SCHEMA.md`。
`weights/` 存放 Rust 代码运行的网络：`devil_mlp`（实际打排位的那个）、`devil_tnet_r1v1`（transformer）和 `basic_ppo2`
（非恶魔模式桌）。

## 目录结构

```
rust/        引擎、模拟器、策略、顾问、评测工具
pyliars/     引擎的 Python 绑定，训练脚本使用
nn/          训练（PPO、蒸馏）、群体拟合
weights/     导出的网络
mod/         MelonLoader mod：事件日志、自动排队状态机、动作执行
tools/       看门狗、顾问启动器、空闲时重启
research/    群体拟合、记分板、质疑评分、虚张定价、数据集导出
data/        发布的数据集及其说明
```

## 许可

MIT。`data/` 中的数据集以同一许可发布。
