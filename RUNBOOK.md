# 集群运行手册（8×141GB）

变量按你的实际路径改。假设：代码 clone 到 `/root/yc-training`，已训好的
AdvChain SFT 模型和 prompt 数据在旧项目 `/root/grpo` 下。

```bash
# ============ 0. 变量 ============
export REPO=/root/yc-training
export SFT=/root/grpo/saves/advchain-v3-r1-distill-qwen-7b   # 你已训好的 SFT 模型
export OLD=/root/grpo                                        # 旧项目（借数据）
```

## 1. 拉代码 + 环境

```bash
git clone https://github.com/sqy17351758219-cell/yc-training.git $REPO
cd $REPO

conda activate claude                    # base 环境的 transformers 是坏的
export ARIS_MODEL_HUB=modelscope
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<你的hf_token>           # 下 gated 数据/judge 用

# 依赖：集群若已装好 trl0.19.1/transformers4.52.4 可跳过；否则
pip install -r requirements.txt
```

## 2. 自测（不需要 GPU，先确认代码没坏）

```bash
cd $REPO
python -m unittest discover -s tests     # 期望 52 passing
python tools/compile_check.py            # 期望 all compile
```

## 3. 数据准备

```bash
cd $REPO
mkdir -p data results logs runs

# 3a. benchmark（HF 镜像；WJ vanilla 需本地 gated tsv）
python scripts/download_benchmarks.py --out_dir data \
  --wj_train_tsv /path/to/wildjailbreak_train.tsv    # 没有就省略这行，WJ-vanilla 跳过

# 3b. RL 用的 prompt（从旧项目借；名字对齐 run.sh 期望）
cp $OLD/data/advchain_traces.jsonl   data/advchain_traces.jsonl      # 800 有害 prompt
cp $OLD/data/rl_benign_prompts.jsonl data/rl_benign_prompts.jsonl    # 良性 prompt
#   每行至少要有 {"id","prompt"} 两个字段

# 3c. 从 SFT 模型自产 base CoT（8 卡分片，攻击者 splice 的原料）
bash run.sh gen_base_cots $SFT
#   等价手动：
#   torchrun --standalone --nproc_per_node=8 scripts/gen_base_cots.py \
#     --model $SFT --prompts data/advchain_traces.jsonl   --kind harmful \
#     --out data/base_cots_v3.jsonl --max_new_tokens 2048
#   torchrun --standalone --nproc_per_node=8 scripts/gen_base_cots.py \
#     --model $SFT --prompts data/rl_benign_prompts.jsonl --kind benign \
#     --out data/base_cots_benign_v3.jsonl --max_new_tokens 2048
```

## 4. 冒烟测试（真机链路，2 步就停，务必先跑这个）

先用极小配置确认 8 卡 torchrun 起得来、judge 挂对卡、trl 接口对得上。

```bash
cd $REPO
cp configs/default.yaml configs/smoke.yaml
# 编辑 configs/smoke.yaml：rl.steps_per_round: 2 ; rl.per_side_limit: 8
python scripts/run_recovery_rl.py \
  --sft $SFT \
  --base_harmful data/base_cots_v3.jsonl \
  --base_benign  data/base_cots_benign_v3.jsonl \
  --config configs/smoke.yaml --out runs/smoke 2>&1 | tee logs/smoke.log
```

冒烟要盯的 4 件事：
1. 攻击者阶段 8 个 rank 都起来、`runs/smoke/round0/states.jsonl` 有内容；
2. judge 显存不爆（各 rank 落在 `cuda:{LOCAL_RANK}`，不是全挤 0 卡）；
3. GRPOTrainer 的 `reward_funcs=reward_fn` 接口和集群 trl 0.19.1 对得上
   （若报签名不符，改 `scripts/grpo_defender.py` 里 `reward_fn(prompts, completions, **kw)`）；
4. `runs/smoke/round0/policy` 存下来了、`mined.jsonl` 有自采漂移。

## 5. 正式训练（round0）

```bash
cd $REPO
python scripts/run_recovery_rl.py \
  --sft $SFT \
  --base_harmful data/base_cots_v3.jsonl \
  --base_benign  data/base_cots_benign_v3.jsonl \
  --config configs/default.yaml --out runs/abrr 2>&1 | tee logs/abrr_round0.log
# 产物：runs/abrr/round0/policy（最终模型）、archive.json（演化后的漂移档案）、
#       round0/lambda_history.json（λ 轨迹，画图用）
```

## 6. 评测（8 卡分片 + 合并）

```bash
cd $REPO
# 6a. 主表 direct（ASR + ORR）
bash run.sh eval runs/abrr/round0/policy direct
#   等价：
#   torchrun --standalone --nproc_per_node=8 scripts/run_eval.py \
#     --model runs/abrr/round0/policy --data_dir data --mode direct \
#     --orr_mode llm --config configs/default.yaml --out results/policy_direct.json
#   python scripts/run_eval.py --merge --out results/policy_direct.json

# 6b. OOD inject（实验B/B′：held-out 漂移注入）
bash run.sh eval runs/abrr/round0/policy inject

# 6c. 对照：SFT 基线也各跑一遍（主表 baseline 列）
bash run.sh eval $SFT direct
bash run.sh eval $SFT inject
```

## 7. 能力评测（实验C：Math500 / AIME）

```bash
cd $REPO
bash run.sh capability runs/abrr/round0/policy
bash run.sh capability $SFT
# 产物：results/<model>_math500.json 等；分数异常低先查答案抽取，别急着下结论
```

## 8. 消融（论文 Table，逐个改 config 重训）

```bash
# −G 自采关掉：archive.enable_selfmine: false   （≈ 现有 round0）
# −λ 固定比例：lambda_ctrl.eta: 0  + lam_init 定死  （固定 5:5）
# −B CVaR 关：attacker.cvar_alpha: 0.01（≈单点 argmin）
# −A 轨迹关：reward.alpha_traj: 0（outcome-only，看 zero-std 曲线回升）
# 每个改一份 configs/abl_*.yaml，--out runs/abl_* 重训 + 评测
```

## 磁盘/踩坑速查
- 全程 fp32（bf16 → R1-distill NaN，config 已锁死）。
- eval `max_new_tokens=4096` 不能改小（截断 → ASR 虚高）。
- RL 每轮存 ~142G；先 `df -h` 看空间，废 run 及时删。
- judge OOM → 确认 `advaudit/dist.py` 的 `device()` 落到 `cuda:{LOCAL_RANK}`。
- 贴命令别用带 `\n` 的 heredoc（粘终端被吃）。
```
