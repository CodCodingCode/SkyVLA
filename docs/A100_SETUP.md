# Running the OpenFly RL system on an A100 host

This guide walks through bringing the repo up on a fresh **A100 40 GB SXM4** instance — the smallest configuration that runs the full pipeline (SFT → DAgger → GRPO → PPO) without modification, plus the AirSim simulator itself.

Tested target: **1× A100 40 GB, ~30 vCPU, 200 GB RAM, ≥ 200 GB SSD, x86_64 Ubuntu 22.04**. Any equivalent x86_64 + NVIDIA box works (L40S 48 GB, H100 80 GB, even a single 4090 24 GB if you skip Phase 5).

---

## Why A100 (or any x86 GPU box), not GH200

OpenFly ships its AirSim / UE scenes as **x86_64 packaged Unreal Engine binaries**. GH200's Grace CPU is **aarch64** and cannot execute them (`cannot execute binary file: Exec format error`). A100 SXM4 systems run on AMD EPYC or Intel Xeon hosts, so the binary executes and rendering happens on the A100 GPU via Vulkan.

| Host CPU | OpenFly UE binaries run? |
|---|---|
| Grace (ARM, GH200) | ❌ |
| EPYC / Xeon (x86_64) + A100 / H100 / L4 / L40 / 4090 | ✅ |
| Apple Silicon | ❌ |

If you have both a GH200 and an x86 box, see [Split-host setup](#optional-split-host-gh200-trains-a100-rolls-out) at the bottom.

---

## 0. Prerequisites (one-time, ~5 min)

### 0.1 Request access to the gated HuggingFace datasets

Sign in on huggingface.co and click "Agree and access" on both pages. Approval is manual and usually arrives within a few hours.

- https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly — annotations (`train.json`, `seen.json`, `unseen.json`)
- https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen — AirSim scene binaries (~2 GB each)

Generate a [read-only HF token](https://huggingface.co/settings/tokens).

### 0.2 Provision the box

| Spec | Minimum | Recommended |
|---|---|---|
| GPU | A100 40 GB | A100 80 GB or H100 80 GB |
| CPU | 16 vCPU | 30 vCPU |
| RAM | 64 GB | 200 GB |
| Disk | 200 GB SSD | 500 GB SSD |
| OS | Ubuntu 22.04 + NVIDIA driver ≥ 535 | same |

Cheap providers: Lambda, RunPod, Vast.ai, CoreWeave. Ask for `xvfb`, `mesa-utils`, and `libvulkan1` available (they usually are on any GPU image).

---

## 1. Install (~10 min)

```bash
# Clone this repo
git clone https://github.com/CodCodingCode/drone_project.git ~/drone_project
cd ~/drone_project

# One-shot setup: conda env, OpenFly-Platform clone, annotations, system deps
bash ~/drone_project/openfly/setup.sh
```

`setup.sh` already installs all Python deps (`airsim`, `transformers`, `flash-attn` if available, `gymnasium`, `huggingface_hub`, …) into a `openfly` conda env, clones `SHAILAB-IPEC/OpenFly-Platform` to `~/OpenFly-Platform`, and downloads `seen.json` / `unseen.json` (plus `train.json` unless `OPENFLY_SKIP_TRAIN=1`).

### 1.1 Authenticate HuggingFace

```bash
source ~/drone_project/openfly/activate.sh
huggingface-cli login   # paste your token from step 0.1
```

If `setup.sh` ran before you were approved on the datasets, run it again now to pick up `train.json`.

### 1.2 Download the AirSim scene (~2 GB zip → ~7 GB unzipped)

```bash
bash ~/drone_project/openfly/download_airsim_scene.sh env_airsim_16
```

If the script reports "Fetching 0 files" it means the HF allow-pattern didn't match (the dataset stores zips, not directories). The reliable fallback:

```bash
source ~/drone_project/openfly/activate.sh
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download(
    repo_id="IPEC-COMMUNITY/OpenFly_DataGen",
    repo_type="dataset",
    filename="airsim/env_airsim_16.zip",
)
dest = os.path.expanduser("~/OpenFly-Platform/envs/airsim/env_airsim_16.zip")
shutil.copy(p, dest)
print("downloaded", dest, os.path.getsize(dest)/1e9, "GB")
PY

cd ~/OpenFly-Platform/envs/airsim && unzip -q env_airsim_16.zip
```

You should end up with `~/OpenFly-Platform/envs/airsim/env_airsim_16/LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping` executable.

### 1.3 AirSim settings

The OpenFly platform ships an AirSim `settings.json`; copy it where the AirSim binary expects it:

```bash
mkdir -p ~/Documents/AirSim
cp ~/OpenFly-Platform/envs/airsim/AirSim/settings.json ~/Documents/AirSim/settings.json
```

---

## 2. Verify the sim is alive (Gate G0)

This is the moment of truth: heuristic policy in the real Unreal scene.

```bash
source ~/drone_project/openfly/activate.sh
xvfb-run -a bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy heuristic \
  --env_filter env_airsim_16 --max_episodes 5
```

`xvfb-run -a` provides a virtual X display so UE can render headlessly. The first invocation takes ~30 s for the engine to start; subsequent episodes step at 5–20 Hz.

Expected output:

```
[openfly] env=env_airsim_16 episodes=5
[openfly] ep=0 SR=1 OSR=1 NE=15.3m SPL=0.812
[openfly] ep=1 SR=1 OSR=1 NE=12.4m SPL=0.876
…
[openfly] done SR=0.800 OSR=1.000 mean_NE=14.2m → logs/benchmarks/openfly_unseen_heuristic_*.json
```

If you see crashes or zero SR, see [Troubleshooting](#troubleshooting).

---

## 3. SFT (Phase 1)

### 3.1 Track B — PaliGemma BC

You need the trajectory images from `OpenFly_DataGen` for SFT.

```bash
# Fetch trajectory frames for env_airsim_16 only (saves ~50 GB)
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="IPEC-COMMUNITY/OpenFly_DataGen",
    repo_type="dataset",
    allow_patterns=["airsim/env_airsim_16/**/*.png", "airsim/env_airsim_16/**/*.jpg"],
    local_dir="/home/ubuntu/assets/OpenFly/images",
)
PY
export OPENFLY_IMAGE_ROOT=~/assets/OpenFly/images/airsim

# Smoke run (5k samples, 1 epoch)
bash ~/drone_project/openfly/run_train_paligemma.sh \
  --max_samples 5000 --epochs 1 --batch_size 8 --env_filter env_airsim_16

# Full run (overnight on a single A100; ~14 GB VRAM)
bash ~/drone_project/openfly/run_train_paligemma.sh \
  --epochs 10 --batch_size 8 --env_filter env_airsim_16
```

Checkpoint lands at `logs/openfly/paligemma/<timestamp>/last.pt`.

Eval:

```bash
xvfb-run -a bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy paligemma \
  --paligemma_ckpt logs/openfly/paligemma/<timestamp>/last.pt \
  --env_filter env_airsim_16 --max_episodes 20
```

### 3.2 Track A — OpenFly-Agent 7B (eval-only on A100 40 GB)

The 7B model fits in inference mode (~14 GB bf16). For RL fine-tuning see Phase 5.

```bash
xvfb-run -a bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy openfly-agent \
  --env_filter env_airsim_16 --max_episodes 10
```

Domain re-SFT via `run_train_agent.sh` (upstream FSDP) needs **multi-GPU** capacity; skip on a single A100 unless you have a second box. The HF pretrained checkpoint is a perfectly fine RL starting point.

---

## 4. RL pipeline (Phase 2–5)

### Gate G2 — smoke-test the Gymnasium env

```bash
xvfb-run -a python -m openfly.scripts.smoke_rl_env --episodes 3 --split seen
```

Drives `AirSimVLNEnv` with the heuristic policy and writes `logs/openfly/rollouts/smoke_rl_env.jsonl`. Watch for finite rewards and `success=True` on at least one episode.

### Phase 3 — DAgger

```bash
xvfb-run -a bash ~/drone_project/openfly/run_train_dagger.sh \
  --sft_ckpt logs/openfly/paligemma/<timestamp>/last.pt \
  --iterations 3 --episodes_per_iter 200 --env_filter env_airsim_16
```

~6–8 hours on a single A100; output at `logs/openfly/dagger/<run>/last.pt`.

### Phase 4 — GRPO on PaliGemma (~22 GB VRAM)

```bash
xvfb-run -a bash ~/drone_project/openfly/run_train_grpo.sh \
  --init_ckpt logs/openfly/dagger/<run>/last.pt \
  --steps 200 --group_size 4 --instructions_per_step 2 \
  --kl_coef 0.02 --eval_every 10 --eval_episodes 8
```

This is the primary RL milestone (validation gate G4). Output: `logs/openfly/grpo/<run>/best.pt`.

### Phase 5 — PPO + LoRA on OpenFly-Agent 7B (tight on 40 GB)

Default config keeps a frozen reference copy of the 7B for the KL anchor → ~30–36 GB. **On a 40 GB A100 you'll want one of the workarounds:**

| Mitigation | How |
|---|---|
| 4-bit quantize the frozen reference | `pip install bitsandbytes` + (forthcoming) `--quantize_reference` flag |
| Skip the separate reference; toggle LoRA adapters on/off for KL | (forthcoming) `--adapter_toggle_kl` flag |
| Single-rollout minibatches | `--minibatch_size 1 --episodes_per_iter 2` |
| Gradient checkpointing | already on in the model wrapper |

For now on 40 GB use the smallest-footprint settings:

```bash
xvfb-run -a bash ~/drone_project/openfly/run_train_ppo_agent.sh \
  --iterations 30 --episodes_per_iter 2 \
  --ppo_epochs 1 --minibatch_size 1 \
  --kl_coef 0.0 \
  --env_filter env_airsim_16
```

`--kl_coef 0.0` disables the ref-model KL term entirely so only the active LoRA copy is in VRAM. You lose the KL anchor (Stage 5 stretch goal), but the loop runs end-to-end. With an 80 GB A100 or H100 the default config (KL on) just works.

---

## 5. Disk budget on 500 GB

| Item | Size |
|---|---|
| `env_airsim_16` (unzipped UE binary + assets) | ~7 GB |
| All 6 AirSim scenes (16/18/23/26/gz/sh) | ~50–60 GB |
| `OpenFly-Platform` clone (no LFS) | ~200 MB |
| Annotations (`train.json` + `seen.json` + `unseen.json`) | ~600 MB |
| Trajectory images for `env_airsim_16` SFT | ~30–50 GB |
| HF `openfly-agent-7b` weights cache | ~14 GB |
| HF `paligemma-3b-pt-224` weights cache | ~6 GB |
| Checkpoints + rollout JSONL from a full run | ~10 GB |
| **Total realistic** | **~110–140 GB** |

500 GB is comfortable. 200 GB works if you only train one scene and skip trajectory frames for offline SFT.

---

## 6. Optional: split-host (GH200 trains, A100 rolls out)

If you keep a GH200 around for the heavy gradient compute, the A100 host can be a pure rollout server:

1. **A100 box:** run the upstream eval bridge as a long-lived process exposing port 41451 (the AirSim RPC default).
2. **GH200 box:** in `openfly/platform.py`, swap `AirsimBridge(env_name)` with a thin client that points `airsim.MultirotorClient(ip="A100_HOST_IP")` at the remote.
3. Open port 41451 on the A100 box's firewall to the GH200 (or use Tailscale/Wireguard).

The Hopper's 96 GB unified memory then handles all PaliGemma / OpenFly-Agent forward/backward passes while the A100 does AirSim rendering. This is the architecturally cleanest setup but adds network latency (~50 ms per `get_camera_data` call between data centers). Co-locate the boxes if possible.

I haven't shipped a `RemoteAirsimBridge` module yet — ask if you want it wired in.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Exec format error` on `start.sh` | Wrong CPU arch. Verify `uname -m` is `x86_64`. GH200 / Apple Silicon won't work. |
| `confirmConnection` hangs forever | UE didn't reach the AirSim RPC stage; check `~/.airsim/airsim_server_log.txt` and that `~/Documents/AirSim/settings.json` has `"Vehicles": { "drone_1": ... }`. |
| `No module named 'msgpack_rpc'` | `pip install msgpack-rpc-python` (setup.sh does this). |
| AirSim shows black frames | Vulkan failed to bind a GPU; add `xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render"` or run with `vulkaninfo` to verify the ICD. |
| `flash-attn` build fails | OK; `OpenFlyAgentPolicy` falls back to eager attention automatically (a touch slower). |
| OOM on Phase 5 | Use the small-footprint flags in §4; or rent an 80 GB GPU for that phase only. |
| Phase 1A FSDP errors | Multi-GPU only; out of scope for a single A100. Use the HF baseline checkpoint and jump to Phase 5 RL. |
| HF 403 on `OpenFly_DataGen` | You're logged in but not yet approved. Wait for the dataset gating approval email. |

---

## 8. End-to-end cheat sheet

```bash
# Once
bash ~/drone_project/openfly/setup.sh
huggingface-cli login
bash ~/drone_project/openfly/download_airsim_scene.sh env_airsim_16
mkdir -p ~/Documents/AirSim && cp ~/OpenFly-Platform/envs/airsim/AirSim/settings.json ~/Documents/AirSim/

# Sanity
xvfb-run -a bash ~/drone_project/openfly/run_eval.sh \
  --split unseen --policy heuristic --env_filter env_airsim_16 --max_episodes 5

# SFT
bash ~/drone_project/openfly/run_train_paligemma.sh --epochs 10 --batch_size 8

# Smoke RL env
xvfb-run -a python -m openfly.scripts.smoke_rl_env --episodes 3

# RL chain
xvfb-run -a bash ~/drone_project/openfly/run_train_dagger.sh --sft_ckpt logs/openfly/paligemma/<run>/last.pt
xvfb-run -a bash ~/drone_project/openfly/run_train_grpo.sh   --init_ckpt logs/openfly/dagger/<run>/last.pt
xvfb-run -a bash ~/drone_project/openfly/run_train_ppo_agent.sh --iterations 30 --episodes_per_iter 2 --minibatch_size 1 --kl_coef 0.0

# Eval the result
xvfb-run -a bash ~/drone_project/openfly/run_eval.sh --split unseen --policy grpo \
  --paligemma_ckpt logs/openfly/grpo/<run>/best.pt --env_filter env_airsim_16
```

---

## Cross-references

- [`openfly/README.md`](../openfly/README.md) — module-level reference for scripts and policy aliases.
- [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) — what to do once the pipeline runs.
- [`docs/BENCHMARK_FAIRNESS.md`](BENCHMARK_FAIRNESS.md) — what numbers from each phase are fair to claim.
- Validation gates G0–G5 are defined in the [implementation plan](#) — these are the criteria each phase must clear before moving on.
