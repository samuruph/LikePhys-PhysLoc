# LikePhys: Evaluating intuitive physics understanding in video diffusion models via likelihood preference 
*LikePhys*, a training-free method that evaluates intuitive physics in video diffusion models by distinguishing physically valid and impossible videos using the denoising objective as an ELBO-based likelihood surrogate on a curated dataset of valid-invalid pairs.
**ICLR 2026**

[[arXiv]](https://arxiv.org/abs/2510.11512) [[Project Page]](https://yuanjianhao508.github.io/LikePhys/) [[Dataset]](https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark)

This repository evaluates on **two benchmarks** with the same method and the same mis-rank metric:

| Benchmark | `--data` | Videos | Caption |
|---|---|---|---|
| **LikePhys** | a scenario, e.g. `ball_drop` | `./data/<scenario>_videos/`, one folder of mp4s per scene | one fixed caption per scenario |
| **PhysLoc** | `physloc` | a PhysLoc release (`--physloc_root` or `--physloc_hub_repo`) | each clip's own caption |

Only how the videos are loaded, grouped and captioned differs; the models, the denoising loss and the mis-rank are shared. See [Datasets](#datasets).



## Usage

### Quick Start

1. **Setup Environment**:
```bash
# Clone repository
git clone https://github.com/YuanJianhao508/LikePhys.git
cd LikePhys

# Install dependencies
pip install torch torchvision diffusers accelerate transformers
pip install opencv-python pillow numpy matplotlib tqdm h5py
# Required for VS Code-compatible H.264 visualization videos
# Conda: conda install -c conda-forge ffmpeg

# Download dataset from Hugging Face
# Option 1: Using git clone (recommended)
git clone https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark data

# Option 2: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark --repo-type dataset --local-dir ./data
```

2. **Run Single Evaluation**:
```bash
# LikePhys: one physics scenario
python evaluator.py --model animatediff --data ball_drop --seed 42 --guidance_scale

# PhysLoc: a release, downloaded from the Hub on first use
python evaluator.py --model animatediff --data physloc --seed 42 --guidance_scale \
    --physloc_hub_repo samueleruf/physloc-mini
```

3. **Run Batch Evaluation**:
```bash
bash run_eval_likephys.sh   # every model on every LikePhys scenario
bash run_eval_physloc.sh    # every model on PhysLoc releases (see the script)
```

### Command Line Arguments

- `--model`: Model to evaluate (e.g., `animatediff`, `cogvideox`, `hunyuan_t2v`, `ltx`, `mochi`)
- `--data`: Benchmark to evaluate: a LikePhys scenario (e.g., `ball_drop`, `ball_collision`, `pendulum`), or `physloc`
- `--seed`: Random seed for reproducibility
- `--guidance_scale`: Use classifier-free guidance (flag)
- `--tag_name`: Custom tag for organizing experiment results

PhysLoc only (`--data physloc`):

- `--physloc_root`: a schema-v3 PhysLoc release root on disk (containing `samples/`)
- `--physloc_hub_repo`: a release to download instead, e.g. `samueleruf/physloc-mini` (into `--physloc_cache`, default `data/physloc`)
- `--physloc_split`: only this split of a downloaded release (`main`, `held_out`, `debug`)
- `--physloc_family`, `--physloc_scenario`, `--physloc_level`, `--physloc_condition`, `--physloc_severity_bin`: only invalid clips matching these; each pair's valid clip is always kept as the reference
- `--physloc_loader`: canonical loader from the PhysLoc checkout; defaults to `../physloc/physloc/loader.py` or `$PHYSLOC_LOADER`
- `--scores`: repeatable or comma-separated PhysLoc metric groups: `base_ppe`, `temporal_ppe`, `spatial_ppe`, `spatiotemporal_ppe`, or `all` (default: `base_ppe`)
- `--visualize`: write synchronized clip MP4s, clip summaries, and valid/invalid pair summaries using a shared denoising-error scale

### Sample Scripts

#### Single Model Evaluation
```bash
# Evaluate a single model on one physics scenario
python evaluator.py \
    --model animatediff \
    --data ball_drop \
    --seed 42 \
    --guidance_scale \
    --tag_name "experiment_1"
```

#### Batch Evaluation
```bash
# LikePhys: all models across all scenarios
bash run_eval_likephys.sh

# PhysLoc: all models on each release in PHYSLOC_ROOTS (space-separated)
PHYSLOC_ROOTS="../physloc/out/review_L0_f37" bash run_eval_physloc.sh

# All localized scores and visual evidence
PHYSLOC_SCORES=all PHYSLOC_VISUALIZE=1 \
  PHYSLOC_ROOTS="../physloc/out/review_L0_f37" bash run_eval_physloc.sh
```

## Datasets

### LikePhys

The dataset is hosted on Hugging Face and contains paired videos (physically plausible vs. implausible) across 12 different physics scenarios.

**Download from Hugging Face:** https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark

```bash
# Option 1: Using git clone (recommended for full dataset)
git clone https://huggingface.co/datasets/JianhaoDYDY/LikePhys-Benchmark data

# Option 2: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download JianhaoDYDY/LikePhys-Benchmark --repo-type dataset --local-dir ./data
```

![LikePhys Dataset Overview](assets/dataset.png)

### PhysLoc

A PhysLoc release groups clips into *pairs*: one valid clip and every invalid clip rendered from the same scene. Each pair is scored as one LikePhys subgroup, and an invalid clip's variation type is `<family>_<severity bin>` (e.g. `permanence_strong`), so the mis-rank is reported per violation family and severity.

Localized PPE keeps the original pair decision (`PPE(invalid) > PPE(valid)`) and adds complementary localization measurements. Temporal PPE stores native latent-frame traces and their projection onto sampled RGB frames. Spatial pair scores apply the same annotation mask to both twins. Spatio-temporal AP (average precision) ranks time-space tokens by PPE error and measures whether annotated violation tokens rank highest; error ratios compare mean PPE inside an annotation with foreground outside it.

The spatial vocabulary is:

- `violating_object`: visible invalid-side silhouette of the violator.
- `active_violating_object`: the visible violator silhouette restricted to its event clock.
- `active_violation`: the valid/invalid footprint union on observable evidence frames.
- `active_violation_visible`: the invalid-side visible subset.
- `expected_object`: the lawful valid-twin trajectory during consequence frames.
- `causal_consequence`: downstream affected objects and causal pixels.
- `outside_violation_foreground` / `outside_violation_all`: comparison regions without / with background.

This union/reference formulation also handles permanence: disappearance and reappearance are event frames, while `expected_object` follows the lawful trajectory through the missing interval. Unweighted scores are used for weak/medium/strong trend tests; severity-weighted PPE is reported only as a separate diagnostic.

Each PhysLoc result directory contains `analysis/data/` CSV files and `analysis/plots/` category and severity plots. With `--visualize`, `visualizations/clips/` contains four-panel MP4s and PNG summaries, and `visualizations/pairs/` contains valid/invalid temporal comparisons.

PhysLoc releases are read with the canonical `physloc/loader.py` from the PhysLoc repository rather than a duplicated loader. The default sibling checkout is `/home/ec2-user/code/physloc`; use `--physloc_loader` or `$PHYSLOC_LOADER` elsewhere. To validate the schema-v3 layout, `h5py` dependency, and pairs before spending GPU time:

```bash
python -m benchmarks.datasets.physloc --physloc_root ../physloc/out/review_L0_f37
```

The evaluation code is organized by responsibility: `datasets/` is the only
dataset-adapter package (LikePhys configuration/video discovery and the
PhysLoc loader bridge); `benchmarks/` owns benchmark-specific scoring
orchestration; `utils/` contains only shared PPE metrics, reporting,
visualization, and generic helpers; and `evaluator.py` owns model setup,
denoising PPE, CLI configuration, and run persistence.

This evaluator intentionally requires schema v3. Passing a schema-v2 `clips/`/NPZ release produces an explicit error before model initialization.

## Supported Models

- **AnimateDiff** (`animatediff`)
- **AnimateDiff SDXL** (`animatediff_sdxl`)
- **CogVideoX** (`cogvideox`, `cogvideox-5b`)
- **Hunyuan Video** (`hunyuan_t2v`)
- **LTX Video** (`ltx`)
- **ModelScope** (`modelscope`)
- **Wan Video** (`wan2.1-T2V-1.3b`, `wan2.1-T2V-14b`)
- **ZeroScope** (`zeroscope`)

## Results Analysis

Every batch script automatically writes a dataset-neutral summary after all
workers finish. It contains tidy variation-level and model-level CSVs plus a
PNG chart. To re-create a summary or aggregate both benchmarks together, use
the evaluator's no-model analysis mode:

```bash
# One LikePhys experiment
python evaluator.py --summarize_results \
  --summary_results_dir results/evaluation_t10_uniform_42_cfg_final

# One PhysLoc experiment
python evaluator.py --summarize_results \
  --summary_results_dir results/evaluation_t10_uniform_42_cfg_final_physloc-review_L0_f37

# All experiments beneath results/
python evaluator.py --summarize_results --summary_results_dir results
```


## Citation

If you use LikePhys in your research, please cite:

```bibtex
@inproceedings{yuan2025likephys,
  title={LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference},
  author={Yuan, Jianhao and Pizzati, Fabio and Pinto, Francesco and Kunze, Lars and Laptev, Ivan and Newman, Paul and Torr, Philip and De Martini, Daniele},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

```
