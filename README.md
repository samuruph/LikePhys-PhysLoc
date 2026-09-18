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

# Install Python dependencies
pip install torch torchvision diffusers accelerate transformers
pip install opencv-python pillow numpy matplotlib tqdm h5py

# Required only for MP4 visualizations
conda install -c conda-forge ffmpeg

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
bash run_eval_likephys.sh   # configured model(s) on every LikePhys scenario
bash run_eval_physloc.sh    # configured model(s) on each PhysLoc release
```

The checked-in batch scripts are configured to run the Wan 1.3B model by
default. Each script also contains the full supported-model list; edit its
active `MODELS=(...)` assignment if you want the complete suite. The scripts
use GPU `0` and seed `42` by default, so edit `GPUS=(...)` and `SEEDS=(...)`
for a different sweep.

The evaluator defaults to seed `42`, ten uniformly spaced timesteps, and
`results/` as the output root. Pass `--tag_name` when starting a new run so
that it cannot be confused with an earlier result directory.

### Command Line Arguments

- `--model`: Model to evaluate (e.g., `animatediff`, `cogvideox`, `hunyuan_t2v`, `ltx`, `mochi`)
- `--data`: Benchmark to evaluate: a LikePhys scenario (e.g., `ball_drop`, `ball_collision`, `pendulum`), or `physloc`
- `--seed`: Random seed for reproducibility
- `--guidance_scale`: Use classifier-free guidance (flag)
- `--tag_name`: Custom tag for organizing experiment results
- `--exp_name`: Experiment name prefix (default: `evaluation_t10_uniform`)
- `--output_dir`: Output root (default: `results`)
- `--timestep_num`: Number of uniformly sampled denoising timesteps (default: `10`)
- `--timestep_strategy`: Timestep strategy label (default: `uniform`)

PhysLoc only (`--data physloc`):

- `--physloc_root`: a schema-v3 PhysLoc release root on disk (containing `samples/`)
- `--physloc_hub_repo`: a release to download instead, e.g. `samueleruf/physloc-mini` (into `--physloc_cache`, default `data/physloc`)
- `--physloc_split`: only this split of a downloaded release (`main`, `held_out`, `debug`)
- `--physloc_family`, `--physloc_scenario`, `--physloc_level`, `--physloc_condition`, `--physloc_severity_bin`: only invalid clips matching these; each pair's valid clip is always kept as the reference
- `--physloc_loader`: canonical loader from the PhysLoc checkout; defaults to `../physloc/physloc/loader.py` or `$PHYSLOC_LOADER`
- `--scores`: repeatable or comma-separated PhysLoc metric groups: `base_ppe`, `temporal_ppe`, `spatial_ppe`, `spatiotemporal_ppe`, or `all` (default: `base_ppe`)
- `--visualize`: write synchronized valid/invalid/delta MP4s and PNG temporal summaries

The `--scores` groups are cumulative only in the sense that each requested
group is written to the result JSON. `base_ppe` is the scalar valid/invalid
pair score; `temporal_ppe` adds temporal traces and event windows;
`spatial_ppe` adds annotation-region PPE; `spatiotemporal_ppe` adds token AP
and error ratios. `all` requests all four groups and is the usual choice for
PhysLoc localization experiments.

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

For one reproducible PhysLoc check, use an explicit tag:

```bash
FFMPEG_BINARY=/home/ec2-user/miniconda3/bin/ffmpeg \
python evaluator.py \
  --model wan2.1-T2V-1.3b \
  --data physloc \
  --physloc_root ../physloc/out/review_L0_f37 \
  --seed 42 \
  --scores all \
  --visualize \
  --tag_name viz_delta
```

The command above writes to
`results/evaluation_t10_uniform_42_no_cfg_viz_delta/physloc/`. Adding
`--guidance_scale` changes `no_cfg` to `cfg` in the directory name.

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

Each PhysLoc result directory contains the following main artifacts:

```text
<run>/physloc/
├── results_<model>.json
├── analysis/
│   ├── data/metrics_<model>.csv
│   ├── data/category_summary_<model>.csv
│   ├── data/severity_ladders_<model>.csv
│   └── plots/<model>/
└── visualizations/<model>/       # only with --visualize
    ├── <invalid_sample_uid>.mp4
    └── <invalid_sample_uid>.png
```

The visualization MP4 uses valid, invalid, and delta rows. It synchronizes
RGB clips, annotations, both absolute denoising-error maps, the signed
invalid-minus-valid error heatmap, per-frame PPE values, and legends for both
colormaps. The PNG contains the paired valid and invalid temporal traces plus
their difference. Old runs may still contain the former `visualizations/clips`
and `visualizations/pairs` layout; new runs use the unified layout above.

MP4 encoding requires an H.264-capable FFmpeg binary. The code checks
`FFMPEG_BINARY`, the current `PATH`, and common locations including
`/home/ec2-user/miniconda3/bin/ffmpeg`. Set `FFMPEG_BINARY` explicitly if a
Conda environment does not expose FFmpeg on its `PATH`.

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

Every completed evaluation writes PhysLoc-specific CSVs and plots inside its
run directory. To re-create those artifacts from an existing PhysLoc JSON,
rerun the evaluator's no-model analysis mode:

```bash
python evaluator.py --summarize_results \
  --summary_results_dir results/evaluation_t10_uniform_42_no_cfg_viz_delta/physloc
```

The command does not load a model. For result directories that contain only
LikePhys JSON files, or for a cross-experiment comparison, use the lightweight
analysis module:

```bash
# One LikePhys experiment
python -m benchmarks.analysis.results \
  --results-dir results/evaluation_t10_uniform_42_cfg_quick_check \
  --output-dir results/evaluation_t10_uniform_42_cfg_quick_check/analysis

# Another experiment for comparison
python -m benchmarks.analysis.results \
  --results-dir results/evaluation_t10_uniform_42_cfg_final_likephys \
  --output-dir results/evaluation_t10_uniform_42_cfg_final_likephys/analysis

# All experiments beneath results/
python -m benchmarks.analysis.results \
  --results-dir results \
  --output-dir results/aggregate_analysis
```

The lightweight module writes `misrank_by_variation.csv`,
`misrank_summary.csv`, and `misrank_summary.png`. The evaluator's PhysLoc
analysis writes the metric CSVs and category/severity plots described above.
There is currently no automatic LaTeX writer; the CSV files are the stable
tabular interface for generating LaTeX tables externally. Keep separate
`--output-dir` values when comparing experiments so their records are not
mixed.

Every batch script also writes a dataset-neutral summary after all workers
finish. The equivalent commands are:

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

If an existing `results_<model>.json` is detected, the evaluator may skip the
expensive run when its requested score groups and dataset root already match.
Use a new `--tag_name`, or remove only the specific stale result directory
after checking it, when you need to regenerate visualizations or changed
analysis artifacts.


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
