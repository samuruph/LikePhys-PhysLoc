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
pip install opencv-python pillow numpy matplotlib tqdm

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

- `--physloc_root`: a release on disk, downloaded/exported or a generator run -- both are `clips/` folders
- `--physloc_hub_repo`: a release to download instead, e.g. `samueleruf/physloc-mini` (into `--physloc_cache`, default `data/physloc`)
- `--physloc_split`: only this split of a downloaded release (`main`, `held_out`, `debug`)
- `--physloc_family`, `--physloc_scenario`, `--physloc_level`, `--physloc_condition`, `--physloc_severity_bin`: only invalid clips matching these; each pair's valid clip is always kept as the reference
- `--physloc_repo`: the PhysLoc checkout whose `physloc/loader.py` reads a release that ships no `loader.py`; a downloaded release is read with its own (default: `$PHYSLOC_REPO`, then `../physloc`)

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
PHYSLOC_ROOTS="data/physloc/samueleruf__physloc-mini" bash run_eval_physloc.sh
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

Releases are read with PhysLoc's dataloader, `physloc/loader.py` from the PhysLoc repository (a checkout at `../physloc`, or wherever `--physloc_repo` / `$PHYSLOC_REPO` points). It needs only numpy, so the generator does not have to be installed here. `utils/physloc_dataset.py` hands the evaluator the loader's own `Pair` and `Clip` objects. To see which pairs a release yields before spending GPU time:

```bash
python -m utils.physloc_dataset --physloc_root data/physloc/samueleruf__physloc-mini
```

Releases exported before PhysLoc schema v2 (videos stored as `rgb.mp4`) cannot be read and are reported as such; re-export them with a current `physloc export`.

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

After evaluation, use the analysis script to check results

```bash
python read_exp_final.py
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
