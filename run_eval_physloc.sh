#!/usr/bin/env bash
# Evaluate every model on one or more PhysLoc releases.
#
#   bash run_eval_physloc.sh                                   # the releases listed below
#   PHYSLOC_ROOTS="data/physloc/a data/physloc/b" bash run_eval_physloc.sh
#
# A release is a downloaded/exported one or a generator run -- both are clips/
# folders. A download is read with the loader.py it ships; a generator run with
# physloc/loader.py from the PhysLoc checkout: set PHYSLOC_REPO if that is not
# ../physloc.
GPUS=(0) # 1 2 3 4 5 6 7)

SEEDS=(42)
# PhysLoc releases; results for each are tagged with its folder name
read -r -a RELEASES <<< "${PHYSLOC_ROOTS:-data/physloc-review_L0_f37}"
MODELS=(animatediff zeroscope modelscope wan2.1-T2V-1.3b hunyuan_t2v ltx-0.9.5 animatediff_sdxl cogvideox mochi cogvideox-5b wan2.1-T2V-14b)
MODELS=(wan2.1-T2V-1.3b)
FLAGS=("--guidance_scale")
SCORES="${PHYSLOC_SCORES:-base_ppe}"
VIZ_FLAG=()
if [ "${PHYSLOC_VISUALIZE:-0}" = "1" ]; then
  VIZ_FLAG=("--visualize")
fi

declare -A GPU_PIDS

get_free_gpu() {
  for idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$idx]}
    pid=${GPU_PIDS[$gpu]}
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "$gpu"
      return
    fi
  done
  echo ""
}

for release in "${RELEASES[@]}"; do
  if [ ! -d "$release" ]; then
    echo "✗ no PhysLoc release at $release; download one with" >&2
    echo "  hf download samueleruf/physloc-mini --repo-type dataset --local-dir $release" >&2
    exit 1
  fi
done

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for release in "${RELEASES[@]}"; do
      for flag in "${FLAGS[@]}"; do

        # Wait for a free GPU
        while true; do
          gpu=$(get_free_gpu)
          if [ -n "$gpu" ]; then
            break
          fi
          sleep 1
        done

        tag="final_$(basename "$release")"
        echo "→ GPU $gpu ← model=$model, release=$release, seed=$seed, flag=$flag"
        CUDA_VISIBLE_DEVICES=$gpu \
          python evaluator.py --model="$model" --data=physloc --physloc_root="$release" \
            --seed="$seed" $flag --scores="$SCORES" "${VIZ_FLAG[@]}" --tag_name="$tag" &
        GPU_PIDS[$gpu]=$!
      done
    done
  done
done

wait
echo "✅ All PhysLoc evaluations done."
