#!/bin/bash

# =============================================================================
# GS-ProCams Synthetic-Surface Rendering for nepmap (synthetic) Datasets
# Hemisphere surface
# =============================================================================
# Companion to scripts/synthetic.sh (which trains the models).
# Projects the pattern onto a front-facing hemisphere.
# It reuses the trained nepmap models and renders via the render_gs_to_surface
# branch in render.py.
#
# Prerequisite: run scripts/synthetic.sh first so the trained models exist under
#   output/nepmap-dataset/wo_psf/setups/<setup>/
#
# To render ALL surface shapes in one run, use scripts/synthetic_surface.sh.
# =============================================================================

# Configuration (kept consistent with scripts/synthetic.sh)
input_dir="data/nepmap-dataset"
output_dir="output/nepmap-dataset"

setup_names=("castle" "planck" "zoo" "pear")
model_types=("wo_psf")

gpu_id="0"

# Surface placement / shape parameters (tune to your scene scale).
curve_radius="1.0"
curvature="0.5"

# -----------------------------------------------------------------------------
# Rendering loop
# -----------------------------------------------------------------------------
for setup_name in "${setup_names[@]}"; do
    for model_type in "${model_types[@]}"; do
        case "$model_type" in
            "wo_psf")
                model_dir="${output_dir}/wo_psf/setups/${setup_name}"
                ;;
            *)
                echo "Error: Unknown model type: $model_type"
                exit 1
                ;;
        esac

        if [ ! -d "$model_dir" ]; then
            echo "[WARN] Trained model not found for '${setup_name}' at: ${model_dir}"
            echo "       Run scripts/synthetic.sh first. Skipping."
            continue
        fi

        save_dir="${model_dir}/render_surface/hemisphere"
        mkdir -p "$save_dir"

        echo "============================================================"
        echo "Rendering setup='${setup_name}' surface='hemisphere'"
        echo "  -> ${save_dir}"
        echo "============================================================"

        python render.py \
            -r "$input_dir" \
            -s "$setup_name" \
            -m "$model_dir" \
            -o "$save_dir" \
            --white_background \
            --surface_mode hemisphere \
            --curve_radius "$curve_radius" \
            --curvature "$curvature" \
            --gpu_id "$gpu_id"
    done
done

echo "Synthetic-surface (hemisphere) rendering done."
