#!/bin/bash

# =============================================================================
# GS-ProCams Synthetic-Surface Rendering for nepmap (synthetic) Datasets
# =============================================================================
# Companion to scripts/synthetic.sh.
#
# scripts/synthetic.sh trains the GS-ProCams models on the nepmap synthetic
# datasets. This script reuses those trained models and renders the projected
# pattern onto VIRTUAL (analytic) surfaces using the new render_gs_to_surface
# branch in render.py:
#       sphere | hemisphere | curved(cylindrical|sinusoidal|parabolic)
#
# It iterates over every surface mode for every setup so you can preview, in one
# run, how each scene would look when projected onto a hemisphere or curved
# screen. For a single surface only, see the per-shape scripts:
#       scripts/synthetic_sphere.sh
#       scripts/synthetic_hemisphere.sh
#       scripts/synthetic_cylindrical.sh
#       scripts/synthetic_sinusoidal.sh
#       scripts/synthetic_parabolic.sh
#
# Prerequisite: run scripts/synthetic.sh first so the trained models exist under
#   output/nepmap-dataset/wo_psf/setups/<setup>/
# =============================================================================

# Configuration (kept consistent with scripts/synthetic.sh)
input_dir="data/nepmap-dataset"
output_dir="output/nepmap-dataset"

setup_names=("castle" "planck" "zoo" "pear")
model_types=("wo_psf")

gpu_id="0"

# Surface placement / shape parameters (tune to your scene scale).
# curve_radius : characteristic radius / spatial scale of the surface.
# curvature    : curvature / amplitude (used by curved surfaces).
curve_radius="1.0"
curvature="0.5"

# Surfaces to render. Each entry is: "<surface_mode>[:<curve_type>]"
# (curve_type only applies to the 'curved' mode).
surfaces=(
    "sphere"
    "hemisphere"
    "curved:cylindrical"
    "curved:sinusoidal"
    "curved:parabolic"
)

# -----------------------------------------------------------------------------
# Rendering loop
# -----------------------------------------------------------------------------
for setup_name in "${setup_names[@]}"; do
    for model_type in "${model_types[@]}"; do
        # Resolve model directory based on model type (matches synthetic.sh).
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

        for surface in "${surfaces[@]}"; do
            # Split "<surface_mode>:<curve_type>" into parts.
            surface_mode="${surface%%:*}"
            curve_type="${surface##*:}"

            # Build the surface-specific arguments and output tag.
            surface_args=("--surface_mode" "$surface_mode" \
                          "--curve_radius" "$curve_radius" \
                          "--curvature" "$curvature")
            if [ "$surface_mode" == "curved" ]; then
                surface_args+=("--curve_type" "$curve_type")
                tag="curved_${curve_type}"
            else
                tag="$surface_mode"
            fi

            save_dir="${model_dir}/render_surface/${tag}"
            mkdir -p "$save_dir"

            echo "============================================================"
            echo "Rendering setup='${setup_name}' surface='${tag}'"
            echo "  -> ${save_dir}"
            echo "============================================================"

            # render.py uses the new render_gs_to_surface branch when
            # --surface_mode is provided.
            python render.py \
                -r "$input_dir" \
                -s "$setup_name" \
                -m "$model_dir" \
                -o "$save_dir" \
                --white_background \
                "${surface_args[@]}" \
                --gpu_id "$gpu_id"
        done
    done
done

echo "Synthetic-surface rendering done."
