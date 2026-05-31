#!/bin/bash

# =============================================================================
# GS-ProCams Synthetic-Surface Rendering for nepmap (synthetic) Datasets
# Curved (sinusoidal) surface
# =============================================================================
# Companion to scripts/synthetic.sh (which trains the models).
# Projects the pattern onto a sine-wave modulated (wrinkled) screen.
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

# Pick a Python interpreter: prefer an active environment's "python",
# otherwise fall back to "python3". Override by exporting PYTHON_BIN.
if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Error: neither 'python' nor 'python3' found on PATH."
        echo "       Activate your environment first, e.g.: conda activate gs-procams"
        exit 1
    fi
fi

# Surface placement / shape parameters (tune to your scene scale).
curve_radius="1.0"
curvature="0.5"

# View ids to render. Leave EMPTY to auto-detect from each model's cameras.json
# (recommended for nepmap, whose view_id set depends on the training config).
# To force specific views, set e.g.  views="1 6 11"
views=""

# Helper: read available view_ids from a model's cameras.json.
get_views_from_cameras_json() {
    local cam_json="$1/cameras.json"
    [ -f "$cam_json" ] || return 1
    "$PYTHON_BIN" -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(str(c['view_id']) for c in d))" "$cam_json"
}

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

        save_dir="${model_dir}/render_surface/curved_sinusoidal"
        mkdir -p "$save_dir"

        echo "============================================================"
        echo "Rendering setup='${setup_name}' surface='curved_sinusoidal'"
        echo "  -> ${save_dir}"
        echo "============================================================"

        # Resolve which views to render: use $views if set, else auto-detect.
        render_views="$views"
        if [ -z "$render_views" ]; then
            render_views="$(get_views_from_cameras_json "$model_dir")"
        fi
        if [ -z "$render_views" ]; then
            echo "[WARN] No views found for '${setup_name}' (set 'views' or check cameras.json). Skipping."
            continue
        fi

        "$PYTHON_BIN" render.py \
            -r "$input_dir" \
            -s "$setup_name" \
            -m "$model_dir" \
            -o "$save_dir" \
            --white_background \
            --views ${render_views} \
            --surface_mode curved \
            --curve_type sinusoidal \
            --curve_radius "$curve_radius" \
            --curvature "$curvature" \
            --gpu_id "$gpu_id"
    done
done

echo "Synthetic-surface (curved_sinusoidal) rendering done."
