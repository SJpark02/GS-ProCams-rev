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
# curve_radius : characteristic radius / spatial scale of the surface.
# curvature    : curvature / amplitude (used by curved surfaces).
curve_radius="1.0"
curvature="0.5"

# View ids to render. Leave EMPTY to auto-detect from each model's cameras.json
# (recommended for nepmap, whose view_id set depends on the training config).
# To force specific views, set e.g.  views="1 6 11"
views=""

# When views are auto-detected from cameras.json, limit the number of views
# rendered per surface to this many (evenly spaced across all available
# views). The nepmap synthetic dataset can contain hundreds of views, so
# rendering all of them onto every surface would be very slow. Set to 0 to
# render ALL detected views. Ignored when "views" is set explicitly above.
max_views="${MAX_VIEWS:-5}"

# Directory of patterns/images to project. Leave EMPTY to let render.py
# auto-detect (nepmap synthetic uses <root>/setups/<setup>/projector).
# Set this (or env PATTERN_PATH) to a folder of your OWN images to do
# custom projection mapping, e.g. pattern_path="data/my_images".
pattern_path="${PATTERN_PATH:-}"

# Also render the SAME pattern on the real object (the original synthetic.sh
# result) next to the surface render, so you can compare how the synthetic.sh
# output looks when re-projected onto a sphere/hemisphere/etc.
# Saved under <view>/object/. Set RENDER_OBJECT_TOO=1 to enable. ON by default
# because the whole point of this script is that comparison.
render_object_too="${RENDER_OBJECT_TOO:-1}"

# Name output files after the source pattern (e.g. img_0001.png) instead of a
# running index (01.png), matching the original synthetic.sh naming.
# Set NAME_BY_PATTERN=0 to use the running-index naming instead.
name_by_pattern="${NAME_BY_PATTERN:-1}"

# Helper: read available view_ids from a model's cameras.json.
get_views_from_cameras_json() {
    local cam_json="$1/cameras.json"
    [ -f "$cam_json" ] || return 1
    "$PYTHON_BIN" -c "
import json, sys
data = json.load(open(sys.argv[1]))
ids = [c['view_id'] for c in data]
def num(v):
    try: return int(str(v))
    except Exception: return None
nums = [num(v) for v in ids]
if all(n is not None for n in nums):
    sel = sorted(set(nums))
else:
    sel = list(dict.fromkeys(str(v) for v in ids))
mx = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 0
if mx and len(sel) > mx:
    step = (len(sel) - 1) / float(mx - 1) if mx > 1 else 0
    sel = [sel[int(round(i * step))] for i in range(mx)]
    seen = set(); sel = [x for x in sel if not (x in seen or seen.add(x))]
print(' '.join(str(x) for x in sel))
" "$cam_json" "$max_views"
}

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

            # Resolve which views to render: use $views if set, else auto-detect
            # from this model's cameras.json.
            render_views="$views"
            if [ -z "$render_views" ]; then
                render_views="$(get_views_from_cameras_json "$model_dir")"
            fi
            if [ -z "$render_views" ]; then
                echo "[WARN] No views found for '${setup_name}' (set 'views' or check cameras.json). Skipping."
                continue
            fi

            # render.py uses the new render_gs_to_surface branch when
            # --surface_mode is provided.
            pattern_arg=()
            if [ -n "$pattern_path" ]; then pattern_arg=(--pattern_path "$pattern_path"); fi

            # Comparison / naming passthrough flags.
            compare_arg=()
            if [ "$render_object_too" != "0" ]; then compare_arg+=(--render_object_too); fi
            if [ "$name_by_pattern" != "0" ]; then compare_arg+=(--name_by_pattern); fi

            "$PYTHON_BIN" render.py \
                -r "$input_dir" \
                -s "$setup_name" \
                -m "$model_dir" \
                -o "$save_dir" \
                --white_background \
                --views ${render_views} \
                "${surface_args[@]}" \
                "${pattern_arg[@]}" \
                "${compare_arg[@]}" \
                --gpu_id "$gpu_id"
        done
    done
done

echo "Synthetic-surface rendering done."
