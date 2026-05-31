# Custom projection images

This folder is the **default** source of images that get projected onto the
analytic surfaces (sphere / hemisphere / curved) by the
`scripts/synthetic_*surface/shape.sh` rendering scripts.

## How to use

1. Drop any images you want to project into this folder, e.g.:

   ```
   data/my_images/logo.png
   data/my_images/photo.jpg
   ```

   Supported extensions: `.png .jpg .jpeg .bmp .tif .tiff`
   (a file literally named `all_white.*` is skipped automatically.)

2. Run a rendering script as usual:

   ```bash
   bash scripts/synthetic_surface.sh        # all shapes
   # or a single shape:
   bash scripts/synthetic_sphere.sh
   ```

   Every image in this folder is projected onto each shape, for each rendered
   view. Outputs land in:

   ```
   output/.../render_surface/<shape>/<view_id>/
   ├── object/   # same image on the real object (synthetic.sh-style result)
   ├── relit/    # the image mapped onto the shape  <-- what you want to see
   └── Ip_out/   # the warped pattern (reference)
   ```

## Add / remove projected images

- **Add**: copy an image file into this folder.
- **Remove**: delete the image file from this folder.

That's it — the scripts pick up whatever is in here.

## Notes

- If this folder is **empty or missing**, the scripts automatically fall back
  to the dataset's own projector patterns
  (nepmap: `<root>/setups/<setup>/projector`) so they still run.
- To use a *different* folder for one run without editing the scripts:

  ```bash
  PATTERN_PATH="path/to/other/images" bash scripts/synthetic_surface.sh
  ```

- Your actual image files placed here are git-ignored; only this README is
  tracked.
