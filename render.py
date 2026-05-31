import json
import os
import time
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from gaussian_renderer import render, render_gs_to_surface
from scene import GaussianModel
from utils.camera_utils import loadMicroCameras_COLMAP, loadMicroCameras_JSON, LoadProjector_JSON
from utils.image_utils import loadImage

def save_img_f32(depthmap: np.ndarray, path: str):
    """Save a float32 depth map as a TIFF using OpenCV."""
    clean = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not cv2.imwrite(path, clean):
        raise IOError(f"Failed to write image to {path}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Simple relighting script parameters")
    parser.add_argument("--model_path", '-m', type=str, required=True, help="Path to the model")
    parser.add_argument("--root", '-r', type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--setup", '-s', type=str, required=True, help="Setup name")
    parser.add_argument("--output", '-o', type=str, required=True, help="Output directory")
    parser.add_argument("--iteration", type=int, default=20_000, help="Iteration of GS-ProCams to load")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")
    parser.add_argument("--views", type=int, nargs='+', default=[1, 6, 11, 16, 21, 26, 27, 28, 29, 30, 31, 32, 33], help="View ids to relight")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--test_fps", action="store_true", help="Test FPS")
    parser.add_argument("--render_scene", action="store_true", help="Render the scene")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    # ---- Virtual surface projection (render_gs_to_surface branch) ----
    parser.add_argument("--surface_mode", type=str, default=None,
                        choices=["sphere", "hemisphere", "curved"],
                        help="Project the pattern onto a synthetic surface instead of the real geometry. "
                             "If omitted, the default depth-based render() is used.")
    parser.add_argument("--curve_type", type=str, default="cylindrical",
                        choices=["cylindrical", "sinusoidal", "parabolic"],
                        help="Sub-type of curved surface (only used when --surface_mode curved).")
    parser.add_argument("--curve_radius", type=float, default=1.0,
                        help="Characteristic radius / spatial scale of the synthetic surface.")
    parser.add_argument("--curvature", type=float, default=0.5,
                        help="Curvature / amplitude parameter for curved surfaces.")
    parser.add_argument("--surface_res", type=int, default=None,
                        help="Optional surface sampling resolution hint (reserved).")
    parser.add_argument("--pattern_path", type=str, default=None,
                        help="Directory of projection patterns to relight with. "
                             "If omitted, it is auto-detected: <root>/patterns/test for "
                             "real-world (colmap) data, or <root>/setups/<setup>/projector "
                             "for the nepmap synthetic dataset. Point this at your own folder "
                             "of images to projection-map custom content.")
    parser.add_argument("--surface_albedo", type=float, default=0.8,
                        help="Uniform diffuse albedo (0-1) of the virtual projection screen. The analytic surface has no Gaussian material, so this drives how brightly the projected pattern shows up. Set <0 to keep the rasterized scene material instead.")
    parser.add_argument("--surface_roughness", type=float, default=0.6,
                        help="Uniform roughness (0-1) of the virtual projection screen. Higher = more matte/diffuse, lower = glossier specular highlights.")
    parser.add_argument("--ambient", type=float, default=0.0,
                        help="Weight of the rasterized scene colour blended into the result. 0 (default) = pure projection on the virtual screen; >0 blends in the real object.")
    parser.add_argument("--render_object_too", action="store_true",
                        help="Also render the SAME pattern on the real, depth-reconstructed object (the original synthetic.sh result) alongside the surface render, so you can compare how the synthetic.sh output looks when applied to a sphere/hemisphere/etc. Saved under <view>/object/.")
    parser.add_argument("--name_by_pattern", action="store_true",
                        help="Name output files after the source pattern file (e.g. img_0001.png) instead of a running index (01.png), matching the original synthetic.sh naming.")
    parser.add_argument("--no_auto_scale", action="store_true",
                        help="Disable scene-aware auto-sizing of the synthetic surface. "
                             "By default the surface is auto-placed/-sized from the trained "
                             "Gaussians so it fills the view; with this flag, --curve_radius "
                             "and --curvature are treated as absolute world-unit values.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    with torch.no_grad():
        # Gaussian model loading
        gaussians = GaussianModel(args.sh_degree)
        ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(ply_path): raise FileNotFoundError(f'Can not find ply file: {ply_path}')
        gaussians.load_ply(ply_path)
        print(f'Gaussians loaded from: {ply_path}')
        
        # Cameras loading
        cam_json_path = os.path.join(args.model_path, "cameras.json")
        try:
            cameras = loadMicroCameras_JSON(cam_json_path, args.views)
            cameras = [cameras] if not isinstance(cameras, list) else cameras
            print(f'Cameras ({len(cameras)}) loaded from json file')
        except:
            colmap_model_dir = os.path.join(args.root, "setups", args.setup, "colmap", "sparse", "0")
            cameras = loadMicroCameras_COLMAP(colmap_model_dir, args.views)
            cameras = [cameras] if not isinstance(cameras, list) else cameras
            print(f"Cameras ({len(cameras)}) loaded from colmap model")
        
        cameras_dict = {}
        for i, view_id in enumerate(args.views):
            cameras_dict[view_id] = cameras[i]

        # Projector loding
        prj_json_path = os.path.join(args.model_path, "projector.json")
        projector = LoadProjector_JSON(prj_json_path) 
        ckpt_path = os.path.join(args.model_path, "procams", f"iteration_{args.iteration}", "procams.ckpt")
        projector.load_ckpt(ckpt_path, weights_only=True)
        print("Projector loaded")
        procams_dict = {"projector": projector}

        # Validation patterns loading.
        # The pattern directory differs between dataset layouts:
        #   * real-world (colmap): <root>/patterns/test
        #   * nepmap synthetic   : <root>/setups/<setup>/projector
        # A user-supplied --pattern_path always wins (e.g. to project custom images).
        if args.pattern_path is not None:
            patterns_valid_dir = args.pattern_path
        else:
            candidates = [
                os.path.join(args.root, "patterns", "test"),
                os.path.join(args.root, "setups", args.setup, "projector"),
            ]
            patterns_valid_dir = next((c for c in candidates if os.path.isdir(c)), candidates[0])
        if not os.path.isdir(patterns_valid_dir):
            raise FileNotFoundError(
                f"Pattern directory not found: {patterns_valid_dir}. "
                f"Pass --pattern_path <dir> to specify the folder of patterns/images to project."
            )
        pattern_files = [f for f in sorted(os.listdir(patterns_valid_dir))
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
                         and os.path.splitext(f)[0] not in ("all_white",)]
        if len(pattern_files) == 0:
            raise FileNotFoundError(f"No image patterns found in {patterns_valid_dir}.")
        print(f"Loading {len(pattern_files)} pattern(s) from: {patterns_valid_dir}")
        # Keep the source pattern file stems so outputs can be named after the
        # pattern (matching the original synthetic.sh naming) when requested.
        pattern_names = [os.path.splitext(f)[0] for f in pattern_files]
        patterns_valid = [loadImage(os.path.join(patterns_valid_dir, pattern_valid_file)) for pattern_valid_file in tqdm(pattern_files, leave=False)]
        patterns_valid = [torch.from_numpy(pattern_valid).float().cuda().permute(2, 0, 1).clamp(0, 1) for pattern_valid in patterns_valid]
        patterns_valid = torch.stack(patterns_valid, dim=0)
        
        # Relighting
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda") if not args.white_background else torch.ones(3, dtype=torch.float32, device="cuda")
        for view_id, camera in cameras_dict.items():
            save_dir = os.path.join(args.output, f"{view_id:02d}", "relit")
            Ip_out_dir = os.path.join(args.output, f"{view_id:02d}", "Ip_out")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(Ip_out_dir, exist_ok=True)
            # When --render_object_too is set we also produce the ORIGINAL synthetic.sh
            # result (same pattern projected on the real, depth-reconstructed object) so
            # the surface render can be compared against it side by side.
            object_dir = os.path.join(args.output, f"{view_id:02d}", "object")
            if args.render_object_too and args.surface_mode is not None:
                os.makedirs(object_dir, exist_ok=True)
            for i, pattern in enumerate(tqdm(patterns_valid, desc=f"Relighting view {view_id:02d}", leave=False)):
                procams_dict.update({"pattern": pattern})
                # Output stem: pattern name (matches synthetic.sh) or running index.
                stem = pattern_names[i] if args.name_by_pattern else f"{i+1:02d}"
                if args.surface_mode is None:
                    render_dic = render(camera, gaussians, pipe=None, bg_color=bg_color, procams_dict=procams_dict)
                else:
                    render_dic = render_gs_to_surface(camera, gaussians, procams_dict, pipe=None, bg_color=bg_color,
                                                      surface_mode=args.surface_mode, curve_type=args.curve_type,
                                                      curve_radius=args.curve_radius, curvature=args.curvature,
                                                      surface_res=args.surface_res, auto_scale=not args.no_auto_scale,
                                                      surface_albedo=args.surface_albedo,
                                                      surface_roughness=args.surface_roughness,
                                                      ambient=args.ambient)
                    # Same pattern on the real object (original synthetic.sh-style result).
                    if args.render_object_too:
                        object_dic = render(camera, gaussians, pipe=None, bg_color=bg_color, procams_dict=procams_dict)
                        save_image(object_dic["render"], os.path.join(object_dir, f"{stem}.png"))
                render_image = render_dic["render"]
                save_path = os.path.join(save_dir, f"{stem}.png")
                save_image(render_image, save_path)
                Ip_out = render_dic["Ip_out"]
                save_path = os.path.join(Ip_out_dir, f"{stem}.png")
                save_image(Ip_out, save_path)

            if args.render_scene:
                save_scene_dir = os.path.join(args.output, f"{view_id:02d}", "scene")
                os.makedirs(save_scene_dir, exist_ok=True)
                render_pkg = render(camera, gaussians, pipe=None, bg_color=bg_color)
                save_image(render_pkg["base_color"], os.path.join(save_scene_dir, "base_color.png"))
                save_image(render_pkg["roughness"], os.path.join(save_scene_dir, "roughness.png"))
                save_image(render_pkg["render_normal"], os.path.join(save_scene_dir, "render_normal.png"))
                save_image(render_pkg["surf_normal"], os.path.join(save_scene_dir, "surf_normal.png"))
                depth = render_pkg["depth"]
                save_image((depth - depth.min()) / (depth.max() - depth.min()), os.path.join(save_scene_dir, "depth.png"))
                # save depth as a float32 TIFF w/o normalization
                save_img_f32(depth[0].cpu().numpy(), os.path.join(save_scene_dir, 'depth.tiff'))
                save_image(render_pkg["render_shs"], os.path.join(save_scene_dir, "render_shs.png"))
        print("Relighting done")

        # Test FPS
        if args.test_fps:
            start_time = time.time()
            for view_id, camera in cameras_dict.items():
                for i, pattern in enumerate(patterns_valid):
                    procams_dict.update({"pattern": pattern})
                    if args.surface_mode is None:
                        render_image = render(camera, gaussians, pipe=None, bg_color=bg_color, procams_dict=procams_dict)['render']
                    else:
                        render_image = render_gs_to_surface(camera, gaussians, procams_dict, pipe=None, bg_color=bg_color,
                                                            surface_mode=args.surface_mode, curve_type=args.curve_type,
                                                            curve_radius=args.curve_radius, curvature=args.curvature,
                                                            surface_res=args.surface_res, auto_scale=not args.no_auto_scale,
                                                            surface_albedo=args.surface_albedo,
                                                            surface_roughness=args.surface_roughness,
                                                            ambient=args.ambient)['render']
            end_time = time.time()
            runtime = end_time - start_time
            fps = len(args.views) * len(patterns_valid) / runtime
            print(f"FPS: {fps}")
            json_file = os.path.join(args.output, "fps.json")
            with open(json_file, "w") as f:
                json.dump({"fps": fps}, f)





