import torch
from utils.system_utils import torch_compile

@torch_compile
def depths_to_points(view, depthmap):
    c2w = view.c2w
    intrins = view.K
    grid_x, grid_y = torch.meshgrid(torch.arange(view.image_width, device='cuda', dtype=torch.float), torch.arange(view.image_height, device='cuda', dtype=torch.float), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3] # view.camera_center
    points = depthmap.reshape(-1, 1) * rays_d + rays_o 
    return points.reshape(*depthmap.shape[1:3], 3) # pts3d (H, W, 3)

@torch_compile
def points_to_normal(points):
    r"""
    Args:
        view: view camera
        points: 3D point per pixel
    Returns:
        psedo_normal: (H, W, 3)
    """
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output 

@torch_compile
def depth_to_normal(view, depth):
    """
    Args:
        view: view camera
        depth: depthmap 
    Returns:
        psedo_normal: (H, W, 3)
    """
    points = depths_to_points(view, depth)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

@torch_compile
def warp_points(view, points):
    """
    Args:
        view: src view camera
        points: 3D points in dst camera frame (H, W, 3)
    Returns:
        prj2cam_grid (H, W, 2)
    """
    H, W, _ = points.shape
    pts3d = points.view(-1, 3)
    pts3d_homo = torch.cat((pts3d, torch.ones_like(pts3d[:, :1])), dim=-1)

    view_pts3d = pts3d_homo @ view.world_view_transform[:, :3] # (H*W, 4) @ (4, 3) = (H*W, 3)

    uvw = view_pts3d @ view.K.T # (H*W, 3) @ (3, 3) = (H*W, 3)
    uv = uvw[:, :2] / uvw[:, 2:3] # de-homo, raster/pixel space (H*W, 2)
    return uv.view(H, W, 2), view_pts3d.view(H, W, 3)


# =====================================================================================
# Synthetic surface generation
# -------------------------------------------------------------------------------------
# These helpers replace the depth-based surface (`surf_pts3d`/`surf_normal`) used in the
# default `render()` path with an *analytic* surface (sphere / hemisphere / curved).
# This enables simulating how a projected pattern would look on a virtual surface placed
# in front of the camera, instead of only on the real geometry reconstructed from depth.
#
# All points are returned in WORLD coordinates and in the same (H, W, 3) layout as
# `depths_to_points`, so the downstream procams pipeline (warp_points -> grid_sample ->
# BRDF) can be reused without modification.
# =====================================================================================

def _camera_rays(view):
    """Generate per-pixel ray origins / directions in WORLD space for a camera.

    Mirrors the ray construction used in `depths_to_points` so that the synthetic
    surface lives in the exact same coordinate frame as the depth-based one.

    Returns:
        rays_o: (3,)      world-space ray origin (camera center)
        rays_d: (H*W, 3)  normalized world-space ray directions
        H, W:   ints      image height / width
    """
    c2w = view.c2w
    intrins = view.K
    H, W = int(view.image_height), int(view.image_width)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device='cuda', dtype=torch.float),
        torch.arange(H, device='cuda', dtype=torch.float),
        indexing='xy',
    )
    pix = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pix @ intrins.inverse().T @ c2w[:3, :3].T  # (H*W, 3)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    rays_o = c2w[:3, 3]  # (3,) camera center in world
    return rays_o, rays_d, H, W


def _surface_center(view, distance):
    """Default surface placement: a point `distance` units in front of the camera
    along its optical (viewing) axis, expressed in world coordinates.

    Returns (center, forward) where `forward` is the unit viewing direction.
    """
    c2w = view.c2w
    cam_center = c2w[:3, 3]
    # The camera looks down +Z in its local frame (OpenCV/COLMAP convention used here).
    forward = c2w[:3, :3] @ torch.tensor([0.0, 0.0, 1.0], device='cuda', dtype=torch.float)
    forward = torch.nn.functional.normalize(forward, dim=0)
    return cam_center + forward * distance, forward


def _ray_sphere_intersection(rays_o, rays_d, center, radius):
    """Intersect rays with a sphere. Returns (t, hit_mask).

    Picks the nearest positive intersection (front face of the sphere as seen from
    the camera). `t` is set to 0 where there is no valid hit; `hit_mask` flags valid
    intersections.
    """
    oc = rays_o[None, :] - center[None, :]          # (N, 3)
    b = torch.sum(oc * rays_d, dim=-1)              # (N,)
    c = torch.sum(oc * oc, dim=-1) - radius ** 2    # (N,)
    disc = b * b - c                                # (N,)
    hit_mask = disc > 0
    sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc
    # nearest positive root
    t = torch.where(t0 > 1e-6, t0, t1)
    hit_mask = hit_mask & (t > 1e-6)
    t = torch.where(hit_mask, t, torch.zeros_like(t))
    return t, hit_mask


def _make_sphere(view, center, radius, hemisphere=False):
    """Sphere / hemisphere surface via analytic ray-sphere intersection.

    Returns surf_pts3d (H, W, 3), surf_normal (H, W, 3), hit_mask (H, W).
    Normals point toward the camera (so the lit side faces the viewer).
    """
    rays_o, rays_d, H, W = _camera_rays(view)
    t, hit_mask = _ray_sphere_intersection(rays_o, rays_d, center, radius)
    pts = rays_o[None, :] + t[:, None] * rays_d        # (N, 3)

    # Outward normal of the sphere
    normal = torch.nn.functional.normalize(pts - center[None, :], dim=-1)
    # Flip normals to face the camera (toward ray origin)
    to_cam = torch.nn.functional.normalize(rays_o[None, :] - pts, dim=-1)
    flip = (torch.sum(normal * to_cam, dim=-1, keepdim=True) < 0).float()
    normal = normal * (1.0 - 2.0 * flip)

    if hemisphere:
        # Keep only the half of the sphere facing the camera (front-facing hemisphere).
        _, forward = _surface_center(view, 0.0)
        # Signed position of the hit point relative to the center along the camera-to-surface axis.
        along = torch.sum((pts - center[None, :]) * (-forward)[None, :], dim=-1)
        hemi_mask = along >= 0.0
        hit_mask = hit_mask & hemi_mask

    pts = pts * hit_mask[:, None].float()
    normal = normal * hit_mask[:, None].float()
    return pts.view(H, W, 3), normal.view(H, W, 3), hit_mask.view(H, W)


def _make_curved(view, center, curve_type, radius, curvature):
    """Curved surfaces defined as implicit functions f(p)=0 in a local frame
    aligned with the camera, solved by ray marching.

    Supported curve types:
        - cylindrical: a cylinder of given radius whose axis is vertical (camera-up).
        - sinusoidal:  a base plane modulated by a sine wave (wrinkled screen).
        - parabolic:   a paraboloid bowl (depth increases with off-axis distance).

    Returns surf_pts3d (H, W, 3), surf_normal (H, W, 3), hit_mask (H, W).
    """
    rays_o, rays_d, H, W = _camera_rays(view)
    c2w = view.c2w

    # Local camera-aligned orthonormal basis (right, up, forward) in world space.
    right = torch.nn.functional.normalize(c2w[:3, 0], dim=0)
    up = torch.nn.functional.normalize(c2w[:3, 1], dim=0)
    forward = torch.nn.functional.normalize(c2w[:3, 2], dim=0)

    def to_local(p):
        # p: (N, 3) world -> local (u=right, v=up, w=forward) relative to surface center
        d = p - center[None, :]
        u = torch.sum(d * right[None, :], dim=-1)
        v = torch.sum(d * up[None, :], dim=-1)
        w = torch.sum(d * forward[None, :], dim=-1)
        return u, v, w

    def sdf(p):
        """Implicit field f(p). Surface is f(p) = 0; a sign change marks a crossing."""
        u, v, w = to_local(p)
        if curve_type == "cylindrical":
            # Cylinder with vertical (up) axis, centered at the surface center.
            # Distance from the axis in the (u, w) plane minus radius.
            r = torch.sqrt(u * u + w * w + 1e-12)
            return r - radius
        elif curve_type == "sinusoidal":
            # Base plane at w = 0 modulated by a sine wave along u.
            # amplitude ~ curvature, spatial frequency ~ 1/radius.
            freq = 1.0 / max(radius, 1e-3)
            disp = curvature * torch.sin(u * freq)
            return w - disp
        elif curve_type == "parabolic":
            # Paraboloid bowl: w = curvature * (u^2 + v^2) / (2 * radius)
            disp = curvature * (u * u + v * v) / (2.0 * max(radius, 1e-3))
            return w - disp
        else:
            raise ValueError("Unsupported curve_type: {}".format(curve_type))

    # ---- Ray marching to find the first sign change of the implicit field ----
    n_samples = 256
    t_near = 1e-3
    t_far = 4.0 * max(radius, abs(curvature), 1.0)
    ts = torch.linspace(t_near, t_far, n_samples, device='cuda')  # (S,)

    prev_p = rays_o[None, :] + ts[0] * rays_d                     # (N, 3)
    prev_f = sdf(prev_p)                                         # (N,)
    N = rays_d.shape[0]
    t_hit = torch.zeros(N, device='cuda')
    hit_mask = torch.zeros(N, dtype=torch.bool, device='cuda')

    for i in range(1, n_samples):
        cur_p = rays_o[None, :] + ts[i] * rays_d
        cur_f = sdf(cur_p)
        crossing = (prev_f * cur_f < 0) & (~hit_mask)
        if crossing.any():
            # Linear interpolation for sub-step accuracy at the crossing.
            denom = (prev_f - cur_f)
            denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
            alpha = prev_f / denom
            t_interp = ts[i - 1] + alpha * (ts[i] - ts[i - 1])
            t_hit = torch.where(crossing, t_interp, t_hit)
            hit_mask = hit_mask | crossing
        prev_f = cur_f

    pts = rays_o[None, :] + t_hit[:, None] * rays_d

    # ---- Normal via finite differences of the implicit field (gradient) ----
    eps = max(radius, 1.0) * 1e-3
    ex = torch.tensor([eps, 0.0, 0.0], device='cuda')
    ey = torch.tensor([0.0, eps, 0.0], device='cuda')
    ez = torch.tensor([0.0, 0.0, eps], device='cuda')
    nx = sdf(pts + ex[None, :]) - sdf(pts - ex[None, :])
    ny = sdf(pts + ey[None, :]) - sdf(pts - ey[None, :])
    nz = sdf(pts + ez[None, :]) - sdf(pts - ez[None, :])
    normal = torch.nn.functional.normalize(torch.stack([nx, ny, nz], dim=-1), dim=-1)
    # Orient normals toward the camera.
    to_cam = torch.nn.functional.normalize(rays_o[None, :] - pts, dim=-1)
    flip = (torch.sum(normal * to_cam, dim=-1, keepdim=True) < 0).float()
    normal = normal * (1.0 - 2.0 * flip)

    pts = pts * hit_mask[:, None].float()
    normal = normal * hit_mask[:, None].float()
    return pts.view(H, W, 3), normal.view(H, W, 3), hit_mask.view(H, W)


def _scene_stats(view, scene_points):
    """Estimate, from the trained Gaussian point cloud, where to place and how
    large to make the synthetic surface so it fills the camera view.

    Returns (center, depth, extent, frustum_half):
        center:       (3,) world point on the optical axis at the scene depth.
        depth:        scalar along-axis distance from camera to that center.
        extent:       scalar robust half-size (radius) of the scene point cloud.
        frustum_half: scalar half-width of the view frustum at `depth`
                      (= depth * tan(FoV/2)); a surface of this radius fills view.
    """
    c2w = view.c2w
    cam_center = c2w[:3, 3]
    forward = torch.nn.functional.normalize(c2w[:3, :3] @ torch.tensor(
        [0.0, 0.0, 1.0], device=scene_points.device, dtype=scene_points.dtype), dim=0)

    pts = scene_points
    # Robust centroid (median) to ignore stray Gaussians / floaters.
    centroid = torch.median(pts, dim=0).values                       # (3,)
    # Along-axis depth of the centroid in front of the camera.
    rel = pts - cam_center[None, :]                                  # (N, 3)
    along = rel @ forward                                            # (N,)
    depth = torch.median(along).clamp(min=1e-3)                      # scalar
    axis_center = cam_center + forward * depth

    # Scene extent: robust radius around the centroid (use a sampling of
    # percentiles so a few floaters do not blow up the size).
    d = torch.linalg.norm(pts - centroid[None, :], dim=-1)           # (N,)
    try:
        extent = torch.quantile(d, 0.9)
    except Exception:
        extent = d.mean() + d.std()
    extent = extent.clamp(min=1e-3)

    # Frustum half-width at the scene depth (depth * tan(FoV/2)); used only as an
    # upper bound so the surface never grows larger than the visible view.
    import math as _math
    half_fov = 0.5 * min(float(view.FoVx), float(view.FoVy))
    frustum_half = depth * _math.tan(half_fov)
    if isinstance(frustum_half, torch.Tensor):
        frustum_half = frustum_half.clamp(min=1e-3)
    # Return the *true 3D centroid* as the center so the surface sits exactly on
    # the object (and therefore inside the projector's illuminated region).
    return centroid, axis_center, depth, extent, frustum_half


def make_surface_points(view, surface_mode, curve_type="cylindrical",
                        curve_radius=1.0, curvature=0.5, surface_res=None,
                        scene_points=None, auto_scale=True):
    """Generate an analytic surface in front of the camera and return per-pixel
    world-space points + normals, replacing the depth-based surface.

    Args:
        view:         the viewpoint camera (provides intrinsics/extrinsics).
        surface_mode: one of {"sphere", "hemisphere", "curved"}.
        curve_type:   for surface_mode == "curved", one of
                      {"cylindrical", "sinusoidal", "parabolic"}.
        curve_radius: characteristic radius of the surface. When `auto_scale` is
                      on and `scene_points` is given, this acts as a *relative*
                      multiplier on the auto-estimated scene size (so the default
                      1.0 fills the view); otherwise it is an absolute world-unit
                      radius.
        curvature:    curvature/amplitude parameter (used by curved surfaces).
        surface_res:  optional sampling resolution hint (reserved).
        scene_points: optional (N, 3) tensor of the trained Gaussian centers,
                      used to auto-place/auto-scale the surface to the scene.
        auto_scale:   if True and scene_points is provided, derive the surface
                      center + radius from the scene geometry so the surface fills
                      the camera view regardless of the dataset's coordinate scale.

    Returns:
        surf_pts3d:  (H, W, 3) world-space surface points (0 where no hit).
        surf_normal: (H, W, 3) world-space normals facing the camera (0 where no hit).
        hit_mask:    (H, W) boolean mask of valid surface hits.
    """
    _ = surface_res  # accepted for API completeness.

    use_auto = bool(auto_scale) and (scene_points is not None) and (scene_points.numel() > 0)
    if use_auto:
        # Derive placement + size from the actual scene so the surface fills the
        # view in whatever coordinate scale the dataset uses. `curve_radius` and
        # `curvature` become relative multipliers in this mode.
        centroid, axis_center, depth, extent, frustum_half = _scene_stats(view, scene_points)
        # Size the surface to the OBJECT (scene extent), not the whole frustum:
        # the projector only illuminates the object's region, so a surface much
        # larger than the object would fall outside the projector and render
        # black. Cap at the frustum half-width so it still fits the view.
        base = float(extent)
        base = min(base, float(frustum_half))
        radius = float(curve_radius) * base
        radius = max(radius, 1e-3)
        # Center the surface on the real object so it lies within the projector's
        # coverage. Keep the camera safely outside the sphere.
        center = centroid
        cam_center = view.c2w[:3, 3]
        cam_to_center = float(torch.linalg.norm(center - cam_center))
        if cam_to_center <= radius:
            # Object is closer than the radius would allow; shrink so the camera
            # stays outside (otherwise every ray starts inside the sphere).
            radius = max(cam_to_center * 0.8, 1e-3)
        eff_curvature = float(curvature) * base
    else:
        # Absolute mode: place the surface a sensible distance in front of camera.
        distance = max(float(curve_radius) * 2.0, 1.0)
        center, _forward = _surface_center(view, distance)
        radius = float(curve_radius)
        eff_curvature = float(curvature)

    if surface_mode == "sphere":
        return _make_sphere(view, center, radius, hemisphere=False)
    elif surface_mode == "hemisphere":
        return _make_sphere(view, center, radius, hemisphere=True)
    elif surface_mode == "curved":
        return _make_curved(view, center, curve_type, radius, eff_curvature)
    else:
        raise ValueError(
            "Unsupported surface_mode: {}. Expected one of: 'sphere', 'hemisphere', 'curved'.".format(surface_mode)
        )
