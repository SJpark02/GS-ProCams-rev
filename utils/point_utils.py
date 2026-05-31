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


def make_surface_points(view, surface_mode, curve_type="cylindrical",
                        curve_radius=1.0, curvature=0.5, surface_res=None):
    """Generate an analytic surface in front of the camera and return per-pixel
    world-space points + normals, replacing the depth-based surface.

    Args:
        view:         the viewpoint camera (provides intrinsics/extrinsics).
        surface_mode: one of {"sphere", "hemisphere", "curved"}.
        curve_type:   for surface_mode == "curved", one of
                      {"cylindrical", "sinusoidal", "parabolic"}.
        curve_radius: characteristic radius of the surface (sphere/cylinder radius,
                      or spatial scale for sinusoidal/parabolic).
        curvature:    curvature/amplitude parameter (used by curved surfaces).
        surface_res:  optional sampling resolution hint (currently the surface is
                      sampled at the camera resolution; reserved for future use).

    Returns:
        surf_pts3d:  (H, W, 3) world-space surface points (0 where no hit).
        surf_normal: (H, W, 3) world-space normals facing the camera (0 where no hit).
        hit_mask:    (H, W) boolean mask of valid surface hits.
    """
    # `surface_res` is accepted for API completeness; the analytic surface is sampled
    # at the camera's native resolution to stay aligned with the procams pipeline.
    _ = surface_res

    # Place the surface center a sensible distance in front of the camera.
    distance = max(float(curve_radius) * 2.0, 1.0)
    center, _forward = _surface_center(view, distance)

    if surface_mode == "sphere":
        return _make_sphere(view, center, float(curve_radius), hemisphere=False)
    elif surface_mode == "hemisphere":
        return _make_sphere(view, center, float(curve_radius), hemisphere=True)
    elif surface_mode == "curved":
        return _make_curved(view, center, curve_type, float(curve_radius), float(curvature))
    else:
        raise ValueError(
            "Unsupported surface_mode: {}. Expected one of: 'sphere', 'hemisphere', 'curved'.".format(surface_mode)
        )
