import torch


def stereo_poses(baseline_m, device):
    w2c = torch.eye(4).repeat(2, 1, 1)
    w2c[1, 0, 3] = -baseline_m
    w2c = w2c.to(device)
    w2c = w2c @ torch.linalg.inv(w2c[0])
    return torch.linalg.inv(torch.linalg.inv(w2c))


def default_intrinsics(fov_rad):
    f = 0.5 / torch.tan(0.5 * torch.tensor([fov_rad]))
    k = f.new_zeros((1, 3, 3))
    k[:, torch.eye(3, dtype=bool)] = torch.stack([f, f, torch.ones_like(f)], dim=-1)
    k[:, :, -1] = torch.tensor([0.5, 0.5, 1.0])
    return k


def _to_hom(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def _cam2world(x, pose34):
    pose = torch.eye(4, device=pose34.device)[None].repeat(pose34.shape[0], 1, 1)
    pose[:, :3, :] = pose34
    return _to_hom(x) @ torch.linalg.inv(pose)[..., :3, :4].transpose(-1, -2)


def _plucker_one(w2c, fov_rad, size):
    k = default_intrinsics(fov_rad).to(w2c.device).clone()
    k[:, :2] *= w2c.new_tensor([size, size]).view(1, -1, 1)
    ys = torch.arange(size, dtype=torch.float32).add_(0.5)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    grid = _to_hom(torch.stack([xx, yy], dim=-1).view(-1, 2))
    grid_cam = grid.to(k.device) @ k.float().inverse().transpose(-1, -2)
    pose = w2c[:, :3, :]
    points = _cam2world(grid_cam, pose)
    centers = _cam2world(torch.zeros_like(grid_cam), pose)
    rays = torch.nn.functional.normalize(points - centers, dim=-1)
    out = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    return out.permute(0, 2, 1).reshape(1, -1, size, size)[0]


def plucker(w2cs, fov_rad, size):
    return torch.stack([_plucker_one(p.view(1, 4, 4), fov_rad, size) for p in w2cs], dim=0)
