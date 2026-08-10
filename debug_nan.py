#!/usr/bin/env python
"""
Single-iteration NaN locator for train.py.

Reproduces train.py's setup exactly (checkpoint restore -> SH downgrade ->
init_vnn), runs ONE render + teacher render, and reports which tensor / loss
term first becomes non-finite.

    python debug_nan.py -s <dataset> --images frames/000000 -m <model> --eval \
        --start_checkpoint <model>/chkpnt30000.pth
"""
import sys
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, GaussianModel
from scene.gaussian_teacher import TeaGaussianModel
from gaussian_renderer import render_imp, render_teacher
from utils.loss_utils import l1_loss, ssim, scale_invariant_loss
from utils.general_utils import safe_state


def stat(name, t):
    if t is None:
        print(f"  {name:<28} None")
        return False
    t = t.detach().float()
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    finite = t[torch.isfinite(t)]
    rng = f"[{finite.min():.4g}, {finite.max():.4g}]" if finite.numel() else "(no finite values)"
    flag = "  <<< BAD" if (n_nan or n_inf) else ""
    print(f"  {name:<28} shape={tuple(t.shape)} nan={n_nan} inf={n_inf} range={rng}{flag}")
    return bool(n_nan or n_inf)


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    # ---- identical to train.py's startup ----
    gaussians = GaussianModel(sh_degree=dataset.sh_degree, training_args=opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    model_params, _ = torch.load(args.start_checkpoint)
    teacher_params, _ = torch.load(args.start_checkpoint)
    gaussians_tea = TeaGaussianModel(sh_degree=3)
    gaussians_tea.restore(teacher_params)
    opt_dict = gaussians.restore(model_params, opt)
    opt_dict = gaussians.filter_optimizer_state(opt_dict)
    gaussians.onedownSHdegree()
    gaussians.init_vnn(opt)
    gaussians.training_setup(opt)
    gaussians.optimizer.load_state_dict(opt_dict)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    print("\n=== gaussian state after restore ===")
    print(f"  count = {len(gaussians._xyz):,}   active_sh={gaussians.active_sh_degree} "
          f"max_sh={gaussians.max_sh_degree}")
    bad = False
    bad |= stat("_xyz", gaussians._xyz)
    bad |= stat("get_scaling", gaussians.get_scaling)
    bad |= stat("get_rotation", gaussians.get_rotation)
    bad |= stat("get_opacity (raw sigmoid)", gaussians.get_opacity)
    bad |= stat("get_features (SH)", gaussians.get_features)
    if bad:
        print("\n>>> The restored model itself already contains NaN/Inf. "
              "The checkpoint from pretrain.py is the problem, not train.py.\n")

    cam = scene.getTrainCameras()[0]
    print(f"\n=== camera: {cam.image_name}  {cam.image_width}x{cam.image_height} ===")
    stat("gt image", cam.original_image)

    print("\n=== MLP output ===")
    xyz = gaussians.get_xyz
    dir_pp = xyz - cam.camera_center.cuda().repeat(xyz.shape[0], 1)
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    phi, opacity = gaussians.opacity_phi_nn(
        gaussians.get_features, gaussians.get_scaling, xyz, dir_pp, gaussians.get_rotation)
    stat("phi", phi)
    stat("opacity (MLP)", opacity)

    print("\n=== student render (render_imp) ===")
    pkg = render_imp(cam, gaussians, pipe, bg)
    stat("render", pkg["render"])
    stat("render_depth", pkg["render_depth"])
    stat("opacity (returned)", pkg.get("opacity"))
    stat("radii", pkg["radii"].float())
    print(f"  visible splats: {int((pkg['radii'] > 0).sum()):,}")

    print("\n=== teacher render ===")
    tpkg = render_teacher(cam, gaussians_tea, pipe, bg)
    stat("teacher render", tpkg["render"])
    stat("teacher depth", tpkg["render_depth"])

    print("\n=== loss terms ===")
    gt = cam.original_image.cuda()
    terms = {}
    terms["L1"] = l1_loss(pkg["render"], gt)
    terms["1-SSIM"] = 1.0 - ssim(pkg["render"], gt)
    terms["distill L1"] = l1_loss(pkg["render"], tpkg["render"])
    terms["depth (scale-inv)"] = scale_invariant_loss(pkg["render_depth"], tpkg["render_depth"], mask=None)
    for k, v in terms.items():
        f = torch.isfinite(v).all()
        print(f"  {k:<28} {float(v):.6g}{'' if f else '   <<< BAD'}")

    total = ((1.0 - opt.lambda_dssim) * terms["L1"]
             + opt.lambda_dssim * terms["1-SSIM"]
             + opt.lambda_distill * terms["distill L1"]
             + 2 * opt.lambda_depth * terms["depth (scale-inv)"])
    print(f"\n  TOTAL  {float(total):.6g}"
          f"   (lambda_dssim={opt.lambda_dssim}, lambda_distill={opt.lambda_distill}, "
          f"lambda_depth={opt.lambda_depth})")

    print("\n=== verdict ===")
    culprits = [k for k, v in terms.items() if not torch.isfinite(v).all()]
    if culprits:
        print("  non-finite loss term(s):", ", ".join(culprits))
        if "depth (scale-inv)" in culprits:
            print("  -> workaround: train.py ... --lambda_depth 0")
        if "distill L1" in culprits:
            print("  -> workaround: train.py ... --lambda_distill 0")
    elif not torch.isfinite(total):
        print("  individual terms are finite but the total is not (check lambdas)")
    else:
        print("  all finite at iteration 0 — the NaN appears LATER in training;")
        print("  re-run train.py and note the iteration where the progress bar turns nan.")


if __name__ == "__main__":
    main()
