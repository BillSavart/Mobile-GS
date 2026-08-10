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
    parser.add_argument("--start_iter", type=int, default=1,
                        help="iteration number to start the lr schedule at; train.py "
                             "resumes at the checkpoint's iteration (30000), which "
                             "gives a different lr than starting from 1")
    parser.add_argument("--steps", type=int, default=0,
                        help="also run N training steps, checking params/grads each "
                             "iteration and stopping at the first non-finite value")
    parser.add_argument("--no_opt_state", action="store_true",
                        help="skip optimizer.load_state_dict (tests whether the "
                             "restored Adam moments are what blows up)")
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
    if args.no_opt_state:
        print(">>> SKIPPING optimizer.load_state_dict (--no_opt_state)")
    else:
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

    print("\n=== verdict (forward pass) ===")
    culprits = [k for k, v in terms.items() if not torch.isfinite(v).all()]
    if culprits:
        print("  non-finite loss term(s):", ", ".join(culprits))
        if "depth (scale-inv)" in culprits:
            print("  -> workaround: train.py ... --lambda_depth 0")
        if "distill L1" in culprits:
            print("  -> workaround: train.py ... --lambda_distill 0")
        return
    if not torch.isfinite(total):
        print("  individual terms are finite but the total is not (check lambdas)")
        return
    print("  forward pass is clean at iteration 0.")

    if args.steps <= 0:
        print("  Re-run with --steps 200 to find which step/tensor breaks.")
        return

    # ------------------------------------------------------------------
    # Training-step loop: the forward pass is fine, so the failure is in the
    # backward / optimizer update. Check params and grads EVERY iteration and
    # stop at the first non-finite value, naming the tensor.
    # ------------------------------------------------------------------
    print(f"\n=== running {args.steps} training steps ===")
    print(f"  spatial_lr_scale = {gaussians.spatial_lr_scale}")
    for g in gaussians.optimizer.param_groups:
        print(f"  lr[{g['name']:<16}] = {g['lr']:.3e}")

    from random import randint
    named = lambda: [(n, getattr(gaussians, n)) for n in
                     ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]]
    cams = scene.getTrainCameras()
    stack = []
    for it in range(1, args.steps + 1):
        gaussians.update_learning_rate(args.start_iter + it - 1)
        if not stack:
            stack = cams.copy()
        c = stack.pop(randint(0, len(stack) - 1))

        p = render_imp(c, gaussians, pipe, bg)
        tp = render_teacher(c, gaussians_tea, pipe, bg)
        g_ = c.original_image.cuda()
        loss = ((1.0 - opt.lambda_dssim) * l1_loss(p["render"], g_)
                + opt.lambda_dssim * (1.0 - ssim(p["render"], g_))
                + opt.lambda_distill * l1_loss(p["render"], tp["render"])
                + 2 * opt.lambda_depth * scale_invariant_loss(p["render_depth"], tp["render_depth"]))
        loss.backward()

        broke = None
        if not torch.isfinite(loss):
            broke = "loss (forward)"
        if broke is None:
            for n, t in named():
                if t.grad is not None and not torch.isfinite(t.grad).all():
                    broke = f"GRADIENT of {n}"
                    break
        if broke:
            print(f"\n  >>> step {it} ({c.image_name}): FIRST non-finite = {broke}")
            print(f"      loss={float(loss):.6g}")
            if broke == "loss (forward)":
                # Which produced the nan: the render, the depth, or a loss term?
                print("\n  -- forward breakdown at the failing step --")
                stat("student render", p["render"])
                stat("student render_depth", p["render_depth"])
                stat("teacher render", tp["render"])
                stat("teacher render_depth", tp["render_depth"])
                stat("MLP opacity", p.get("opacity"))
                d = p["render_depth"].detach()
                dpos = d[d > 0]
                print(f"     student depth >0 min = {float(dpos.min()) if dpos.numel() else float('nan'):.6g}"
                      f"   negatives = {int((d < 0).sum())}")
                for nm, fn in [
                        ("L1", lambda: l1_loss(p["render"], g_)),
                        ("1-SSIM", lambda: 1.0 - ssim(p["render"], g_)),
                        ("distill L1", lambda: l1_loss(p["render"], tp["render"])),
                        ("depth (scale-inv)", lambda: scale_invariant_loss(p["render_depth"], tp["render_depth"]))]:
                    with torch.no_grad():
                        v = fn()
                    print(f"     {nm:<20} {float(v):.6g}" + ("   <<< BAD" if not torch.isfinite(v) else ""))
                # Degenerate-gaussian hunt: the NaN is LOCALIZED (few hundred pixels)
                # and the teacher render is clean, so a handful of gaussians have been
                # driven into a state the rasterizer cannot handle.
                with torch.no_grad():
                    q = gaussians._rotation
                    qn = q.norm(dim=1)
                    s = gaussians.get_scaling
                    smax = s.max(dim=1)[0]
                    smin = s.min(dim=1)[0]
                    aniso = smax / smin.clamp_min(1e-20)
                    rad = p["radii"].float()
                    print(f"     quat norm:  min={float(qn.min()):.3e}  "
                          f"#(<1e-3)={int((qn < 1e-3).sum())}  #(nonfinite)={int((~torch.isfinite(qn)).sum())}")
                    print(f"     scaling:    min={float(s.min()):.3e}  max={float(s.max()):.3e}  "
                          f"#(>10)={int((smax > 10).sum())}")
                    print(f"     anisotropy: max={float(aniso.max()):.3e}  "
                          f"#(>1e5)={int((aniso > 1e5).sum())}   <- near-singular 3D covariance")
                    print(f"     radii:      max={float(rad.max()):.0f}  "
                          f"#(>2000)={int((rad > 2000).sum())}   (image is {p['render'].shape[-1]} px wide)")
                    print(f"     opacity raw: min={float(gaussians._opacity.min()):.3f} "
                          f"max={float(gaussians._opacity.max()):.3f}")

                # Per-gaussian quantities feeding the Mobile-GS weight exp(maxScale/depth)
                with torch.no_grad():
                    xyz_ = gaussians.get_xyz
                    zc = ((xyz_ - c.camera_center.cuda()) @ torch.tensor(
                        c.R, dtype=torch.float32, device="cuda"))[:, 2]
                    front = zc[zc > 0]
                    print(f"     view-space z: min={float(zc.min()):.4g} "
                          f"min_positive={float(front.min()) if front.numel() else float('nan'):.4g} "
                          f"behind_camera={int((zc <= 0).sum())}")
                    ms = gaussians.get_scaling.max(dim=1)[0]
                    print(f"     max scale: {float(ms.max()):.4g}   "
                          f"worst exp(maxS/z) exponent ~ "
                          f"{float((ms[zc > 0] / front.clamp_min(1e-6)).max()) if front.numel() else float('nan'):.4g}"
                          f"   (>88 overflows fp32)")
            for n, t in named():
                stat(f"{n}", t)
                if t.grad is not None:
                    stat(f"{n}.grad", t.grad)
            print("\n  Forward was clean, so this is a gradient explosion, not bad data.")
            print("  Try, in order:")
            print("    1) --no_opt_state         (restored Adam moments mismatched)")
            print("    2) --lambda_depth 0       (scale-invariant depth grads)")
            print("    3) --lambda_distill 0")
            return

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.opacity_nn_optimizer.step()
        gaussians.opacity_nn_optimizer.zero_grad(set_to_none=True)

        for n, t in named():
            if not torch.isfinite(t).all():
                print(f"\n  >>> step {it}: parameter '{n}' became non-finite AFTER optimizer.step()")
                stat(n, t)
                print("\n  The update itself diverged (learning rate / Adam state).")
                print("  Try: --no_opt_state, then a lower --position_lr_init.")
                return

        if it % 10 == 0 or it == 1:
            print(f"  step {it:4d}  loss={float(loss):.6f}")

    print(f"\n  {args.steps} steps completed with no NaN.")
    print("  The failure must occur later — likely at a pruning boundary")
    print(f"  (every {opt.pruning_interval} iters) or at net_itr/svq_itr.")


if __name__ == "__main__":
    main()
