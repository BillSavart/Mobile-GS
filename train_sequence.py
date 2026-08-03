#
# Mobile-GS 4D: per-frame sequence training driver.
#
# Trains a multi-view VIDEO as a sequence of Mobile-GS frames that share ONE
# opacity/phi MLP and a FIXED gaussian topology, so playback streams only the
# per-splat tensors (fixed-size frames, delta-friendly) through the sort-free
# renderer.
#
# Dataset layout (shared calibration, per-frame image folders):
#   <root>/sparse/0/...                shared COLMAP (one calibration)
#   <root>/frames/000000/<cam>.jpg     frame 0 images; filenames must match the
#   <root>/frames/000001/<cam>.jpg     image names registered in COLMAP
#
# Workflow per segment (e.g. 15 s = 450 frames):
#   1) Train frame 0 with the ORIGINAL Mobile-GS recipe (pretrain.py + train.py)
#      on --images frames/000000  ->  <seed_model>/comp.xz
#   2) Extract a plain-tensor seed (writes seed_ply + shared MLP weights):
#        python train_sequence.py --extract_frame0 -s <root> --images frames/000000 \
#               -m <seed_model> --out_root <out_seg>
#   3) Train the rest of the segment (warm start, frozen MLP, fixed topology):
#        python train_sequence.py -s <root> -m <seed_model> --out_root <out_seg> \
#               --frame_start 1 --frame_end 449 --iters 1000
#
# Output:
#   <out_seg>/opacity_phi_nn.pt            shared MLP (one per sequence)
#   <out_seg>/frame_000000/point_cloud.ply per-frame gaussians (fixed count)
#   <out_seg>/frame_000001/point_cloud.ply ...
#
import os
import sys
import torch
from random import randint
from argparse import ArgumentParser

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_imp
from utils.general_utils import safe_state
from tqdm import tqdm


def load_frame_cameras(args, frame_images_dir):
    """Cameras for one frame: shared COLMAP poses + that frame's image folder."""
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, frame_images_dir, False)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)


def freeze_mlp(gaussians):
    for p in gaussians.opacity_phi_nn.parameters():
        p.requires_grad = False


def releaf(gaussians):
    """Ensure per-splat tensors are trainable leaf Parameters (a decoded model
    may carry plain tensors)."""
    import torch.nn as nn
    for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        t = getattr(gaussians, name)
        if not isinstance(t, nn.Parameter):
            setattr(gaussians, name, nn.Parameter(t.detach().clone().requires_grad_(True)))


def extract_frame0(dataset, out_root, images_dir):
    """Decode the frame-0 comp.xz model into a plain-tensor seed + shared MLP."""
    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    gaussians.init_vnn()
    dataset.images = images_dir
    Scene(dataset, gaussians, load_iteration=-1, shuffle=False, decode=True)
    os.makedirs(out_root, exist_ok=True)
    seed_dir = os.path.join(out_root, "frame_000000")
    os.makedirs(seed_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(seed_dir, "point_cloud.ply"))
    torch.save(gaussians.opacity_phi_nn.state_dict(),
               os.path.join(out_root, "opacity_phi_nn.pt"))
    print(f"[seq] seed written: {seed_dir}  gaussians={len(gaussians._xyz)}")


def train_sequence(dataset, opt, pipe, args):
    out_root = args.out_root
    mlp_path = os.path.join(out_root, "opacity_phi_nn.pt")
    assert os.path.exists(mlp_path), \
        f"missing shared MLP {mlp_path} — run --extract_frame0 first"

    # Model: warm-start from the previous frame's ply; MLP shared and frozen.
    gaussians = GaussianModel(sh_degree=dataset.sh_degree, training_args=opt)
    gaussians.init_vnn()  # no optimizer for the MLP: frozen by construction
    gaussians.opacity_phi_nn.load_state_dict(torch.load(mlp_path, weights_only=True))
    freeze_mlp(gaussians)

    prev_dir = os.path.join(out_root, f"frame_{args.frame_start - 1:06d}")
    gaussians.load_ply(os.path.join(prev_dir, "point_cloud.ply"))
    releaf(gaussians)
    print(f"[seq] warm start from {prev_dir}  gaussians={len(gaussians._xyz)}")

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    for frame in range(args.frame_start, args.frame_end + 1):
        frame_images = args.images_pattern.format(frame)
        cams = load_frame_cameras(dataset, frame_images)

        # Fresh optimizer per frame (fixed topology: NO densify/prune/svq/net —
        # the compression stages stay a frame-0 concern; sequence frames only
        # move/reshape/recolor the existing gaussians).
        gaussians.spatial_lr_scale = max(gaussians.spatial_lr_scale, 1e-8)
        gaussians.training_setup(opt)

        iters = args.iters
        stack = []
        bar = tqdm(range(1, iters + 1), desc=f"frame {frame:06d}")
        for it in bar:
            gaussians.update_learning_rate(min(it * (opt.iterations // max(iters, 1)), opt.iterations))
            if not stack:
                stack = cams.copy()
            cam = stack.pop(randint(0, len(stack) - 1))

            pkg = render_imp(cam, gaussians, pipe, bg)
            image = pkg["render"]
            gt = cam.original_image.cuda()
            loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) \
                 + opt.lambda_dssim * (1.0 - ssim(image, gt))
            loss.backward()

            with torch.no_grad():
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if it % 50 == 0:
                    bar.set_postfix({"loss": f"{loss.item():.5f}"})

        frame_dir = os.path.join(out_root, f"frame_{frame:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        gaussians.save_ply(os.path.join(frame_dir, "point_cloud.ply"))
        del cams
        torch.cuda.empty_cache()

    print(f"[seq] done: frames {args.frame_start}..{args.frame_end} -> {out_root}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Mobile-GS 4D sequence training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_root", type=str, required=True,
                        help="sequence output root (frame_XXXXXX/ subdirs + shared MLP)")
    parser.add_argument("--images_pattern", type=str, default="frames/{:06d}",
                        help="per-frame images subfolder pattern under source_path")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1000,
                        help="optimization iterations per frame (warm-started)")
    parser.add_argument("--extract_frame0", action="store_true",
                        help="decode the comp.xz model at -m into out_root seed")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    dataset = lp.extract(args)
    if args.extract_frame0:
        extract_frame0(dataset, args.out_root, args.images_pattern.format(0))
    else:
        train_sequence(dataset, op.extract(args), pp.extract(args), args)
