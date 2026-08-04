#!/usr/bin/env python
"""
Mobile-GS 4D environment self-check.

Run from the repo root:      python check_env_4d.py

Verifies every dependency the training path actually touches, and — crucially —
RUNS real CUDA kernels. Import success alone is not proof: an extension built
for the wrong arch imports fine and only fails at launch with
"no kernel image is available for execution on the device".

Exit code 0 = ready to train.
"""
import os
import shutil
import subprocess
import sys
import traceback

RESULTS = []          # (level, name, detail) ; level: OK / FAIL / WARN


def rec(level, name, detail=""):
    RESULTS.append((level, name, detail))
    tag = {"OK": "\033[32m OK \033[0m", "FAIL": "\033[31mFAIL\033[0m", "WARN": "\033[33mWARN\033[0m"}[level]
    print(f"[{tag}] {name}" + (f"  —  {detail}" if detail else ""))


def check_import(mod, label=None, hint=""):
    label = label or mod
    try:
        __import__(mod)
        rec("OK", label)
        return True
    except Exception as e:
        rec("FAIL", label, f"{type(e).__name__}: {e}. {hint}")
        return False


def main():
    print("=" * 72)
    print("Mobile-GS 4D environment check")
    print("=" * 72)

    # ---------- 1. interpreter ----------
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        rec("OK", f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        rec("WARN", f"Python {v.major}.{v.minor}.{v.micro}", "guide targets 3.11")

    if not os.path.exists("train_sequence.py"):
        rec("FAIL", "working directory", "run this from the Mobile-GS repo root")

    # ---------- 2. torch / GPU ----------
    try:
        import torch
        rec("OK", f"torch {torch.__version__}", f"built for CUDA {torch.version.cuda}")
    except Exception as e:
        rec("FAIL", "torch", str(e))
        summary()
        return

    if not torch.cuda.is_available():
        rec("FAIL", "CUDA available", "torch cannot see a GPU (driver / cuda build mismatch)")
        summary()
        return
    cap = torch.cuda.get_device_capability(0)
    rec("OK", f"GPU: {torch.cuda.get_device_name(0)}", f"compute capability sm_{cap[0]}{cap[1]}")

    try:
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        rec("OK", "basic CUDA op", f"matmul {float((x @ x).sum()):.1f}")
    except Exception as e:
        rec("FAIL", "basic CUDA op", str(e))

    # nvcc (needed only to BUILD the extensions, but a mismatch explains build errors)
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
            ver = [l for l in out.splitlines() if "release" in l]
            ver = ver[0].split("release")[-1].strip() if ver else "?"
            lvl = "OK" if ver.startswith("11.8") else "WARN"
            rec(lvl, f"nvcc {ver}", "" if lvl == "OK" else "guide targets 11.8 (only matters when rebuilding extensions)")
        except Exception as e:
            rec("WARN", "nvcc", str(e))
    else:
        rec("WARN", "nvcc not on PATH", "fine if extensions are already built")

    # ---------- 3. python packages ----------
    # Complete set of third-party imports in the repo, obtained by statically
    # scanning every module (requirements.txt lists only a subset).
    check_import("numpy")
    check_import("torchvision")
    check_import("tqdm")
    check_import("plyfile")
    check_import("PIL", "pillow (PIL)", hint="pip install pillow   <-- MISSING from requirements.txt")
    check_import("matplotlib", "matplotlib (imported by gaussian_renderer)",
                 hint="pip install matplotlib   <-- MISSING from requirements.txt")
    check_import("dahuffman", hint="pip install dahuffman")
    # module-level imports inside scene/gaussian_model.py -> hard requirements
    check_import("tinycudann", "tinycudann (required by scene/gaussian_model.py)",
                 hint="pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
    check_import("cupy", "cupy (required by scene/gaussian_model.py)",
                 hint="pip install cupy-cuda11x   <-- MISSING from requirements.txt")
    check_import("icecream", "icecream (required by scene/gaussian_model.py)",
                 hint="pip install icecream   <-- MISSING from requirements.txt")
    check_import("cuml", "cuml (required by scene/gaussian_model.py)",
                 hint='pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu11==25.2.*" "cuml-cu11==25.2.*"')

    # ---------- 4. CUDA extensions ----------
    ras_ms = check_import("diff_gaussian_rasterization_ms", "rasterizer: _ms (used by render_imp)",
                          hint="pip install submodules/diff-gaussian-rasterization_ms")
    check_import("diff_gaussian_rasterization_ms_nosorting", "rasterizer: _nosorting (imported by gaussian_renderer)",
                 hint="pip install submodules/diff-gaussian-rasterization_ms_nosorting  <-- MISSING from requirements.txt")
    check_import("diff_gaussian_rasterization_msori", "rasterizer: _msori",
                 hint="pip install submodules/diff-gaussian-rasterization_msori")
    knn = check_import("simple_knn._C", "simple_knn", hint="pip install submodules/simple-knn")

    # ---------- 5. run real kernels ----------
    if knn:
        try:
            from simple_knn._C import distCUDA2
            pts = torch.rand(2048, 3, device="cuda")
            d = distCUDA2(pts)
            torch.cuda.synchronize()
            rec("OK", "simple_knn CUDA kernel runs", f"mean dist2 {float(d.mean()):.5f}")
        except Exception as e:
            rec("FAIL", "simple_knn CUDA kernel", f"{e}  (rebuild with TORCH_CUDA_ARCH_LIST=\"{cap[0]}.{cap[1]}\")")

    if ras_ms:
        try:
            from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
            import math
            N, S = 64, 128
            znear, zfar, fov = 0.01, 100.0, math.radians(60)
            tan = math.tan(fov * 0.5)
            view = torch.eye(4, device="cuda")                 # camera at origin, +z forward
            proj = torch.zeros(4, 4, device="cuda")
            proj[0, 0] = 1.0 / tan
            proj[1, 1] = 1.0 / tan
            proj[2, 2] = zfar / (zfar - znear)
            proj[3, 2] = 1.0
            proj[2, 3] = -(zfar * znear) / (zfar - znear)
            full = (proj @ view).transpose(0, 1).contiguous()   # repo convention: transposed
            settings = GaussianRasterizationSettings(
                image_height=S, image_width=S, tanfovx=tan, tanfovy=tan,
                bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                viewmatrix=view.transpose(0, 1).contiguous(), projmatrix=full,
                sh_degree=1, campos=torch.zeros(3, device="cuda"),
                prefiltered=False, debug=False)
            r = GaussianRasterizer(raster_settings=settings)
            means3D = torch.randn(N, 3, device="cuda") * 0.3 + torch.tensor([0., 0., 3.], device="cuda")
            means2D = torch.zeros_like(means3D, requires_grad=True)
            shs = torch.zeros(N, 4, 3, device="cuda"); shs[:, 0, :] = 0.5
            rots = torch.zeros(N, 4, device="cuda"); rots[:, 0] = 1.0
            phi = torch.ones(N, 1, device="cuda")
            out = r(means3D=means3D, means2D=means2D, shs=shs, colors_precomp=None,
                    opacities=torch.full((N, 1), 0.7, device="cuda"),
                    theta=torch.zeros_like(phi), phi=phi,
                    scales=torch.full((N, 3), 0.08, device="cuda"),
                    rotations=rots, cov3D_precomp=None)
            img, radii = out[0], out[1]
            torch.cuda.synchronize()
            vis = int((radii > 0).sum())
            rec("OK", "Mobile-GS rasterizer kernel runs",
                f"{vis}/{N} splats on screen, image {tuple(img.shape)}")
            if vis == 0:
                rec("WARN", "rasterizer produced no visible splats",
                    "kernel executed fine; smoke-test camera only")
        except Exception as e:
            rec("FAIL", "Mobile-GS rasterizer kernel",
                f"{e}  (rebuild with TORCH_CUDA_ARCH_LIST=\"{cap[0]}.{cap[1]}\")")
            traceback.print_exc()

    # ---------- 6. repo modules ----------
    try:
        from scene import Scene, GaussianModel          # noqa: F401
        from gaussian_renderer import render_imp        # noqa: F401
        rec("OK", "repo modules import (scene + gaussian_renderer)")
    except Exception as e:
        rec("FAIL", "repo modules import", f"{type(e).__name__}: {e}")

    try:
        g = None
        from scene.gaussian_model import GaussianModel as GM
        g = GM(sh_degree=1)
        g.init_vnn()
        n = sum(p.numel() for p in g.opacity_phi_nn.parameters())
        rec("OK", "OpacityPhiNN builds", f"{n:,} params (sh_degree 1 -> input dim 22)")
    except Exception as e:
        rec("FAIL", "OpacityPhiNN builds", f"{type(e).__name__}: {e}")

    # ---------- 7. tmc3 (frame-0 compression) ----------
    tmc3 = os.path.join(".", "mpeg-pcc-tmc13", "build", "tmc3", "tmc3")
    if os.path.isfile(tmc3) and os.access(tmc3, os.X_OK):
        rec("OK", "tmc3 (GPCC) present", tmc3)
    else:
        rec("FAIL", "tmc3 (GPCC) missing",
            "needed by train.py compression; build mpeg-pcc-tmc13 at repo root (see SETUP_4D.md 2.7)")

    summary()


def summary():
    print("\n" + "=" * 72)
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    if not fails:
        print(f"READY TO TRAIN  ({len(RESULTS) - len(warns)} checks passed"
              + (f", {len(warns)} warnings)" if warns else ")"))
        print("\nNext: prepare the dataset (SETUP_4D.md section 3), then run step 1.")
        sys.exit(0)
    print(f"NOT READY — {len(fails)} problem(s):\n")
    for _, name, detail in fails:
        print(f"  * {name}\n      {detail}")
    print("\nSee SETUP_4D.md section 6 (troubleshooting).")
    sys.exit(1)


if __name__ == "__main__":
    main()
