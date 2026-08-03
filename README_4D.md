# Mobile-GS 4D: per-frame sequence training (volumetric video)

Trains a multi-view **video** as a sequence of Mobile-GS frames that share **one
opacity/phi MLP** and a **fixed gaussian topology**, so playback streams only the
per-splat tensors through the sort-free renderer (fixed-size frames, delta-friendly).

## Dataset layout (shared calibration, per-frame image folders)

```
<root>/
  sparse/0/...                 # shared COLMAP calibration (one for the whole capture)
  frames/000000/<cam>.jpg      # frame 0: one image per camera; FILENAMES MUST MATCH
  frames/000001/<cam>.jpg      #          the image names registered in COLMAP
  ...
```

If your layout differs, adjust `--images_pattern` (default `frames/{:06d}`).

## Workflow per segment (e.g. 15 s @ 30 fps = 450 frames)

```bash
# 1) Frame 0 with the ORIGINAL Mobile-GS recipe (pretrain + finetune + compress)
python pretrain.py -s <root> --images frames/000000 -m out/seg0_f0 --eval \
       --imp_metric outdoor --sh_degree 3 --iterations 30000
python train.py -s <root> --images frames/000000 -m out/seg0_f0 --eval \
       --start_checkpoint out/seg0_f0/chkpnt30000.pth

# 2) Extract the plain-tensor seed + shared MLP from the compressed frame-0 model
python train_sequence.py --extract_frame0 -s <root> --images frames/000000 \
       -m out/seg0_f0 --out_root out/seg0

# 3) Train the remaining frames (warm start from previous frame, frozen MLP)
python train_sequence.py -s <root> -m out/seg0_f0 --out_root out/seg0 \
       --frame_start 1 --frame_end 449 --iters 1000
```

Output:

```
out/seg0/opacity_phi_nn.pt          # ONE shared MLP for the whole sequence
out/seg0/frame_000000/point_cloud.ply
out/seg0/frame_000001/point_cloud.ply
...
```

## Design decisions (v1)

- **Frozen MLP after frame 0** — playback keeps a single MLP on device; only splat
  data streams. (The MLP maps SH/viewdir/scale/rot -> phi/opacity and generalizes
  across neighboring frames of the same scene.)
- **Fixed topology** — no densify/prune/SVQ on sequence frames; every frame has the
  same gaussian count, which keeps runtime buffers fixed-size and makes inter-frame
  compression (delta/keyframe) straightforward later.
- **Photometric-only loss on sequence frames** (L1 + SSIM). The teacher/distill
  machinery is a frame-0 (from-scratch) concern; warm-started frames converge in
  ~1k iters without it.
- Per-frame outputs are raw PLY for now; the Unity exporter consumes
  per-frame PLY + the shared MLP. Sequence-level compression is a later stage.

## Tuning

- `--iters` per frame: start at 1000; raise if fast motion blurs, lower if static.
- Segment boundaries: retrain frame 0 per segment (new seed) — keeps error from
  accumulating across minutes and parallelizes segments across GPUs/machines.
