# Mobile-GS 4D 訓練完整說明(Ubuntu + RTX 4090)

從零開始:環境建置 → 資料集擺放 → 分段訓練 → 輸出。搭配 `README_4D.md`(設計說明)使用。

---

## 1. 系統需求

| 項目 | 需求 |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| GPU | RTX 4090(sm_89,CUDA 11.8 有支援)|
| CUDA Toolkit | **11.8**(編譯 rasterizer 擴充用,需 `nvcc`)|
| Python | **3.11**(conda)|
| 磁碟 | 每段 15 秒約需 10–15 GB(影像 + 逐幀 PLY)|

---

## 2. 環境建置

```bash
# 2.1 取得程式碼(注意 4dgs-sequence 分支)
git clone https://github.com/BillSavart/Mobile-GS.git -b 4dgs-sequence
cd Mobile-GS

# 2.2 conda 環境
conda create -n mobile-gs python==3.11 -y
conda activate mobile-gs

# 2.3 PyTorch(cu118)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 2.4 確認 nvcc 是 11.8(編譯擴充時用到)
nvcc --version   # 若非 11.8,安裝 CUDA Toolkit 11.8 並把它排在 PATH 前面

# 2.5 官方相依(會編譯 CUDA 擴充,幾分鐘)
export TORCH_CUDA_ARCH_LIST="8.9"       # 4090,加速編譯
pip install -r requirements.txt

# 2.6 ⚠️ 官方 requirements 漏掉的:sort-free 光柵化器(訓練必須!)
pip install submodules/diff-gaussian-rasterization_ms_nosorting

# 2.7 GPCC 壓縮器 tmc3(frame 0 的 train.py 壓縮階段需要)
#     預設路徑寫死為 ./mpeg-pcc-tmc13/build/tmc3/tmc3(相對 repo 根目錄)
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13 && mkdir build && cd build && cmake .. && make -j && cd ../..
ls mpeg-pcc-tmc13/build/tmc3/tmc3    # 確認存在
```

### 2.8 驗證安裝(一行搞定)

```bash
python check_env_4d.py       # 必須在 repo 根目錄執行
```

會逐項檢查並印出 PASS/FAIL:Python/torch/GPU、nvcc、四個 CUDA 擴充、tinycudann、cuml、
tmc3,並且**實際執行 CUDA kernel**(simple_knn + Mobile-GS 光柵化器)。

> 為什麼要跑 kernel:擴充若是用錯誤的 GPU 架構編譯,`import` 會成功,直到訓練跑下去才炸
> `no kernel image is available for execution on the device`。這支腳本會當場抓出來。

全綠(exit code 0)才代表可以開始訓練;有 FAIL 時腳本會直接告訴你缺什麼、指令是什麼。

---

## 3. 資料集擺放

### 3.1 目錄結構(共用一份 COLMAP 校正 + 逐幀影像資料夾)

```
<root>/                          # 例:~/data/mycapture_seg0
  sparse/0/
    cameras.bin                  # 共用校正(整段只有一份)
    images.bin
    points3D.bin
  frames/
    000000/                      # 第 0 幀:每台相機一張
      cam00.jpg
      cam01.jpg
      ...
    000001/
      cam00.jpg
      ...
    000449/                      # 15 秒 @ 30fps = 450 幀(000000–000449)
```

### 3.2 鐵則

1. **影像檔名必須與 COLMAP 內註冊的影像名完全一致**(`images.bin` 裡是 `cam00.jpg`,每個 frames/XXXXXX/ 裡就要叫 `cam00.jpg`)。
2. **所有幀共用同一份校正** → 相機在拍攝期間不能動(固定多相機陣列)。
3. 相機模型需為 **PINHOLE / SIMPLE_PINHOLE**(已去畸變)。若用 COLMAP 自算,跑完 `colmap image_undistorter` 再把去畸變結果放進來。

### 3.3 從多視角影片產生逐幀資料夾(ffmpeg)

```bash
# 每台相機的影片切成幀(30fps),再重組成 frames/{frame}/{cam}.jpg
for cam in cam00 cam01 cam02; do
  mkdir -p tmp/$cam
  ffmpeg -i videos/$cam.mp4 -vf fps=30 -q:v 2 tmp/$cam/%06d.jpg
done
python - <<'EOF'
import os, shutil, glob
cams = sorted(os.listdir('tmp'))
n = len(glob.glob(f'tmp/{cams[0]}/*.jpg'))
for f in range(n):
    d = f'frames/{f:06d}'; os.makedirs(d, exist_ok=True)
    for c in cams:
        shutil.move(f'tmp/{c}/{f+1:06d}.jpg', f'{d}/{c}.jpg')
EOF
```

> 注意 ffmpeg 輸出從 000001 起算,上面腳本已對齊成 frames/000000 開始。

### 3.4 校正來源二選一

- **已有校正**(例如 DualGS 用的那套):轉成 COLMAP 的 `sparse/0`(cameras/images/points3D),影像名對齊即可。
- **沒有校正**:對第 0 幀跑 COLMAP(feature_extractor → matcher → mapper → image_undistorter),把 `sparse/0` 放到 `<root>/`。

---

## 4. 分段訓練(每段 15 秒 = 450 幀)

```bash
cd Mobile-GS
ROOT=~/data/mycapture_seg0        # 資料集根目錄
OUT=out/seg0                      # 本段輸出

# ── 步驟 1:第 0 幀(原始 Mobile-GS 配方:pretrain → finetune+壓縮)──
# 人物/室內用 --imp_metric indoor,室外場景用 outdoor
python pretrain.py -s $ROOT --images frames/000000 -m ${OUT}_f0 --eval \
    --imp_metric indoor --sh_degree 3 --iterations 30000
python train.py -s $ROOT --images frames/000000 -m ${OUT}_f0 --eval \
    --start_checkpoint ${OUT}_f0/chkpnt30000.pth
# ⏱ 4090 上兩步合計約 1–2 小時(一段只需做一次)

# ── 步驟 2:抽出種子(純張量 PLY + 共用 MLP)──
python train_sequence.py --extract_frame0 -s $ROOT --images frames/000000 \
    -m ${OUT}_f0 --out_root $OUT

# ── 步驟 3:序列訓練(暖啟動、凍結 MLP、固定拓撲)──
python train_sequence.py -s $ROOT -m ${OUT}_f0 --out_root $OUT \
    --frame_start 1 --frame_end 449 --iters 1000
# ⏱ 約 30–60 秒/幀 → 450 幀約 4–8 小時
```

**中斷續跑**:`--frame_start` 從斷掉的幀接續即可(自動從前一幀的 PLY 暖啟動)。
**動作快糊掉** → `--iters` 提高(如 2000);**幾乎靜止** → 降低(如 500)。

---

## 5. 輸出結構(拷回 Windows / Unity 用)

```
out/seg0/
  opacity_phi_nn.pt              # 共用 MLP(整段一顆)
  frame_000000/point_cloud.ply   # 每幀點數固定
  frame_000001/point_cloud.ply
  ...
out/seg0_f0/
  comp.xz                        # 第 0 幀壓縮模型(含 MLP,Unity 匯出也會用到)
  storage.txt / cameras.json ...
```

需要拷回的:`out/seg0/` 整個資料夾 + `out/seg0_f0/comp.xz`。

---

## 6. 常見問題排查

| 症狀 | 原因 / 解法 |
|---|---|
| `No module named diff_gaussian_rasterization_ms_nosorting` | 步驟 2.6 沒做(官方 requirements 漏了它)|
| 編譯擴充時 `no kernel image / sm_89` 錯誤 | `export TORCH_CUDA_ARCH_LIST="8.9"` 後重裝 submodules |
| `tmc3: not found` 或壓縮階段 crash | tmc3 沒編好,或不是從 repo 根目錄執行(路徑寫死 `./mpeg-pcc-tmc13/...`)|
| 讀不到影像 / 相機數不對 | frames/XXXXXX/ 內檔名與 COLMAP 註冊名不一致(3.2 鐵則 1)|
| cuml 安裝失敗 | 參考 RAPIDS 安裝指南;確認用 `--extra-index-url=https://pypi.nvidia.com` |
| `--extract_frame0` 在 decode 階段報錯 | 這是新程式碼最可能出問題的點——把完整錯誤訊息回報,修正後會推上分支 |

---

## 7. 訓練規模建議

- **先跑一段 15 秒 pilot 打通全流程**,再批量開段。
- 各段獨立(各自的 frame 0 + 種子),可**多段平行**在不同 GPU/時段跑,誤差也不會跨段累積。
- 全長 30–40 分鐘 = 120–160 段;以每段約 6–10 小時計,單張 4090 需數週——建議依內容優先序分批訓練。
