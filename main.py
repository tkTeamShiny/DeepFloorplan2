# -*- coding: utf-8 -*-
"""
TF2/Keras3 版 DeepFloorplan ドライバ（データ構成を自動検出）
対応するディレクトリ例：
  [A] r3d_index 方式：
      data_root/
        r3d_train.txt
        r3d_val.txt     (任意)
        r3d_test.txt    (任意)
      - 行形式:
          img_path, room_mask_path[, cw_mask_path]
        もしくは空白区切り、相対パス可（data_root 起点）

  [B] YOLO/flat 方式：
      data_root/
        train/images/*.jpg|png
        train/masks/*.png              (任意)
        train/labels/*.png             (任意)
        images/*.jpg|png               (任意)
        masks/*.png                    (任意)
      - ラベル解決規則:
         <stem><mask_suffix> を最優先（例: 1000001_multi.png）
         次候補: <stem>_room.png / <stem>.png なども探索

CLI 例:
  Train:
    !python main.py --phase Train --data_root /content/YOLO_SetA_seg_no1_部屋種別改変版_flat \
      --img_size 512 512 --num_classes 23 --mask_suffix _multi.png \
      --weights runs/checkpoints/best.ckpt
  Test:
    !python main.py --phase Test --data_root /content/... --weights runs/checkpoints/best.ckpt
"""

from __future__ import annotations
import os
import argparse
from glob import glob
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from net import build_network

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # INFO/WARN 抑制

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# -------------------------
# パスユーティリティ
# -------------------------

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _is_image(p: str) -> bool:
    return p.lower().endswith(IMG_EXTS)

def _read_index_file(idx_path: str) -> List[List[str]]:
    lines = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            # カンマ優先、次に空白
            if "," in s:
                cols = [c.strip() for c in s.split(",")]
            else:
                cols = s.split()
            if len(cols) < 2:
                continue
            lines.append(cols)
    return lines

def _resolve_rel(p: str, root: str) -> str:
    q = Path(p)
    if not q.is_absolute():
        q = Path(root) / q
    return str(q)

def _list_images_candidates(root: str) -> List[str]:
    # YOLO/flat の代表的場所を総当たりで探索
    pats = [
        f"{root}/train/images/*",
        f"{root}/images/*",
        f"{root}/val/images/*",
        f"{root}/test/images/*",
        f"{root}/*",
    ]
    files = []
    for p in pats:
        files.extend([f for f in glob(p) if _is_image(f)])
    return sorted(set(files))

def _label_search_dirs(img_path: str) -> List[str]:
    # 画像と同階層 / 親階層の masks / labels / annotations も探索
    img_dir = Path(img_path).parent
    dirs = [
        img_dir,
        img_dir / "masks",
        img_dir / "labels",
        img_dir.parent / "masks",
        img_dir.parent / "labels",
        img_dir.parent / "annotations",
        img_dir.parent,  # 親直下
    ]
    uniq = []
    for d in dirs:
        d = str(d)
        if d not in uniq:
            uniq.append(d)
    return uniq

def _find_label_for_image(img_path: str,
                          mask_suffix: str,
                          fallback_suffixes: List[str]) -> Optional[str]:
    """
    優先順：
      1) <stem><mask_suffix>（例: _multi.png）
      2) fallback_suffixes（例: _room.png, .png, _rooms.png など）
    検索場所：画像と同階層 → masks → labels → 親/masks → 親/labels → 親直下
    """
    stem = Path(img_path).stem
    search_dirs = _label_search_dirs(img_path)

    # 1) 明示 mask_suffix
    cand_names = [f"{stem}{mask_suffix}"] if mask_suffix else []
    # 2) フォールバック
    cand_names += [f"{stem}{suf}" for suf in fallback_suffixes]

    for d in search_dirs:
        for name in cand_names:
            c = str(Path(d) / name)
            if os.path.isfile(c):
                return c
    return None

# -------------------------
# 画像 / ラベル読み込み
# -------------------------

def _load_image(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    data = tf.io.read_file(path)
    img  = tf.image.decode_image(data, channels=3, expand_animations=False)
    img  = tf.image.resize(img, img_size, method="bilinear")
    img  = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def _load_label(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    data = tf.io.read_file(path)
    lab  = tf.image.decode_image(data, channels=1, expand_animations=False)
    lab  = tf.image.resize(lab, img_size, method="nearest")
    lab  = tf.squeeze(tf.cast(lab, tf.int32), axis=-1)
    return lab.numpy()

# -------------------------
# データセット構築（Train）
# -------------------------

def make_dataset_train(data_root: str,
                       img_size: Tuple[int, int],
                       num_room_classes: int,
                       mask_suffix: str = "_multi.png",
                       cw_suffix_hint: Optional[str] = None,
                       batch_size: int = 4,
                       shuffle: bool = True):
    """
    優先: r3d_train.txt を使う
      - 2列:   img, room
      - 3列:   img, room, cw  または img, cw, room（自動判別）
    次点: 画像を列挙して <stem><mask_suffix> を探す（無ければ _room.png などを試す）
    """
    root = data_root
    idx_path = Path(root) / "r3d_train.txt"
    pairs = []

    if idx_path.is_file():
        rows = _read_index_file(str(idx_path))
        print(f"[Train] r3d index found: {idx_path} (rows={len(rows)})")
        for cols in rows:
            # 2列 or 3列に対応
            if len(cols) == 2:
                img_p, room_p = cols[0], cols[1]
                cw_p = None
            else:
                # 3列は room/cw の順が不定なので推測
                a, b, c = cols[0], cols[1], cols[2]
                # img は画像拡張子を含むはず
                img_p = a
                # b, c のうち「room らしい方」を room とみなす（_room / _multi / rooms を優先）
                if any(x in b.lower() for x in ["_room", "room", "multi", "rooms"]):
                    room_p, cw_p = b, c
                elif any(x in c.lower() for x in ["_room", "room", "multi", "rooms"]):
                    room_p, cw_p = c, b
                else:
                    # よく分からない場合は b=room, c=cw と仮定
                    room_p, cw_p = b, c

            img_abs  = _resolve_rel(img_p, root)
            room_abs = _resolve_rel(room_p, root)
            cw_abs   = _resolve_rel(cw_p, root) if cw_p else None

            if not os.path.isfile(img_abs):
                print(f"[Warn] image not found: {img_abs} -> skip")
                continue
            if not os.path.isfile(room_abs):
                print(f"[Warn] room label not found: {room_abs} -> skip")
                continue
            if cw_abs and (not os.path.isfile(cw_abs)):
                print(f"[Warn] cw label not found: {cw_abs} -> ignore cw")
                cw_abs = None

            pairs.append((img_abs, cw_abs, room_abs))

    else:
        # 画像列挙してラベルを自動発見
        imgs = _list_images_candidates(root)
        print(f"[Train] images discovered: {len(imgs)}")
        if len(imgs) == 0:
            raise RuntimeError("学習用の画像が見つかりませんでした。data_root 配下をご確認ください。")

        # フォールバック候補（順序重要）
        fb = []
        if mask_suffix and mask_suffix != "_multi.png":
            fb.append("_multi.png")
        fb += ["_room.png", "_rooms.png", ".png"]  # 例: 1000001.png（部屋ラベルのみ）

        for img in imgs:
            room_lbl = _find_label_for_image(img, mask_suffix=mask_suffix, fallback_suffixes=fb)
            if room_lbl is None:
                # 見つからない場合はスキップ（学習用は room 必須）
                continue

            # cw は任意。ヒントがあれば同様に探索
            cw_lbl = None
            cw_candidates = []
            if cw_suffix_hint:  # 例: "_cw.png" や "_close_wall.png"
                cw_candidates.append(cw_suffix_hint)
            cw_candidates += ["_cw.png", "_closewall.png", "_close_wall.png", "_boundary.png"]
            for suf in cw_candidates:
                cw_try = _find_label_for_image(img, mask_suffix=suf, fallback_suffixes=[])
                if cw_try is not None:
                    cw_lbl = cw_try
                    break

            pairs.append((img, cw_lbl, room_lbl))

    if len(pairs) == 0:
        raise RuntimeError("学習用の (画像, roomラベル) が見つかりませんでした。アップロード構成（r3d_train.txt / *_multi.png など）をご確認ください。")

    print(f"[Train] usable pairs: {len(pairs)}")
    # 先頭数件を表示
    for i in range(min(5, len(pairs))):
        print(f"  [sample#{i+1}] img={pairs[i][0]}\n               room={pairs[i][2]}\n               cw={pairs[i][1]}")

    def _gen():
        for img, cw_lbl, room_lbl in pairs:
            img_np  = _load_image(img, img_size)
            room_np = _load_label(room_lbl, img_size)
            if cw_lbl is not None:
                cw_np = _load_label(cw_lbl, img_size)
            else:
                cw_np = np.zeros_like(room_np, dtype=np.int32)
            yield img_np, cw_np, room_np

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size[0], img_size[1]), dtype=tf.int32),
            tf.TensorSpec(shape=(img_size[0], img_size[1]), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(200, len(pairs)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(pairs)

# -------------------------
# データセット構築（Test）
# -------------------------

def make_dataset_test(data_root: str,
                      img_size: Tuple[int, int],
                      batch_size: int = 1):
    root = data_root
    # r3d_test.txt / r3d_val.txt があれば、それらの1列目（画像）を読み込む
    idx_test = Path(root) / "r3d_test.txt"
    idx_val  = Path(root) / "r3d_val.txt"

    img_files = []
    picked_idx = None
    if idx_test.is_file():
        rows = _read_index_file(str(idx_test))
        picked_idx = idx_test
        for cols in rows:
            if len(cols) == 0:
                continue
            img_files.append(_resolve_rel(cols[0], root))
    elif idx_val.is_file():
        rows = _read_index_file(str(idx_val))
        picked_idx = idx_val
        for cols in rows:
            if len(cols) == 0:
                continue
            img_files.append(_resolve_rel(cols[0], root))
    else:
        img_files = _list_images_candidates(root)

    img_files = [p for p in img_files if os.path.isfile(p)]
    if len(img_files) == 0:
        raise RuntimeError("推論対象の画像が見つかりませんでした。data_root 配下をご確認ください。")
    if picked_idx:
        print(f"[Test] index used: {picked_idx} (images={len(img_files)})")
    else:
        print(f"[Test] images discovered: {len(img_files)})")

    def _gen():
        for p in img_files:
            img_np = _load_image(p, img_size)
            yield img_np, p

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, img_files

# -------------------------
# 学習 / 推論ループ
# -------------------------

@tf.function(jit_compile=False)
def _train_step(model, optimizer, images, labels_cw, labels_r):
    with tf.GradientTape() as tape:
        logits_cw, logits_r = model(images, training=True)
        losses = model.losses_dict(labels_cw, labels_r, logits_cw, logits_r)
    grads = tape.gradient(losses["loss_total"], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return losses

def train(cfg):
    img_h, img_w = cfg.img_size
    ds, n_samples = make_dataset_train(
        cfg.data_root, (img_h, img_w),
        num_room_classes=cfg.num_classes,
        mask_suffix=cfg.mask_suffix,
        cw_suffix_hint=cfg.cw_suffix_hint,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    steps_per_epoch = max(1, n_samples // cfg.batch_size)

    model = build_network(img_size=(img_h, img_w),
                          num_room_classes=cfg.num_classes,
                          num_cw_classes=2)
    optimizer = optimizers.Adam(learning_rate=cfg.lr)

    _ensure_dir(os.path.dirname(cfg.weights))
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=os.path.dirname(cfg.weights), max_to_keep=3)

    if os.path.isfile(cfg.weights):
        try:
            ckpt.restore(cfg.weights)
            print(f"[Info] restored from {cfg.weights}")
        except Exception as e:
            print(f"[Warn] failed to restore: {e}")

    for ep in range(cfg.epochs):
        hist = {"loss_total": [], "loss_cw": [], "loss_r": []}
        for batch in ds:
            images, labels_cw, labels_r = batch
            losses = _train_step(model, optimizer, images, labels_cw, labels_r)
            for k in hist.keys():
                hist[k].append(float(losses[k].numpy()))

        log = ", ".join([f"{k}={np.mean(v):.4f}" for k, v in hist.items()])
        print(f"[Train][ep={ep+1}/{cfg.epochs}] {log}")

        save_path = ckpt_mgr.save()
        try:
            model.save(cfg.weights.replace(".ckpt", ".keras"))
        except Exception:
            pass
        print(f"[Info] checkpoint saved to {save_path}")

def test(cfg):
    img_h, img_w = cfg.img_size
    ds, files = make_dataset_test(cfg.data_root, (img_h, img_w), batch_size=1)

    model = build_network(img_size=(img_h, img_w),
                          num_room_classes=cfg.num_classes,
                          num_cw_classes=2)

    loaded = False
    if os.path.isfile(cfg.weights):
        try:
            tf.train.Checkpoint(model=model).restore(cfg.weights).expect_partial()
            print(f"[Info] loaded ckpt: {cfg.weights}")
            loaded = True
        except Exception as e:
            print(f"[Warn] ckpt restore failed: {e}")
    keras_path = cfg.weights if cfg.weights.endswith(".keras") else cfg.weights + ".keras"
    if (not loaded) and os.path.isfile(keras_path):
        try:
            m2 = tf.keras.models.load_model(keras_path, compile=False)
            for w_src, w_dst in zip(m2.weights, model.weights):
                w_dst.assign(w_src)
            print(f"[Info] loaded keras: {keras_path}")
            loaded = True
        except Exception as e:
            print(f"[Warn] keras load failed: {e}")

    if not loaded:
        print("[Warn] weights not found. Using randomly initialized weights.")

    _ensure_dir(cfg.save_pred)

    for images, pth in ds:
        logits_cw, logits_r = model(images, training=False)
        pred_r = tf.argmax(logits_r, axis=-1)[0].numpy().astype(np.uint8)
        stem = Path(bytes(pth.numpy()[0]).decode()).stem
        out_path = os.path.join(cfg.save_pred, f"{stem}_room_pred.png")
        tf.keras.utils.save_img(out_path, pred_r, scale=False)
        print(f"[Save] {out_path}")

# -------------------------
# エントリポイント
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, default="Train", choices=["Train", "Test"])
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, nargs=2, default=[512, 512])
    ap.add_argument("--num_classes", type=int, default=23)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default="runs/checkpoints/best.ckpt")
    ap.add_argument("--save_pred", type=str, default="runs/preds")

    # 追加：アップロード構成に合わせるための可変パラメータ
    ap.add_argument("--mask_suffix", type=str, default="_multi.png",
                    help="部屋ラベルのサフィックス（例: _multi.png, _room.png）")
    ap.add_argument("--cw_suffix_hint", type=str, default=None,
                    help="境界ラベルのサフィックスヒント（例: _cw.png, _close_wall.png）")
    return ap.parse_args()

def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    print("========== Config ==========")
    print(f"phase: {cfg.phase}")
    print(f"data_root: {cfg.data_root}")
    print(f"img_size: {tuple(cfg.img_size)}")
    print(f"num_classes: {cfg.num_classes}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"epochs: {cfg.epochs}")
    print(f"lr: {cfg.lr}")
    print(f"out_dir: {cfg.out_dir}")
    print(f"seed: {cfg.seed}")
    print(f"weights: {cfg.weights}")
    print(f"save_pred: {cfg.save_pred}")
    print(f"mask_suffix: {cfg.mask_suffix}")
    print(f"cw_suffix_hint: {cfg.cw_suffix_hint}")
    print("============================")

    if cfg.phase.lower() == "train":
        train(cfg)
    else:
        test(cfg)

if __name__ == "__main__":
    main()
