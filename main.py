# -*- coding: utf-8 -*-
"""
TF2/Keras3 版 DeepFloorplan 互換ドライバ
- zlzeng/DeepFloorplan の main.py のオプション構成に合わせた CLI
- Train / Test フェーズ両対応
- データ構成（例）:
    data_root/
      images/               ... 入力画像 (.jpg/.png)
      masks/                ... 教師ラベル（multi ラベル or 個別ラベル）
         *_cw.png           ... 0/1 の境界ラベル（任意）
         *_room.png         ... 0..C-1 の部屋種別ラベル
         *_multi.png        ... まとめラベル（r を主で利用）
    もしくは YOLO-like:
      train/images, train/labels (未使用)
      val/images, val/labels     (推論可)
- Test は画像だけあれば OK（room ヘッドの argmax を出力）
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

# -------------------------
# ユーティリティ
# -------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _list_images(root: str) -> List[str]:
    pats = [
        f"{root}/images/*",
        f"{root}/train/images/*",
        f"{root}/val/images/*",
        f"{root}/*",  # 直下置き
    ]
    files = []
    for p in pats:
        files.extend([f for f in glob(p) if f.lower().endswith(IMG_EXTS)])
    return sorted(set(files))

def _match_label_path(img_path: str, lbl_root: str, suffix: str) -> Optional[str]:
    """
    画像ファイル名に対応するラベルを推定:
      - <stem>_<suffix>.png
      - <stem>.png （suffix 無し）
    """
    stem = Path(img_path).stem
    cands = [
        os.path.join(lbl_root, f"{stem}_{suffix}.png"),
        os.path.join(lbl_root, f"{stem}.png"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None

def _load_image(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    """HWC, float32[0,1], RGB"""
    data = tf.io.read_file(path)
    img  = tf.image.decode_image(data, channels=3, expand_animations=False)
    img  = tf.image.resize(img, img_size, method="bilinear")
    img  = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def _load_label(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    """HW int32"""
    data = tf.io.read_file(path)
    lab  = tf.image.decode_image(data, channels=1, expand_animations=False)
    lab  = tf.image.resize(lab, img_size, method="nearest")
    lab  = tf.squeeze(tf.cast(lab, tf.int32), axis=-1)
    return lab.numpy()

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# -------------------------
# データセット
# -------------------------

def make_dataset_train(data_root: str,
                       img_size: Tuple[int, int],
                       num_room_classes: int,
                       cw_suffix: str = "cw",
                       room_suffix: str = "room",
                       multi_suffix: str = "multi",
                       batch_size: int = 4,
                       shuffle: bool = True):
    """
    学習用: cw と room の教師を探す
    - masks/ に *_cw.png と *_room.png があれば優先
    - *_multi.png のみの場合、room を multi から読み込む（cw は未使用/ゼロ）
    """
    img_files = _list_images(data_root)
    masks_dir = os.path.join(data_root, "masks")
    pairs = []

    for img in img_files:
        room_lbl = _match_label_path(img, masks_dir, room_suffix)
        cw_lbl   = _match_label_path(img, masks_dir, cw_suffix)
        multi    = _match_label_path(img, masks_dir, multi_suffix)

        if room_lbl is None and multi is not None:
            room_lbl = multi

        if room_lbl is None:
            # 学習用は room ラベル必須
            continue

        pairs.append((img, cw_lbl, room_lbl))

    if len(pairs) == 0:
        raise RuntimeError("学習用の (画像, roomラベル) が見つかりませんでした。masks ディレクトリをご確認ください。")

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

def make_dataset_test(data_root: str,
                      img_size: Tuple[int, int],
                      batch_size: int = 1):
    img_files = _list_images(data_root)
    if len(img_files) == 0:
        raise RuntimeError("推論対象の画像が見つかりませんでした。data_root 配下の images/ などをご確認ください。")

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
# ループ
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
    ds, n_samples = make_dataset_train(cfg.data_root, (img_h, img_w),
                                       num_room_classes=cfg.num_classes,
                                       cw_suffix="close_wall",  # 既定名例: *_close_wall.png
                                       room_suffix="room",
                                       multi_suffix="multi",
                                       batch_size=cfg.batch_size,
                                       shuffle=True)
    steps_per_epoch = max(1, n_samples // cfg.batch_size)

    model = build_network(img_size=(img_h, img_w),
                          num_room_classes=cfg.num_classes,
                          num_cw_classes=2)
    optimizer = optimizers.Adam(learning_rate=cfg.lr)

    # チェックポイント
    _ensure_dir(os.path.dirname(cfg.weights))
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=os.path.dirname(cfg.weights), max_to_keep=3)

    # 既存読み込み
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

        # ログ
        log = ", ".join([f"{k}={np.mean(v):.4f}" for k, v in hist.items()])
        print(f"[Train][ep={ep+1}/{cfg.epochs}] {log}")

        # save
        save_path = ckpt_mgr.save()
        # 互換のため .keras も保存（デプロイ簡便化）
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

    # 重みロード: .ckpt 優先、失敗したら .keras
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
            # Keras の save/load 互換保持のために別 Model をロードして重みコピー
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
        pred_r = tf.argmax(logits_r, axis=-1)[0].numpy().astype(np.uint8)  # HW
        # 保存
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
    print("============================")

    if cfg.phase.lower() == "train":
        train(cfg)
    else:
        test(cfg)

if __name__ == "__main__":
    main()
