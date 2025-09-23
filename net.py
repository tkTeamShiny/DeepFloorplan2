# -*- coding: utf-8 -*-
"""
DeepFloorplan2 - net.py
- main.py から呼び出される共通モジュール
- *_multi.png（索引 or RGB）に対応
- TF2/Keras の軽量U-Netを内蔵（必要に応じてここを書き換えてOK）
"""

import os, glob, random
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ========== 乱数固定 ==========
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ========== I/O ==========
def read_image(path, target_size):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        h, w = target_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img

def _rgb_to_index(rgb_mask):
    """
    RGBマスクをユニーク色→インデックスに変換。
    注意: 画像ごとに色→IDが異なるとクラス意味がズレるため、
          通常は *_multi.png を単一チャネル（索引）として出力しておくのが安全。
    """
    if rgb_mask.ndim == 2:
        return rgb_mask
    h, w, c = rgb_mask.shape
    assert c == 3, "mask RGB expected"
    flat = rgb_mask.reshape(-1, 3)
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    idx = inv.reshape(h, w).astype(np.int32)
    return idx

def read_mask(path, target_size, num_classes=None):
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    # 1ch 索引か、3ch RGB のいずれかを想定
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        m = _rgb_to_index(m)
    if target_size is not None:
        h, w = target_size
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = m.astype(np.int32)
    if num_classes is not None:
        m = np.clip(m, 0, num_classes - 1)
    return m

def pair_image_mask_lists(images_dir, labels_dir, mask_suffix="_multi.png"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    img_paths = sorted([p for p in images_dir.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    pairs = []
    for ip in img_paths:
        stem = ip.stem
        mp = labels_dir / f"{stem}{mask_suffix}"
        if mp.exists():
            pairs.append((str(ip), str(mp)))
    return pairs

def discover_splits(data_root: str):
    data_root = Path(data_root)
    tr_img = data_root / "train" / "images"
    tr_lab = data_root / "train" / "labels"
    va_img = data_root / "val" / "images"
    va_lab = data_root / "val" / "labels"

    train_pairs = pair_image_mask_lists(tr_img, tr_lab)
    val_pairs = pair_image_mask_lists(va_img, va_lab) if va_img.exists() else []

    if not train_pairs:
        raise RuntimeError(f"No train pairs found under {tr_img} / {tr_lab}")

    if not val_pairs:
        # 9:1 分割（先頭を val）
        n = len(train_pairs)
        k = max(1, int(n * 0.1))
        val_pairs = train_pairs[:k]
        train_pairs = train_pairs[k:]
        print(f"[Info] val split not found -> use split: train={len(train_pairs)} val={len(val_pairs)}")
    else:
        print(f"[Info] discovered: train={len(train_pairs)} val={len(val_pairs)}")
    return train_pairs, val_pairs

# ========== tf.data ==========
def make_tf_dataset(pairs, img_size, num_classes, batch_size, shuffle, aug=False):
    H, W = img_size

    def _load(ip, mp):
        ip = ip.decode("utf-8")
        mp = mp.decode("utf-8")
        img = read_image(ip, (H, W))
        msk = read_mask(mp, (H, W), num_classes)
        return img, msk

    def _tf_map(ip, mp):
        img, msk = tf.numpy_function(_load, [ip, mp], [tf.float32, tf.int32])
        img.set_shape([H, W, 3])
        msk.set_shape([H, W])
        if aug:
            # 左右/上下フリップのみ（元図面の幾何学一貫性を壊しにくい簡易Aug）
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_left_right(img)
                msk = tf.image.flip_left_right(msk[..., None])[..., 0]
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_up_down(img)
                msk = tf.image.flip_up_down(msk[..., None])[..., 0]
        return img, msk

    ips = [p[0] for p in pairs]
    mps = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((ips, mps))
    if shuffle:
        ds = ds.shuffle(len(pairs), reshuffle_each_iteration=True)
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

# ========== モデル ==========
def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def _down(x, filters):
    c = _conv_block(x, filters)
    p = layers.MaxPool2D()(c)
    return c, p

def _up(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = _conv_block(x, filters)
    return x

def build_model(input_shape, num_classes):
    """
    既定: 軽量U-Net
    - 必要ならここを既存のバックボーンに差し替え可
    """
    inputs = layers.Input(shape=input_shape)
    c1, p1 = _down(inputs, 32)
    c2, p2 = _down(p1, 64)
    c3, p3 = _down(p2, 128)
    c4, p4 = _down(p3, 256)
    bn = _conv_block(p4, 512)

    u1 = _up(bn, c4, 256)
    u2 = _up(u1, c3, 128)
    u3 = _up(u2, c2, 64)
    u4 = _up(u3, c1, 32)

    logits = layers.Conv2D(num_classes, 1, padding="same", name="logits")(u4)  # from_logits=True 前提
    model = keras.Model(inputs, logits, name="UNetLite")
    return model

# ========== メトリクス ==========
class MeanIoU(tf.keras.metrics.Metric):
    """
    Keras MeanIoU を logits + sparse label で扱いやすくするラッパ
    """
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.miou.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.miou.result()

    def reset_state(self):
        self.miou.reset_state()

# ========== 可視化補助 ==========
def colorize_index_mask(index_mask, num_classes):
    import matplotlib
    cmap = matplotlib.cm.get_cmap("tab20", num_classes)
    rgb = (cmap(index_mask)[..., :3] * 255).astype(np.uint8)
    return rgb
