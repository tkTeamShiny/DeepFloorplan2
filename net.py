# -*- coding: utf-8 -*-
"""
DeepFloorplan2 - net.py
- *_multi.png（索引 or RGB→索引）に対応
- 任意構成/フラット配置データを再帰探索して <ID> と <ID>_multi.png をペアリング
- TF2/Keras の軽量U-Netを内蔵（必要に応じて build_model を差し替えてOK）
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
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {p}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        h, w = target_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img

def _rgb_to_index(rgb_mask):
    """
    RGBマスク → ユニーク色インデックス化
    *安全策*: 本来は全データで同一定義の索引マスクを用意するのがベスト
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
    p = Path(path)
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"failed to read mask: {p}")
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

# ========== データ探索 ==========
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _rglob_files(root: Path, exts):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def pair_image_mask_lists_generic(root: Path, mask_suffix: str = "_multi.png"):
    """
    再帰的に画像と <stem> + mask_suffix を探索し、(image, mask) のリストを返す
    - 画像: *_IMG_EXTS
    - マスク: 同一stem + mask_suffix（場所はどこでもOK、優先は同ディレクトリ→labels/masks ディレクトリ）
    """
    root = Path(root)
    img_files = _rglob_files(root, _IMG_EXTS)
    # マスク候補の高速ルックアップ用
    #  stem -> [absolute paths of candidates]
    mask_map = {}
    for p in root.rglob(f"*{mask_suffix}"):
        if p.is_file():
            mask_map.setdefault(p.stem.replace(mask_suffix.replace(".png", ""), ""), []).append(p)

    pairs = []
    for ip in sorted(img_files):
        stem = ip.stem
        # 優先度: 同ディレクトリ /labels /masks をざっくり担保（候補が複数なら距離が近い順）
        cands = mask_map.get(stem, [])
        if not cands:
            # 例: 画像が a/b/c/xxx.jpg のとき、xxx_multi.png を近隣に探すフォールバック
            local = list(ip.parent.glob(f"{stem}{mask_suffix}"))
            if local:
                cands = local
        if cands:
            # 近いパスを優先
            cands_sorted = sorted(cands, key=lambda p: len(os.path.relpath(str(p), start=str(ip.parent))))
            pairs.append((str(ip), str(cands_sorted[0])))
    return pairs

def _has_yolo_layout(root: Path):
    return (root / "train" / "images").exists() and (root / "train" / "labels").exists()

def _pair_yolo_like(root: Path, mask_suffix: str):
    tr = (root / "train" / "images", root / "train" / "labels")
    va = (root / "val" / "images", root / "val" / "labels")
    def pair_in(img_dir, lab_dir):
        img_paths = sorted([p for p in Path(img_dir).glob("*.*") if p.suffix.lower() in _IMG_EXTS])
        pairs = []
        for ip in img_paths:
            mp = Path(lab_dir) / f"{ip.stem}{mask_suffix}"
            if mp.exists():
                pairs.append((str(ip), str(mp)))
        return pairs
    train_pairs = pair_in(*tr)
    val_pairs = pair_in(*va) if (root / "val").exists() else []
    return train_pairs, val_pairs

def discover_pairs_and_splits(data_root: str, mask_suffix: str = "_multi.png",
                              val_ratio: float = 0.1, split_only: bool = False):
    """
    data_root 直下が YOLOライク（train/images, train/labels, val/...）ならそれを採用。
    そうでなければ、全体を再帰探索してペア集合を作り、val が存在しない場合は val_ratio で分割する。
    split_only=True の場合、既存の構成を尊重（val が見つからねば train 全体のみ返す）。
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    if _has_yolo_layout(root):
        train_pairs, val_pairs = _pair_yolo_like(root, mask_suffix)
        if not train_pairs:
            raise RuntimeError(f"No train pairs found under YOLO-like layout: {root}")
        if not val_pairs and not split_only:
            # val が無ければ分割
            n = len(train_pairs)
            k = max(1, int(n * val_ratio))
            val_pairs = train_pairs[:k]
            train_pairs = train_pairs[k:]
            print(f"[Info] YOLO-like but no val -> split: train={len(train_pairs)} val={len(val_pairs)}")
        else:
            print(f"[Info] YOLO-like discovered: train={len(train_pairs)} val={len(val_pairs)}")
        return train_pairs, val_pairs

    # 汎用フラット/任意構成
    all_pairs = pair_image_mask_lists_generic(root, mask_suffix=mask_suffix)
    if not all_pairs:
        raise RuntimeError(f"No (image, mask) pairs found in: {root} with suffix={mask_suffix}")

    # train/val サブフォルダがあるかを簡易判定
    train_hint = [p for p in (root / "train").rglob("*") if p.is_file()] if (root / "train").exists() else []
    val_hint   = [p for p in (root / "val").rglob("*")   if p.is_file()] if (root / "val").exists() else []

    if train_hint or val_hint:
        # サブフォルダヒントに基づく割当（各ペアの画像パスが train/ または val/ を含むか）
        train_pairs = [(ip, mp) for ip, mp in all_pairs if "/train/" in Path(ip).as_posix()]
        val_pairs   = [(ip, mp) for ip, mp in all_pairs if "/val/"   in Path(ip).as_posix()]
        if not val_pairs and not split_only:
            # val 無ければ分割
            rest = [(ip, mp) for ip, mp in all_pairs if "/train/" in Path(ip).as_posix() or "/val/" in Path(ip).as_posix()]
            base = train_pairs if train_pairs else rest
            if not base:
                base = all_pairs
            n = len(base)
            k = max(1, int(n * val_ratio))
            val_pairs = base[:k]
            train_pairs = base[k:]
            print(f"[Info] flat train/val hint -> split: train={len(train_pairs)} val={len(val_pairs)}")
        else:
            print(f"[Info] flat hint discovered: train={len(train_pairs)} val={len(val_pairs)}")
        return train_pairs, val_pairs

    # 完全フラット（train/val すら無い）: 分割 or split_only
    if split_only:
        # 既存分割は無いので train 全体のみ返す
        print(f"[Info] split_only: returning all as train, none as val (flat dataset)")
        return all_pairs, []
    n = len(all_pairs)
    k = max(1, int(n * val_ratio))
    val_pairs = all_pairs[:k]
    train_pairs = all_pairs[k:]
    print(f"[Info] flat split -> train={len(train_pairs)} val={len(val_pairs)}  (total={n})")
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
            # 幾何を壊しにくい簡易Aug
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
    - ここを既存バックボーンや別モデルに差し替えてもOK
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

    logits = layers.Conv2D(num_classes, 1, padding="same", name="logits")(u4)  # from_logits=True
    model = keras.Model(inputs, logits, name="UNetLite")
    return model

# ========== メトリクス ==========
class MeanIoU(tf.keras.metrics.Metric):
    """
    Keras MeanIoU を logits + sparse label で扱うラッパ
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
