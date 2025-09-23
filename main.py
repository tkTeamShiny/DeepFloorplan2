# -*- coding: utf-8 -*-
"""
DeepFloorplan2 - main.py
- 既存の net.py に依存して学習/推論を実行
- マスクは <ID>_multi.png のみを使用（索引マスク or RGB→索引に自動変換）
- データ構成: data_root/{train,val}/{images,labels}
"""

import os
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

# ここで既存(更新後)の net.py に依存
from net import (
    set_global_seed,
    discover_splits,
    make_tf_dataset,
    build_model,
    MeanIoU,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="Train", choices=["Train", "Test"],
                   help="Train または Test を指定")
    p.add_argument("--data_root", type=str, default="yolo",
                   help="data_root/{train,val}/{images,labels} を想定")
    p.add_argument("--img_size", nargs=2, type=int, default=[512, 512],
                   help="入力解像度 [H W]")
    p.add_argument("--num_classes", type=int, default=10,
                   help="クラス数（*_multi.png のラベル最大+1）")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)

    # Test 用
    p.add_argument("--weights", type=str, default="runs/checkpoints/best.keras",
                   help="学習済み重み（.keras or .h5）")
    p.add_argument("--test_split", type=str, default="val", choices=["train", "val"],
                   help="推論対象の分割")
    p.add_argument("--save_pred", type=str, default="runs/preds",
                   help="推論結果の保存先（PNGカラー可視化）")
    return p.parse_args()

def train(args):
    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = Path(args.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.keras"

    # データ検出（val が無ければ自動で 9:1 分割）
    train_pairs, val_pairs = discover_splits(args.data_root)

    # tf.data の用意
    H, W = args.img_size
    train_ds = make_tf_dataset(
        pairs=train_pairs,
        img_size=(H, W),
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        shuffle=True,
        aug=True,
    )
    val_ds = make_tf_dataset(
        pairs=val_pairs,
        img_size=(H, W),
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        shuffle=False,
        aug=False,
    )

    # モデル構築
    model = build_model(input_shape=(H, W, 3), num_classes=args.num_classes)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [MeanIoU(args.num_classes)]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mean_iou", mode="max", patience=8, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(str(Path(args.out_dir) / "train_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    print("[Done] Training finished.")
    print(f"[Best] checkpoint saved at: {ckpt_path}")

def test(args):
    from net import pair_image_mask_lists, read_image, colorize_index_mask
    import numpy as np
    from imageio.v2 import imwrite

    H, W = args.img_size
    split = args.test_split
    img_dir = Path(args.data_root) / split / "images"
    lab_dir = Path(args.data_root) / split / "labels"
    pairs = pair_image_mask_lists(img_dir, lab_dir)
    if not pairs:
        raise RuntimeError(f"No pairs found under: {img_dir} / {lab_dir}")

    print(f"[Info] loading weights: {args.weights}")
    model = keras.models.load_model(args.weights, custom_objects={"MeanIoU": MeanIoU})
    out_dir = Path(args.save_pred); out_dir.mkdir(parents=True, exist_ok=True)

    for ip, _ in pairs:
        img = read_image(ip, target_size=(H, W))  # float32 0-1, RGB
        logits = model.predict(img[None, ...], verbose=0)[0]
        pred = np.argmax(logits, axis=-1).astype(np.uint8)
        rgb = colorize_index_mask(pred, args.num_classes)

        stem = Path(ip).stem
        imwrite(str(out_dir / f"{stem}_pred.png"), rgb)
    print(f"[Done] saved predictions to: {out_dir}")

def main():
    args = parse_args()
    print("========== Config ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    if args.phase.lower() == "train":
        train(args)
    else:
        test(args)

if __name__ == "__main__":
    main()
