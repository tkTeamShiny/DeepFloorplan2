# -*- coding: utf-8 -*-
"""
DeepFloorplan2 - main.py
- 既存(更新版) net.py に依存
- フラット配置 or 任意構成のデータに対応（ZIP を渡せば自動展開＆再帰探索）
- マスクは <ID>_multi.png（索引 or RGB→索引）を使用
"""

import os
import argparse
from pathlib import Path
import shutil
import zipfile
import tempfile

import tensorflow as tf
from tensorflow import keras

# 依存（この会話で提示の更新版 net.py）
from net import (
    set_global_seed,
    discover_pairs_and_splits,
    make_tf_dataset,
    build_model,
    MeanIoU,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="Train", choices=["Train", "Test"],
                   help="Train または Test")
    # データ指定（どちらか一方、または両方）
    p.add_argument("--data_zip", type=str, default="",
                   help="データZIPへのパス（例: /mnt/data/YOLO_SetA_seg_no1_部屋種別改変版_flat.zip）")
    p.add_argument("--data_root", type=str, default="",
                   help="展開済みデータのルート（空なら ZIP を展開して自動決定）")

    # 走査設定
    p.add_argument("--mask_suffix", type=str, default="_multi.png",
                   help="マスクのファイル名サフィックス（既定: _multi.png）")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="val が見つからない場合の分割比（既定: 0.1 = 9:1）")

    # 学習設定
    p.add_argument("--img_size", nargs=2, type=int, default=[512, 512],
                   help="入力サイズ [H W]")
    p.add_argument("--num_classes", type=int, default=23,
                   help="クラス数（既定: 23）")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)

    # 推論設定
    p.add_argument("--weights", type=str, default="runs/checkpoints/best.keras",
                   help="学習済み重み（.keras or .h5）")
    p.add_argument("--test_split", type=str, default="auto",
                   choices=["auto", "train", "val"],
                   help="推論対象分割。auto=val優先、無ければtrain")
    p.add_argument("--save_pred", type=str, default="runs/preds",
                   help="推論結果の保存先（PNGカラー可視化）")
    return p.parse_args()

def _ensure_data_root(args):
    """
    data_root が空で data_zip が与えられたら ZIP を展開し、展開先を返す。
    data_root が指定されていればそのまま返す。
    """
    if args.data_root:
        root = Path(args.data_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"data_root not found: {root}")
        return str(root)

    if not args.data_zip:
        raise ValueError("データが指定されていません。--data_zip か --data_root のどちらかを指定してください。")

    zip_path = Path(args.data_zip).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    out_base = Path(args.out_dir) / "data_extracted"
    if out_base.exists():
        # 既存展開物を使い回し（再現性のため残す）
        return str(out_base.resolve())

    out_base.mkdir(parents=True, exist_ok=True)
    print(f"[Info] extracting ZIP to: {out_base}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(out_base))
    # 展開直下に一階層フォルダがある場合はそこをルートと見なす
    candidates = [p for p in out_base.iterdir() if p.is_dir()]
    root = candidates[0] if len(candidates) == 1 else out_base
    print(f"[Info] data root resolved to: {root}")
    return str(root.resolve())

def train(args):
    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = Path(args.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.keras"

    data_root = _ensure_data_root(args)

    # データ検出（val が無ければ自動分割）
    train_pairs, val_pairs = discover_pairs_and_splits(
        data_root=data_root,
        mask_suffix=args.mask_suffix,
        val_ratio=args.val_ratio,
    )

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
    from net import colorize_index_mask, read_image
    import numpy as np
    from imageio.v2 import imwrite

    data_root = _ensure_data_root(args)
    # 分割の自動解決: val優先、無ければtrain
    if args.test_split == "auto":
        pref = ["val", "train"]
    else:
        pref = [args.test_split]

    # 既存の探索関数を使ってペア集合を取得
    all_train, all_val = discover_pairs_and_splits(
        data_root=data_root,
        mask_suffix=args.mask_suffix,
        val_ratio=args.val_ratio,
        split_only=True,  # 既存フォルダ構成を保持（無いときだけ分割情報が返る）
    )

    split_pairs = None
    split_name = None
    for name in pref:
        if name == "val" and all_val:
            split_pairs = all_val; split_name = "val"; break
        if name == "train" and all_train:
            split_pairs = all_train; split_name = "train"; break
    if not split_pairs:
        raise RuntimeError("テストに使える split が見つかりません。")

    H, W = args.img_size
    print(f"[Info] loading weights: {args.weights}")
    model = keras.models.load_model(args.weights, custom_objects={"MeanIoU": MeanIoU})
    out_dir = Path(args.save_pred); out_dir.mkdir(parents=True, exist_ok=True)

    for ip, _ in split_pairs:
        img = read_image(ip, (H, W))  # float32 0-1, RGB
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
