# -*- coding: utf-8 -*-
"""
TF2/Keras3 版 DeepFloorplan 互換ネットワーク（固定レイヤー化）
- すべてのレイヤーを __init__/build で作成し、call() では作成しない
- 2ヘッド出力:
    * logits_cw:   壁/境界 (C_cw=2 を想定)
    * logits_r:    部屋タイプ (C_r=num_room_classes)
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model


class ResBlock(L.Layer):
    def __init__(self, filters: int, k: int = 3, d: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.filters = filters
        self.k = k
        self.d = d

        # convs
        self.conv1 = L.Conv2D(filters, k, padding="same", dilation_rate=d, use_bias=False)
        self.bn1 = L.BatchNormalization()
        self.act1 = L.ReLU()

        self.conv2 = L.Conv2D(filters, k, padding="same", dilation_rate=d, use_bias=False)
        self.bn2 = L.BatchNormalization()

        # projection (build で入力チャンネルに応じて有効化)
        self.proj_conv = None
        self.proj_bn = None
        self.act_out = L.ReLU()

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if in_ch != self.filters:
            self.proj_conv = L.Conv2D(self.filters, 1, padding="same", use_bias=False)
            self.proj_bn = L.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.bn2(h, training=training)

        shortcut = x
        if self.proj_conv is not None:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut, training=training)

        out = L.Add()([h, shortcut])
        out = self.act_out(out)
        return out


class DownBlock(L.Layer):
    """ MaxPool2D -> ResBlock """
    def __init__(self, filters: int, k: int = 3, d: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.pool = L.MaxPool2D(2)
        self.res = ResBlock(filters, k=k, d=d)

    def call(self, x, training=False):
        x = self.pool(x)
        x = self.res(x, training=training)
        return x


class UpBlock(L.Layer):
    """ UpSampling2D -> Concat(skip) -> ResBlock """
    def __init__(self, filters: int, k: int = 3, d: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.up = L.UpSampling2D(size=2, interpolation="bilinear")
        self.concat = L.Concatenate()
        self.res = ResBlock(filters, k=k, d=d)

    def call(self, x, skip, training=False):
        x = self.up(x)
        x = self.concat([x, skip])
        x = self.res(x, training=training)
        return x


class Network(Model):
    """
    Keras Model 実装（動的レイヤー生成なし）
    Attributes:
        num_room_classes: 部屋クラス数（r ヘッド）
        num_cw_classes:   境界クラス数（cw ヘッド）
    """
    def __init__(self,
                 num_room_classes: int,
                 num_cw_classes: int = 2,
                 base_channels: int = 64,
                 name: str = "DeepFloorplanMTL",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_room_classes = num_room_classes
        self.num_cw_classes = num_cw_classes
        ch = base_channels

        # Encoder stem
        self.stem_conv = L.Conv2D(ch, 7, strides=2, padding="same", use_bias=False)
        self.stem_bn = L.BatchNormalization()
        self.stem_act = L.ReLU()

        # Encoder levels
        self.enc1 = ResBlock(ch, name="enc1")        # 1/2
        self.enc2 = DownBlock(ch * 2, name="enc2")   # 1/4
        self.enc3 = DownBlock(ch * 4, name="enc3")   # 1/8
        self.enc4 = DownBlock(ch * 8, name="enc4")   # 1/16

        # Bottleneck (dilated)
        self.bottleneck1 = ResBlock(ch * 8, d=2, name="bneck1")
        self.bottleneck2 = ResBlock(ch * 8, d=4, name="bneck2")

        # Decoder
        self.dec3 = UpBlock(ch * 4, name="dec3")     # 1/8
        self.dec3_conv = L.Conv2D(ch * 4, 3, padding="same", activation="relu")

        self.dec2 = UpBlock(ch * 2, name="dec2")     # 1/4
        self.dec2_conv = L.Conv2D(ch * 2, 3, padding="same", activation="relu")

        self.dec1 = UpBlock(ch, name="dec1")         # 1/2
        self.dec1_conv = L.Conv2D(ch, 3, padding="same", activation="relu")

        self.up0 = L.UpSampling2D(size=2, interpolation="bilinear")  # 1/1
        self.dec0_conv = L.Conv2D(ch, 3, padding="same", activation="relu")

        # Heads
        self.head_cw = L.Conv2D(self.num_cw_classes, 1, name="logits_cw")
        self.head_r  = L.Conv2D(self.num_room_classes, 1, name="logits_r")

    def call(self, x, training=False):
        # Encoder
        x0 = self.stem_conv(x)
        x0 = self.stem_bn(x0, training=training)
        x0 = self.stem_act(x0)

        s1 = self.enc1(x0, training=training)   # 1/2
        s2 = self.enc2(s1, training=training)   # 1/4
        s3 = self.enc3(s2, training=training)   # 1/8
        s4 = self.enc4(s3, training=training)   # 1/16

        # Bottleneck
        b = self.bottleneck1(s4, training=training)
        b = self.bottleneck2(b, training=training)

        # Decoder
        d3 = self.dec3(b, s3, training=training)
        d3 = self.dec3_conv(d3)

        d2 = self.dec2(d3, s2, training=training)
        d2 = self.dec2_conv(d2)

        d1 = self.dec1(d2, s1, training=training)
        d1 = self.dec1_conv(d1)

        d0 = self.up0(d1)
        d0 = self.dec0_conv(d0)

        # Heads (logits)
        logits_cw = self.head_cw(d0)
        logits_r  = self.head_r(d0)
        return logits_cw, logits_r

    @staticmethod
    def losses_dict(y_true_cw, y_true_r, y_pred_cw, y_pred_r,
                    cw_weight: float = 1.0, r_weight: float = 1.0) -> Dict[str, tf.Tensor]:
        y_true_cw = tf.cast(y_true_cw, tf.int32)
        y_true_r  = tf.cast(y_true_r, tf.int32)

        loss_cw = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_cw, y_pred_cw, from_logits=True)
        loss_r  = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_r, y_pred_r, from_logits=True)

        loss_cw = tf.reduce_mean(loss_cw)
        loss_r  = tf.reduce_mean(loss_r)

        return {
            "loss_cw": loss_cw * cw_weight,
            "loss_r":  loss_r  * r_weight,
            "loss_total": loss_cw * cw_weight + loss_r * r_weight,
        }


def build_network(img_size: Tuple[int, int] = (512, 512),
                  num_room_classes: int = 23,
                  num_cw_classes: int = 2) -> Network:
    """Factory: ここではレイヤーは __init__ 時点で作成されるため build は不要"""
    net = Network(num_room_classes=num_room_classes, num_cw_classes=num_cw_classes)
    # Keras に形状を伝えて重み初期化したい場合はダミーを一度通す
    _ = net(tf.zeros([1, img_size[0], img_size[1], 3], dtype=tf.float32), training=False)
    return net
