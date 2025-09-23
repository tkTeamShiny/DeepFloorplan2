# -*- coding: utf-8 -*-
"""
TF2/Keras3 版 DeepFloorplan 互換ネットワーク
- zlzeng/DeepFloorplan の net.py と同等の役割・命名に寄せたクラス設計
- 2ヘッド出力:
    * logits_cw:   壁+開口（close_wall）などの境界系 (C_cw=2 を想定: 非境界/境界)
    * logits_r:    部屋タイプ (C_r=num_room_classes)
- build()/compile() 相当の役割は Network.build() で実施
- forward() 相当は call()
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, Dict

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model


def _conv_bn_relu(x, filters, k=3, s=1, d=1):
    x = L.Conv2D(filters, k, strides=s, padding="same", dilation_rate=d, use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x)

def _res_block(x, filters, k=3, d=1):
    h = _conv_bn_relu(x, filters, k, 1, d)
    h = L.Conv2D(filters, k, padding="same", dilation_rate=d, use_bias=False)(h)
    h = L.BatchNormalization()(h)
    if x.shape[-1] != filters:
        x = L.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
    h = L.Add()([h, x])
    return L.ReLU()(h)

def _down(x, filters):
    x = L.MaxPool2D(2)(x)
    return _res_block(x, filters)

def _up(x, skip, filters):
    x = L.UpSampling2D(size=2, interpolation="bilinear")(x)
    x = L.Concatenate()([x, skip])
    return _res_block(x, filters)


class Network(Model):
    """
    Keras Model として実装。zlzeng/DeepFloorplan のインターフェイスに寄せたシンプルラッパ。
    Attributes:
        num_room_classes: 部屋クラス数（r ヘッド）
        num_cw_classes:   近接壁/境界クラス数（cw ヘッド。既定=2）
    Methods:
        build(inputs_shape): 入力形状でモデルグラフを構築
        call(x, training=False): 前向き
        losses_dict(y_true_cw, y_true_r, y_pred_cw, y_pred_r): タスク別 loss を計算
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

        # Encoder
        self.stem1 = L.Conv2D(base_channels, 7, strides=2, padding="same", use_bias=False)
        self.stem1_bn = L.BatchNormalization()
        self.stem1_act = L.ReLU()

        self.enc1 = _res_block  # 1/2
        self.enc2 = _down       # 1/4
        self.enc3 = _down       # 1/8
        self.enc4 = _down       # 1/16

        # Bottleneck with RBGA っぽい拡張畳み込み（簡易版）
        self.bottleneck1 = lambda x: _res_block(x, base_channels * 8, d=2)
        self.bottleneck2 = lambda x: _res_block(x, base_channels * 8, d=4)

        # Decoder (UNet)
        self.dec3_conv = None  # 後で dynamic build
        self.dec2_conv = None
        self.dec1_conv = None
        self.dec0_conv = None

        # Heads
        self.head_cw = None
        self.head_r = None

        # 事前構築されるまでのプレースホルダ
        self._built = False
        self.input_spec = None

    def build(self, input_shape):
        """Keras の build: 入力 shape を受け取り，層を確定"""
        _, H, W, C = input_shape
        ch = 64

        # ダミーで一度 forward してスキップ接続サイズを確定
        xin = L.Input(shape=(H, W, C))
        x0 = self.stem1_act(self.stem1_bn(self.stem1(xin)))  # 1/2
        s1 = self.enc1(x0, ch)                               # 1/2
        s2 = self.enc2(s1, ch * 2)                           # 1/4
        s3 = self.enc3(s2, ch * 4)                           # 1/8
        s4 = self.enc4(s3, ch * 8)                           # 1/16

        b = self.bottleneck1(s4)
        b = self.bottleneck2(b)

        d3 = _up(b, s3, ch * 4)  # 1/8
        d2 = _up(d3, s2, ch * 2) # 1/4
        d1 = _up(d2, s1, ch)     # 1/2
        d0 = L.UpSampling2D(size=2, interpolation="bilinear")(d1)  # 1/1

        # 軽い整形 conv
        self.dec3_conv = L.Conv2D(ch * 4, 3, padding="same", activation="relu")
        self.dec2_conv = L.Conv2D(ch * 2, 3, padding="same", activation="relu")
        self.dec1_conv = L.Conv2D(ch, 3, padding="same", activation="relu")
        self.dec0_conv = L.Conv2D(ch, 3, padding="same", activation="relu")

        # heads
        self.head_cw = L.Conv2D(self.num_cw_classes, 1, name="logits_cw")   # [B,H,W,C_cw]
        self.head_r  = L.Conv2D(self.num_room_classes, 1, name="logits_r")  # [B,H,W,C_r]

        # mark built
        super().build(input_shape)
        self._built = True

    def call(self, x, training=False):
        # Encoder
        x0 = self.stem1_act(self.stem1_bn(self.stem1(x), training=training), training=training)
        s1 = self.enc1(x0, 64)
        s2 = self.enc2(s1, 128)
        s3 = self.enc3(s2, 256)
        s4 = self.enc4(s3, 512)

        # Bottleneck
        b = self.bottleneck1(s4)
        b = self.bottleneck2(b)

        # Decoder
        d3 = _up(b, s3, 256)
        d3 = self.dec3_conv(d3)
        d2 = _up(d3, s2, 128)
        d2 = self.dec2_conv(d2)
        d1 = _up(d2, s1, 64)
        d1 = self.dec1_conv(d1)
        d0 = L.UpSampling2D(size=2, interpolation="bilinear")(d1)
        d0 = self.dec0_conv(d0)

        # Heads (logits)
        logits_cw = self.head_cw(d0)
        logits_r  = self.head_r(d0)
        return logits_cw, logits_r

    @staticmethod
    def losses_dict(y_true_cw, y_true_r, y_pred_cw, y_pred_r,
                    cw_weight: float = 1.0, r_weight: float = 1.0) -> Dict[str, tf.Tensor]:
        """
        タスク別 softmax CE の合計（Cross-and-within の重み付けの簡易版）
        y_* 形状: [B,H,W]（int32 ラベル） / y_pred_*: [B,H,W,C]
        """
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
    """Factory"""
    net = Network(num_room_classes=num_room_classes, num_cw_classes=num_cw_classes)
    dummy = tf.zeros([1, img_size[0], img_size[1], 3], dtype=tf.float32)
    net.build(dummy.shape)
    return net
