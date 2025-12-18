#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence as TypingSequence

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow_addons as tfa


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    h5_file: str = "train_20x.h5"
    seed: int = 42
    val_frac: float = 0.10

    # Data
    batch_train: int = 2
    batch_val: int = 4
    shuffle: bool = True

    # Model
    depth: int = 10
    H: int = 512
    W: int = 512
    input_channels: int = 1
    output_channels: int = 1
    base_filters: int = 8

    # Optim / training
    lr: float = 1e-2
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 5

    # Outputs
    out_dir: str = "runs/phalloidin"
    model_name: str = "phalloidin_unet3d"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_group_count(channels: int) -> int:
    g = max(1, channels // 4)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def conv3d_block(x, filters: int):
    x = layers.Conv3D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = tfa.layers.GroupNormalization(groups=get_group_count(filters), axis=-1)(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = tfa.layers.GroupNormalization(groups=get_group_count(filters), axis=-1)(x)
    x = layers.ReLU()(x)
    return x


def simple_unet3d(depth: int, H: int, W: int, input_channels: int, output_channels: int, base_filters: int = 8) -> Model:
    inputs = Input(shape=(depth, H, W, input_channels), name="input")

    # Encoder
    c1 = conv3d_block(inputs, base_filters)
    p1 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c1)

    c2 = conv3d_block(p1, base_filters * 2)
    p2 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c2)

    c3 = conv3d_block(p2, base_filters * 4)
    p3 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c3)

    c4 = conv3d_block(p3, base_filters * 8)
    p4 = layers.MaxPooling3D(pool_size=(1, 2, 2))(c4)

    # Bottleneck
    b = conv3d_block(p4, base_filters * 16)

    # Decoder
    u4 = layers.Conv3DTranspose(base_filters * 8, 3, strides=(1, 2, 2), padding="same")(b)
    u4 = tfa.layers.GroupNormalization(groups=get_group_count(base_filters * 8), axis=-1)(u4)
    u4 = layers.ReLU()(u4)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv3d_block(u4, base_filters * 8)

    u3 = layers.Conv3DTranspose(base_filters * 4, 3, strides=(1, 2, 2), padding="same")(c5)
    u3 = tfa.layers.GroupNormalization(groups=get_group_count(base_filters * 4), axis=-1)(u3)
    u3 = layers.ReLU()(u3)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv3d_block(u3, base_filters * 4)

    u2 = layers.Conv3DTranspose(base_filters * 2, 3, strides=(1, 2, 2), padding="same")(c6)
    u2 = tfa.layers.GroupNormalization(groups=get_group_count(base_filters * 2), axis=-1)(u2)
    u2 = layers.ReLU()(u2)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv3d_block(u2, base_filters * 2)

    u1 = layers.Conv3DTranspose(base_filters, 3, strides=(1, 2, 2), padding="same")(c7)
    u1 = tfa.layers.GroupNormalization(groups=get_group_count(base_filters), axis=-1)(u1)
    u1 = layers.ReLU()(u1)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv3d_block(u1, base_filters)

    x = layers.Dropout(0.1)(c8)
    out = layers.Conv3D(output_channels, 3, padding="same", activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=out)


# -----------------------------
# Loss + metric (kept same logic, cleaned)
# -----------------------------
def gaussian_blur(x, kernel_size: int = 3, sigma: float = 1.0):
    coords = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1.0) / 2.0
    g = tf.exp(-(coords**2) / (2 * sigma**2))
    g = g / tf.reduce_sum(g)
    g2d = tf.tensordot(g, g, axes=0)                       # [k,k]
    g2d = g2d[:, :, tf.newaxis, tf.newaxis]                # [k,k,1,1]
    in_ch = tf.shape(x)[-1]
    kernel = tf.tile(g2d, [1, 1, in_ch, 1])                # [k,k,c,1]
    return tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], "SAME")


def edge_soft_alignment(y_true, y_pred, eps: float = 1e-12):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    n, z, h, w, c = tf.unstack(tf.shape(y_true))

    def _prep(x):
        x = tf.transpose(x, [0, 1, 4, 2, 3])                 # (N,Z,C,H,W)
        x = tf.reshape(x, [n * z * c, h, w, 1])              # (N*Z*C,H,W,1)
        return x

    yt = _prep(y_true)
    yp = _prep(y_pred)

    Et = tf.image.sobel_edges(yt)
    Ep = tf.image.sobel_edges(yp)

    mag_t = tf.reduce_sum(tf.square(Et), axis=-1) + eps
    mag_p = tf.reduce_sum(tf.square(Ep), axis=-1) + eps

    mag_t = gaussian_blur(mag_t, kernel_size=3, sigma=1.0)
    mag_p = gaussian_blur(mag_p, kernel_size=3, sigma=1.0)

    return tf.reduce_mean(tf.square(mag_t - mag_p))


def _gaussian_kernel_2d(size: int, sigma: float):
    coords = tf.range(size, dtype=tf.float32) - (size - 1) / 2.0
    g = tf.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    gauss2d = tf.tensordot(g, g, axes=0)
    return gauss2d / tf.reduce_sum(gauss2d)


def _to_4d(x):
    x = tf.cast(x, tf.float32)
    if x.shape.rank == 5:  # [B,Z,H,W,C] -> fold Z into batch
        s = tf.shape(x)
        return tf.reshape(x, [s[0] * s[1], s[2], s[3], s[4]])
    if x.shape.rank == 4:
        return x
    raise ValueError("Expected rank 4 or 5 tensor.")


def _custom_ssim_2d(y_true_4d, y_pred_4d, max_val: float, filter_size: int, sigma: float):
    gauss2d = _gaussian_kernel_2d(filter_size, sigma)
    gauss2d = gauss2d[:, :, tf.newaxis, tf.newaxis]         # [F,F,1,1]
    in_ch = tf.shape(y_true_4d)[-1]
    kernel = tf.tile(gauss2d, [1, 1, in_ch, 1])             # [F,F,C,1]

    mu1 = tf.nn.depthwise_conv2d(y_true_4d, kernel, [1, 1, 1, 1], "SAME")
    mu2 = tf.nn.depthwise_conv2d(y_pred_4d, kernel, [1, 1, 1, 1], "SAME")
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = tf.nn.depthwise_conv2d(y_true_4d * y_true_4d, kernel, [1, 1, 1, 1], "SAME") - mu1_sq
    sigma2_sq = tf.nn.depthwise_conv2d(y_pred_4d * y_pred_4d, kernel, [1, 1, 1, 1], "SAME") - mu2_sq
    sigma12 = tf.nn.depthwise_conv2d(y_true_4d * y_pred_4d, kernel, [1, 1, 1, 1], "SAME") - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return tf.reduce_mean(ssim_map)


def CustomSSIMLoss(filter_size: int, sigma: float | None = None, max_val: float = 1.0):
    if sigma is None:
        sigma = float(filter_size) / 6.0

    def loss(y_true, y_pred):
        yt = _to_4d(y_true)
        yp = _to_4d(y_pred)
        ssim_mean = _custom_ssim_2d(yt, yp, max_val, filter_size, float(sigma))
        return 1.0 - ssim_mean

    return loss


def pearson_correlation_loss(y_true, y_pred, eps: float = 1e-12):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_mean = tf.reduce_mean(y_true, axis=[1, 2, 3, 4], keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=[1, 2, 3, 4], keepdims=True)

    yt = y_true - y_true_mean
    yp = y_pred - y_pred_mean

    num = tf.reduce_sum(yt * yp, axis=[1, 2, 3, 4])
    den = tf.sqrt(tf.reduce_sum(tf.square(yt), axis=[1, 2, 3, 4]) * tf.reduce_sum(tf.square(yp), axis=[1, 2, 3, 4]) + eps)
    corr = num / (den + eps)
    return tf.reduce_mean(1.0 - corr)


def sharp_fiber_loss(y_true, y_pred):
    ssim_loss = CustomSSIMLoss(filter_size=11, sigma=None, max_val=1.0)
    return (
        0.45 * edge_soft_alignment(y_true, y_pred)
        + 0.10 * ssim_loss(y_true, y_pred)
        + 0.45 * pearson_correlation_loss(y_true, y_pred)
    )


def edge_metric(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 1])

    edges_true = tf.image.sobel_edges(y_true_flat)
    edges_pred = tf.image.sobel_edges(y_pred_flat)

    mag_true = tf.sqrt(tf.reduce_sum(tf.square(edges_true), axis=-1) + 1e-12)
    mag_pred = tf.sqrt(tf.reduce_sum(tf.square(edges_pred), axis=-1) + 1e-12)

    num = tf.reduce_sum(mag_true * mag_pred)
    den = tf.reduce_sum(mag_true**2) + tf.reduce_sum(mag_pred**2) + 1e-12
    score = (2.0 * num) / den
    return tf.where(tf.math.is_finite(score), score, 0.0)


# -----------------------------
# Data generator
# -----------------------------
class PhalloidinTransmissionGenerator(Sequence):
    """
    Expects:
      x_patches: (N, Z, H, W, 2)  -> we use channel 0 (Transmission per your comment; check!)
      y_patches: (N, Z, H, W, 3)  -> we use channel 1 (Phalloidin)
    """
    def __init__(self, h5_path: str, indices: np.ndarray, batch_size: int = 4, shuffle: bool = True):
        self.h5_path = h5_path
        self.indices = np.array(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        with h5py.File(self.h5_path, "r") as f:
            x_p = f["x_patches"]
            y_p = f["y_patches"]
            Z, H, W = x_p.shape[1:4]

            x = np.zeros((len(batch_idx), Z, H, W, 1), dtype=np.float32)
            y = np.zeros((len(batch_idx), Z, H, W, 1), dtype=np.float32)

            for i, j in enumerate(batch_idx):
                x[i] = x_p[j][..., 0:1]  # NOTE: your comment says Transmission; confirm channel meaning in your H5
                y[i] = y_p[j][..., 1:2]  # Phalloidin

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# -----------------------------
# Train entrypoint
# -----------------------------
def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine N
    with h5py.File(cfg.h5_file, "r") as f:
        total_patches = f["x_patches"].shape[0]

    all_idx = np.arange(total_patches)
    train_idx, val_idx = train_test_split(all_idx, test_size=cfg.val_frac, random_state=cfg.seed)

    train_gen = PhalloidinTransmissionGenerator(cfg.h5_file, train_idx, batch_size=cfg.batch_train, shuffle=cfg.shuffle)
    val_gen = PhalloidinTransmissionGenerator(cfg.h5_file, val_idx, batch_size=cfg.batch_val, shuffle=False)

    model = simple_unet3d(
        depth=cfg.depth,
        H=cfg.H,
        W=cfg.W,
        input_channels=cfg.input_channels,
        output_channels=cfg.output_channels,
        base_filters=cfg.base_filters,
    )

    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
        loss=sharp_fiber_loss,
        metrics=[edge_metric],
    )

    ckpt_path = out_dir / f"{cfg.model_name}.keras"
    log_path = out_dir / f"{cfg.model_name}_log.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_path), append=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.epochs,
        callbacks=callbacks,
    )

    model.save(str(out_dir / f"{cfg.model_name}_final.keras"))
    return history.history


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)

