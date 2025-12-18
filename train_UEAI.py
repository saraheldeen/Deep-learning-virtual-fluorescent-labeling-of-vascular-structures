#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    h5_file: str = "train_20x.h5"
    seed: int = 42
    val_frac: float = 0.10

    batch_train: int = 2
    batch_val: int = 4
    epochs: int = 300
    patience: int = 5

    depth: int = 10
    H: int = 512
    W: int = 512
    base_filters: int = 8

    lr: float = 1e-3
    weight_decay: float = 1e-5

    out_dir: str = "runs/lectin"
    model_name: str = "lectin_unet3d"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Model
# -----------------------------
def get_group_count(channels: int) -> int:
    # choose <= channels//4 but ensure divisibility
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


def up_block(x, skip, filters: int):
    x = layers.Conv3DTranspose(filters, 3, strides=(1, 2, 2), padding="same")(x)
    x = tfa.layers.GroupNormalization(groups=get_group_count(filters), axis=-1)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip])
    x = conv3d_block(x, filters)
    return x


def simple_unet3d(depth: int, H: int, W: int, input_channels: int = 1, output_channels: int = 1, base_filters: int = 8) -> Model:
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

    # Decoder (fixed: GN+ReLU+concat included)
    x = up_block(b,  c4, base_filters * 8)
    x = up_block(x,  c3, base_filters * 4)
    x = up_block(x,  c2, base_filters * 2)
    x = up_block(x,  c1, base_filters)

    x = layers.Dropout(0.1)(x)
    out = layers.Conv3D(output_channels, 3, padding="same", activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=out)


# -----------------------------
# Loss + metric (your same idea)
# -----------------------------
def _gaussian_kernel_2d(size: int, sigma: float):
    coords = tf.range(size, dtype=tf.float32) - (size - 1) / 2.0
    g = tf.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    g2d = tf.tensordot(g, g, axes=0)
    return g2d / tf.reduce_sum(g2d)


def _to_4d(x):
    x = tf.cast(x, tf.float32)
    if x.shape.rank == 5:  # [B,Z,H,W,C]
        s = tf.shape(x)
        return tf.reshape(x, [s[0] * s[1], s[2], s[3], s[4]])
    if x.shape.rank == 4:
        return x
    raise ValueError("Expected rank 4 or 5 tensor.")


def _custom_ssim_2d(y_true_4d, y_pred_4d, max_val: float, filter_size: int, sigma: float):
    g2d = _gaussian_kernel_2d(filter_size, sigma)
    g2d = g2d[:, :, tf.newaxis, tf.newaxis]          # [F,F,1,1]
    ch = tf.shape(y_true_4d)[-1]
    kernel = tf.tile(g2d, [1, 1, ch, 1])             # [F,F,C,1]

    mu1 = tf.nn.depthwise_conv2d(y_true_4d, kernel, [1, 1, 1, 1], "SAME")
    mu2 = tf.nn.depthwise_conv2d(y_pred_4d, kernel, [1, 1, 1, 1], "SAME")
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    s1 = tf.nn.depthwise_conv2d(y_true_4d * y_true_4d, kernel, [1, 1, 1, 1], "SAME") - mu1_sq
    s2 = tf.nn.depthwise_conv2d(y_pred_4d * y_pred_4d, kernel, [1, 1, 1, 1], "SAME") - mu2_sq
    s12 = tf.nn.depthwise_conv2d(y_true_4d * y_pred_4d, kernel, [1, 1, 1, 1], "SAME") - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return tf.reduce_mean(ssim)


def CustomSSIMLoss(filter_size: int, sigma: float | None = None, max_val: float = 1.0):
    if sigma is None:
        sigma = float(filter_size) / 6.0

    def loss(y_true, y_pred):
        yt = _to_4d(y_true)
        yp = _to_4d(y_pred)
        return 1.0 - _custom_ssim_2d(yt, yp, max_val, filter_size, float(sigma))

    return loss


def gaussian_blur(x, kernel_size: int = 3, sigma: float = 1.0):
    coords = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1.0) / 2.0
    g = tf.exp(-(coords**2) / (2 * sigma**2))
    g = g / tf.reduce_sum(g)
    g2d = tf.tensordot(g, g, axes=0)
    g2d = g2d[:, :, tf.newaxis, tf.newaxis]
    ch = tf.shape(x)[-1]
    kernel = tf.tile(g2d, [1, 1, ch, 1])
    return tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], "SAME")


def edge_soft_alignment(y_true, y_pred, eps: float = 1e-12):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    n, z, h, w, c = tf.unstack(tf.shape(y_true))

    def _prep(x):
        x = tf.transpose(x, [0, 1, 4, 2, 3])     # (N,Z,C,H,W)
        return tf.reshape(x, [n * z * c, h, w, 1])

    yt = _prep(y_true)
    yp = _prep(y_pred)

    Et = tf.image.sobel_edges(yt)
    Ep = tf.image.sobel_edges(yp)

    mt = tf.reduce_sum(tf.square(Et), axis=-1) + eps
    mp = tf.reduce_sum(tf.square(Ep), axis=-1) + eps

    mt = gaussian_blur(mt, 3, 1.0)
    mp = gaussian_blur(mp, 3, 1.0)
    return tf.reduce_mean(tf.square(mt - mp))


def pearson_correlation_loss(y_true, y_pred, eps: float = 1e-12):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mt = tf.reduce_mean(y_true, axis=[1, 2, 3, 4], keepdims=True)
    mp = tf.reduce_mean(y_pred, axis=[1, 2, 3, 4], keepdims=True)

    yt = y_true - mt
    yp = y_pred - mp

    num = tf.reduce_sum(yt * yp, axis=[1, 2, 3, 4])
    den = tf.sqrt(tf.reduce_sum(tf.square(yt), axis=[1, 2, 3, 4]) * tf.reduce_sum(tf.square(yp), axis=[1, 2, 3, 4]) + eps)
    corr = num / (den + eps)
    return tf.reduce_mean(1.0 - corr)


def lectin_loss(y_true, y_pred):
    ssim_loss = CustomSSIMLoss(filter_size=11, sigma=None, max_val=1.0)
    return (1.0 / 3.0) * edge_soft_alignment(y_true, y_pred) + (1.0 / 3.0) * ssim_loss(y_true, y_pred) + (1.0 / 3.0) * pearson_correlation_loss(y_true, y_pred)


def signal_correlation_metric(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    mt = tf.reduce_mean(y_true_flat)
    mp = tf.reduce_mean(y_pred_flat)
    num = tf.reduce_sum((y_true_flat - mt) * (y_pred_flat - mp))
    den = tf.sqrt(tf.reduce_sum(tf.square(y_true_flat - mt)) * tf.reduce_sum(tf.square(y_pred_flat - mp))) + 1e-6
    return num / den


# -----------------------------
# Generator (Lectin)
# -----------------------------
class LectinTransmissionGenerator(Sequence):
    def __init__(self, h5_path: str, indices: np.ndarray, batch_size: int = 4, shuffle: bool = True):
        self.h5_path = h5_path
        self.indices = np.array(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        with h5py.File(self.h5_path, "r") as f:
            x_p = f["x_patches"]
            y_p = f["y_patches"]
            Z, H, W = x_p.shape[1:4]

            x = np.zeros((len(batch_idx), Z, H, W, 1), dtype=np.float32)
            y = np.zeros((len(batch_idx), Z, H, W, 1), dtype=np.float32)

            for i, j in enumerate(batch_idx):
                x[i] = x_p[j][..., 0:1]   # NOTE: channel meaning depends on how H5 was built
                y[i] = y_p[j][..., 2:3]   # Lectin (3rd label channel)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# -----------------------------
# Train
# -----------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cfg.h5_file, "r") as f:
        total = f["x_patches"].shape[0]

    idx_all = np.arange(total)
    idx_tr, idx_va = train_test_split(idx_all, test_size=cfg.val_frac, random_state=cfg.seed)

    train_gen = LectinTransmissionGenerator(cfg.h5_file, idx_tr, batch_size=cfg.batch_train, shuffle=True)
    val_gen = LectinTransmissionGenerator(cfg.h5_file, idx_va, batch_size=cfg.batch_val, shuffle=False)

    model = simple_unet3d(cfg.depth, cfg.H, cfg.W, input_channels=1, output_channels=1, base_filters=cfg.base_filters)
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
        loss=lectin_loss,
        metrics=[signal_correlation_metric],
    )

    best_path = out_dir / f"{cfg.model_name}.keras"
    final_path = out_dir / f"{cfg.model_name}_final.keras"
    log_path = out_dir / f"{cfg.model_name}.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True, mode="min", verbose=1),
        tf.keras.callbacks.CSVLogger(str(log_path), append=True),
    ]
    history = model.fit(train

