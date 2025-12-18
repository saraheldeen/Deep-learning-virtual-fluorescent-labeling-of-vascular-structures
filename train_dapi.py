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

    # batches
    batch_train: int = 2
    batch_val: int = 4

    # model shape
    depth: int = 10
    H: int = 512
    W: int = 512
    base_filters: int = 8

    # training
    lr: float = 1e-2
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 5

    # generator behavior
    min_label_sum: float = 0.01
    label_percentile: float = 99.0
    label_threshold_true: float = 0.10
    metric_threshold: float = 0.15

    # outputs
    out_dir: str = "runs/dapi"
    model_name: str = "dapi_unet3d"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Model
# -----------------------------
def get_group_count(channels: int) -> int:
    # keep ~channels/4 groups but ensure divisibility
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

    # Decoder (no stray text lines; safe)
    u4 = layers.Conv3DTranspose(base_filters * 8, 3, strides=(1, 2, 2), padding="same")(b)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv3d_block(u4, base_filters * 8)

    u3 = layers.Conv3DTranspose(base_filters * 4, 3, strides=(1, 2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv3d_block(u3, base_filters * 4)

    u2 = layers.Conv3DTranspose(base_filters * 2, 3, strides=(1, 2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv3d_block(u2, base_filters * 2)

    u1 = layers.Conv3DTranspose(base_filters, 3, strides=(1, 2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv3d_block(u1, base_filters)

    x = layers.Dropout(0.1)(c8)
    out = layers.Conv3D(output_channels, 3, padding="same", activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=out)


# -----------------------------
# Loss parts (your DAPI composite)
# -----------------------------
def _spatial_axes(y):
    return tf.range(1, tf.rank(y))


def pearson_correlation_loss(y_true, y_pred, eps: float = 1e-8):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = _spatial_axes(y_true)

    mu_t = tf.reduce_mean(y_true, axis=axes, keepdims=True)
    mu_p = tf.reduce_mean(y_pred, axis=axes, keepdims=True)

    t_c = y_true - mu_t
    p_c = y_pred - mu_p

    num = tf.reduce_sum(t_c * p_c, axis=axes)
    den = tf.sqrt(tf.reduce_sum(tf.square(t_c), axis=axes) * tf.reduce_sum(tf.square(p_c), axis=axes)) + eps
    r = num / den
    return 1.0 - tf.reduce_mean(r)


def tversky_loss(y_true, y_pred, alpha: float = 0.9, beta: float = 0.1, eps: float = 1e-6):
    # your thresholding behavior kept
    y_true = tf.cast(y_true > 0.2, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)

    axes = _spatial_axes(y_true)
    tp = tf.reduce_sum(y_true * y_pred, axis=axes)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tf.reduce_mean(tversky)


def Dapi_loss(y_true, y_pred, w_tversky: float = 0.5, w_corr: float = 0.5):
    return w_tversky * tversky_loss(y_true, y_pred) + w_corr * pearson_correlation_loss(y_true, y_pred)


# -----------------------------
# Generator (keeps your skip-empty + percentile scaling)
# -----------------------------
class DapiTransmissionGenerator(Sequence):
    def __init__(self, h5_path: str, indices: np.ndarray, cfg: TrainConfig, batch_size: int = 4, shuffle: bool = True):
        self.h5_path = h5_path
        self.indices = np.array(indices, dtype=np.int64)
        self.cfg = cfg
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

            x_list, y_list = [], []

            for j in batch_idx:
                x = x_p[j][..., 0:1]  # NOTE: this is whatever channel 0 is in your H5
                y = y_p[j][..., 0:1]  # DAPI

                if float(np.sum(y)) < self.cfg.min_label_sum:
                    continue

                pos = y[y > 0]
                if pos.size > 0:
                    p = np.percentile(pos, self.cfg.label_percentile)
                    if p > 0:
                        y = np.clip(y / p, 0, 1)

                x_list.append(x)
                y_list.append(y)

            # Avoid empty batch
            if len(x_list) == 0:
                x_list.append(np.zeros((Z, H, W, 1), dtype=np.float32))
                y_list.append(np.zeros((Z, H, W, 1), dtype=np.float32))

            x_batch = np.stack(x_list).astype(np.float32)
            y_batch = np.stack(y_list).astype(np.float32)

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# -----------------------------
# Metric
# -----------------------------
def dice_metric(cfg: TrainConfig):
    def _dice(y_true, y_pred):
        y_true_bin = tf.cast(y_true > cfg.metric_threshold, tf.float32)
        y_pred_bin = tf.cast(y_pred > cfg.metric_threshold, tf.float32)
        inter = tf.reduce_sum(y_true_bin * y_pred_bin)
        union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin)
        return (2.0 * inter + 1e-5) / (union + 1e-5)
    return _dice


# -----------------------------
# Train
# -----------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cfg.h5_file, "r") as f:
        total = f["x_patches"].shape[0]

    all_idx = np.arange(total)
    tr_idx, va_idx = train_test_split(all_idx, test_size=cfg.val_frac, random_state=cfg.seed)

    train_gen = DapiTransmissionGenerator(cfg.h5_file, tr_idx, cfg, batch_size=cfg.batch_train, shuffle=True)
    val_gen = DapiTransmissionGenerator(cfg.h5_file, va_idx, cfg, batch_size=cfg.batch_val, shuffle=False)

    model = simple_unet3d(cfg.depth, cfg.H, cfg.W, input_channels=1, output_channels=1, base_filters=cfg.base_filters)

    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
        loss=Dapi_loss,
        metrics=[dice_metric(cfg)],
    )

    best_path = out_dir / f"{cfg.model_name}.keras"
    final_path = out_dir / f"{cfg.model_name}_final.keras"
    log_path = out_dir / f"{cfg.model_name}.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val__dice",  # will be overwritten below to the correct name
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val__dice",
            patience=cfg.patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_path), append=True),
    ]

    # Fix callback monitor names based on the actual metric name
    # Keras uses the function name for metric key.
    metric_name = model.metrics_names[-1]  # e.g. "_dice"
    callbacks[0].monitor = f"val_{metric_name}"
    callbacks[1].monitor = f"val_{metric_name}"

    history = model.fit(train_gen, validation_data=val_gen, epochs=cfg.epochs, callbacks=callbacks)
    model.save(str(final_path))
    return history.history


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)

