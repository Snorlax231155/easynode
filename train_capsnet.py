"""
train_capsnet.py
================
Train a Capsule Network (CapsNet) on the EasyNodule_DL pulmonary nodule dataset.

Three independent models are trained, one per CT view axis:
  - ModelX  ->  mid-axial slice   (depth axis, index 32 of dim 0)
  - ModelY  ->  mid-coronal slice (height axis, index 32 of dim 1)
  - ModelZ  ->  mid-sagittal slice(width axis,  index 32 of dim 2)

Usage
-----
  python train_capsnet.py                  # train all three views
  python train_capsnet.py --axis X         # train ModelX only
  python train_capsnet.py --epochs 30 --batch-size 8
"""

import os, sys, argparse, csv, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import tensorflow as tf
import tensorflow_io as tfio
from PIL import Image
import cv2

# ----------------------------------------------------------------
# Hyper-parameters (match the paper / Preporcessing.py)
# ----------------------------------------------------------------
EPSILON     = 1e-7
M_PLUS      = 0.9
M_MINUS     = 0.1
LAMBDA_     = 0.5
ALPHA       = 0.0005
NUM_CLASSES = 2
ROUTING_ITER = 5

PARAMS = {
    "no_of_conv_kernels"       : 256,
    "no_of_primary_capsules"   : 32,
    "primary_capsule_vector"   : 32,
    "no_of_secondary_capsules" : NUM_CLASSES,
    "secondary_capsule_vector" : 64,
    "routing_iterations"       : ROUTING_ITER,
}


# ----------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------
class CapsuleNetwork(tf.keras.Model):
    """3-layer CapsNet with Gabor filter and dynamic routing."""

    def __init__(self, no_of_conv_kernels, no_of_primary_capsules,
                 primary_capsule_vector, no_of_secondary_capsules,
                 secondary_capsule_vector, routing_iterations):
        super().__init__()
        self.no_of_primary_capsules   = no_of_primary_capsules
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.routing_iterations       = routing_iterations
        self.primary_capsule_vector   = primary_capsule_vector

        self.convolution = tf.keras.layers.Conv2D(
            no_of_conv_kernels, [9, 9], strides=[1, 1],
            activation="relu", name="ConvolutionLayer")
        self.primary_capsule = tf.keras.layers.Conv2D(
            no_of_primary_capsules * primary_capsule_vector,
            [9, 9], strides=[2, 2], padding="same", name="PrimaryCapsule")
        # 4608 = 12 x 12 x 32
        self.w = tf.Variable(
            tf.random.normal([1, 4608, no_of_secondary_capsules,
                              secondary_capsule_vector, primary_capsule_vector],
                             stddev=0.05),
            dtype=tf.float32, name="PoseEstimation", trainable=True)

        self.dense_1   = tf.keras.layers.Dense(256,  activation="relu")
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_2   = tf.keras.layers.Dense(512,  activation="relu")
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dense_3   = tf.keras.layers.Dense(1024, activation="sigmoid", dtype="float32")

    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return (tf.square(s_norm) / (1.0 + tf.square(s_norm))) * (s / (s_norm + EPSILON))

    def _encode(self, input_x, training=False):
        x = self.convolution(input_x, training=training)
        x = self.primary_capsule(x, training=training)

        x_gabor = tfio.experimental.filter.gabor(x, freq=0.7, theta=0.9)
        x_gabor = tf.cast(tf.math.real(x_gabor), dtype=tf.float32)

        batch = tf.shape(x_gabor)[0]
        u = tf.reshape(x_gabor,
            (batch,
             self.no_of_primary_capsules * x.shape[1] * x.shape[2],
             self.primary_capsule_vector))
        u = tf.expand_dims(tf.expand_dims(u, axis=-2), axis=-1)
        u_hat = tf.squeeze(tf.matmul(self.w, u), [4])

        b = tf.zeros((batch, 4608, self.no_of_secondary_capsules, 1))
        for _ in range(self.routing_iterations):
            c = tf.nn.softmax(b, axis=-2)
            s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
            v = self.squash(s)
            agreement = tf.squeeze(
                tf.matmul(tf.expand_dims(u_hat, axis=-1),
                          tf.expand_dims(v,     axis=-1), transpose_a=True), [4])
            b = b + agreement
        return v, u_hat

    def call(self, inputs, training=False):
        input_x, y = inputs
        v, _ = self._encode(input_x, training=training)

        y_exp    = tf.cast(tf.expand_dims(tf.expand_dims(y, axis=-1), axis=1), dtype=tf.float32)
        v_masked = tf.multiply(y_exp, v)

        v_flat  = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector])
        rec     = self.dense_1(v_flat, training=training)
        rec     = self.dropout_1(rec,  training=training)
        rec     = self.dense_2(rec,    training=training)
        rec     = self.dropout_2(rec,  training=training)
        rec_out = self.dense_3(rec)
        return v, rec_out

    def predict_capsule_output(self, inputs, training=False):
        v, _ = self._encode(inputs, training=training)
        return v

    def regenerate_image(self, v):
        v_flat  = tf.reshape(v, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector])
        rec     = self.dense_1(v_flat)
        rec     = self.dropout_1(rec, training=False)
        rec     = self.dense_2(rec)
        rec     = self.dropout_2(rec, training=False)
        return self.dense_3(rec)


# ----------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------
def safe_norm(v, axis=-1):
    return tf.sqrt(tf.reduce_sum(tf.square(v), axis=axis, keepdims=True) + EPSILON)


def margin_loss(y_true, v):
    v_norm  = tf.squeeze(safe_norm(v, axis=-1), [1, 3])
    present = tf.square(tf.maximum(0.0, M_PLUS  - v_norm))
    absent  = tf.square(tf.maximum(0.0, v_norm  - M_MINUS))
    loss    = y_true * present + LAMBDA_ * (1.0 - y_true) * absent
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


def reconstruction_loss(y_true_img, reconstructed):
    flat = tf.reshape(y_true_img, [tf.shape(y_true_img)[0], -1])
    flat = tf.cast(flat, tf.float32)
    return tf.reduce_mean(tf.square(flat - reconstructed))


def total_loss(y_onehot, v, x_img, reconstructed):
    ml = margin_loss(y_onehot, v)
    rl = reconstruction_loss(x_img, reconstructed)
    return ml + ALPHA * rl, ml, rl


# ----------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------
def preprocess_slice(img_2d):
    img = cv2.resize(img_2d.astype(np.uint8), (64, 64))
    pil = Image.fromarray(img)
    left, top = (64 - 32) // 2, (64 - 32) // 2
    sub = np.asarray(pil.crop((left, top, left + 32, top + 32)))
    sub = cv2.GaussianBlur(sub, (3, 3), 3)
    sub = sub.astype(np.float32) / 255.0
    sub = np.expand_dims(sub, axis=-1)
    return sub


def extract_view(volume, axis):
    mid = volume.shape[0] // 2
    if axis == "X": return volume[mid, :, :]
    if axis == "Y": return volume[:, mid, :]
    return volume[:, :, mid]


# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
def load_data(data_path):
    print(f"\n[DATA] Loading {data_path} ...")
    raw = np.load(data_path, allow_pickle=True)
    print(f"[DATA] {len(raw)} samples found.")

    X_data = {"X": [], "Y": [], "Z": []}
    labels = []
    for volume, label in tqdm(raw, desc="Preprocessing"):
        labels.append(int(label))
        for ax in ("X", "Y", "Z"):
            slc = extract_view(volume, ax)
            img = preprocess_slice(slc)
            X_data[ax].append(img)

    for ax in ("X", "Y", "Z"):
        X_data[ax] = np.array(X_data[ax], dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    print(f"[DATA] Per-view shape: {X_data['X'].shape}  Labels: {np.bincount(labels)}")
    return X_data, labels


def split_data(X_views, labels, seed=42):
    idx = np.arange(len(labels))
    idx_train, idx_tmp   = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    idx_val,   idx_test  = train_test_split(idx_tmp, test_size=0.50, stratify=labels[idx_tmp], random_state=seed)

    splits = {}
    for ax in ("X", "Y", "Z"):
        X = X_views[ax]
        splits[ax] = {
            "train": (X[idx_train], labels[idx_train]),
            "val"  : (X[idx_val],   labels[idx_val]),
            "test" : (X[idx_test],  labels[idx_test]),
        }
    print(f"[SPLIT] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    return splits


def make_dataset(X, y, batch_size, shuffle=False):
    y_oh = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES).astype(np.float32)
    ds   = tf.data.Dataset.from_tensor_slices(((X, y_oh), {"images": X, "labels": y_oh}))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------
# Train / val steps
# ----------------------------------------------------------------
@tf.function
def train_step(model, optimizer, x_batch, y_batch):
    with tf.GradientTape() as tape:
        v, reconstructed = model([x_batch, y_batch], training=True)
        loss, ml, rl     = total_loss(y_batch, v, x_batch, reconstructed)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, ml, rl


@tf.function
def val_step(model, x_batch, y_batch):
    v, reconstructed = model([x_batch, y_batch], training=False)
    loss, ml, rl     = total_loss(y_batch, v, x_batch, reconstructed)
    return loss, v


# ----------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------
def evaluate(model, X, y, batch_size=16):
    all_probs, all_preds, all_true = [], [], []
    for i in range(0, len(X), batch_size):
        xb  = tf.constant(X[i:i+batch_size], dtype=tf.float32)
        v   = model.predict_capsule_output(xb, training=False)
        nrm = tf.squeeze(safe_norm(v, axis=-1), [1, 3]).numpy()
        all_probs.extend(nrm[:, 1].tolist())
        all_preds.extend(np.argmax(nrm, axis=1).tolist())
        all_true.extend(y[i:i+batch_size].tolist())

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc  = np.mean(all_preds == all_true)
    try:    auc = roc_auc_score(all_true, all_probs)
    except: auc = float("nan")
    prec = precision_score(all_true, all_preds, zero_division=0)
    rec  = recall_score(all_true, all_preds, zero_division=0)
    f1   = f1_score(all_true, all_preds, zero_division=0)
    return acc, auc, prec, rec, f1


# ----------------------------------------------------------------
# Train one model
# ----------------------------------------------------------------
def train_model(axis, split_data_ax, epochs, batch_size, save_dir, log_writer):
    print(f"\n{'='*60}")
    print(f"  Training Model{axis}  ({axis}-view, {epochs} epochs, bs={batch_size})")
    print(f"{'='*60}")

    X_train, y_train = split_data_ax["train"]
    X_val,   y_val   = split_data_ax["val"]
    X_test,  y_test  = split_data_ax["test"]

    model     = CapsuleNetwork(**PARAMS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Warm-up build
    dummy_x = tf.zeros([1, 32, 32, 1])
    dummy_y = tf.zeros([1, NUM_CLASSES])
    _       = model([dummy_x, dummy_y], training=False)
    print(f"  [Model{axis}] Trainable params: {model.count_params():,}")

    train_ds   = make_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds     = make_dataset(X_val,   y_val,   batch_size, shuffle=False)

    best_val_acc = -1.0
    best_ckpt    = os.path.join(save_dir, f"best_Model{axis}")
    ckpt_dir     = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0             = time.time()
        train_loss_sum = 0.0
        n_batches      = 0
        for (x_b, y_b), _ in tqdm(train_ds,
                                   desc=f"  Epoch {epoch:02d}/{epochs} [train]",
                                   leave=False):
            loss, _, _ = train_step(model, optimizer, x_b, y_b)
            train_loss_sum += float(loss)
            n_batches += 1
        train_loss_avg = train_loss_sum / max(n_batches, 1)

        val_loss_sum = 0.0
        n_val = 0
        for (x_b, y_b), _ in val_ds:
            vloss, _ = val_step(model, x_b, y_b)
            val_loss_sum += float(vloss)
            n_val += 1
        val_loss_avg = val_loss_sum / max(n_val, 1)

        val_acc, val_auc, val_prec, val_rec, val_f1 = evaluate(model, X_val, y_val, batch_size)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:02d}/{epochs}  "
            f"train={train_loss_avg:.4f}  val={val_loss_avg:.4f}  "
            f"acc={val_acc:.4f}  auc={val_auc:.4f}  ({elapsed:.1f}s)"
        )

        log_writer.writerow({
            "axis": axis, "epoch": epoch,
            "train_loss": f"{train_loss_avg:.6f}",
            "val_loss"  : f"{val_loss_avg:.6f}",
            "val_acc"   : f"{val_acc:.6f}",
            "val_auc"   : f"{val_auc:.6f}",
            "val_prec"  : f"{val_prec:.6f}",
            "val_rec"   : f"{val_rec:.6f}",
            "val_f1"    : f"{val_f1:.6f}",
        })

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:02d}")
        model.save_weights(ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(best_ckpt)
            print(f"  >> Best val_acc={best_val_acc:.4f} saved -> {best_ckpt}")

    # Final test eval with best weights
    print(f"\n  [Model{axis}] Loading best weights for test evaluation ...")
    model.load_weights(best_ckpt)
    test_acc, test_auc, test_prec, test_rec, test_f1 = evaluate(model, X_test, y_test, batch_size)
    print(f"  [Model{axis}] TEST acc={test_acc:.4f}  auc={test_auc:.4f}  "
          f"prec={test_prec:.4f}  rec={test_rec:.4f}  f1={test_f1:.4f}")

    # Save final capsule weights (compatible naming with existing codebase)
    final_path = os.path.join(save_dir, f"capsule-{axis}")
    model.save_weights(final_path)
    print(f"  [Model{axis}] Final weights -> {final_path}")

    return model, {"acc": test_acc, "auc": test_auc, "prec": test_prec,
                   "rec": test_rec,  "f1": test_f1}


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CapsNet on EasyNodule_DL")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--axis",        type=str, default="all",
                        choices=["X", "Y", "Z", "all"])
    parser.add_argument("--data-path",   type=str,
                        default=r"archive (1)/DataSet_all.npy")
    parser.add_argument("--weights-dir", type=str, default="weights")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] {len(gpus)} GPU(s) detected")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[GPU] No GPU detected - training on CPU")

    X_views, labels = load_data(args.data_path)
    splits          = split_data(X_views, labels)

    log_path   = "training_log.csv"
    log_fields = ["axis","epoch","train_loss","val_loss",
                  "val_acc","val_auc","val_prec","val_rec","val_f1"]
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    axes    = ["X", "Y", "Z"] if args.axis == "all" else [args.axis]
    results = {}

    for ax in axes:
        save_dir = os.path.join(args.weights_dir, f"Model{ax}")
        os.makedirs(save_dir, exist_ok=True)
        _, metrics = train_model(
            axis=ax, split_data_ax=splits[ax],
            epochs=args.epochs, batch_size=args.batch_size,
            save_dir=save_dir, log_writer=log_writer)
        results[ax] = metrics

    log_file.close()

    print(f"\n{'='*60}")
    print("  FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  {'Axis':<6} {'Acc':>8} {'AUC':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*46}")
    for ax, m in results.items():
        print(f"  {ax:<6} {m['acc']:>8.4f} {m['auc']:>8.4f} "
              f"{m['prec']:>8.4f} {m['rec']:>8.4f} {m['f1']:>8.4f}")
    print(f"\n  Training log -> {log_path}")
    print(f"  Weights      -> {args.weights_dir}/Model{{X,Y,Z}}/")
    print("  Done.")


if __name__ == "__main__":
    main()
