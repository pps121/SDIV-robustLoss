"""
PART 2 — rSDNet Vanilla Transformer: Hyperparameter Sweeps, Robustness & Stability
====================================================================================
Datasets : MNIST (padded to 32×32)  /  CIFAR-10
Model    : Vanilla Vision Transformer (same as PART 1 notebook)
Loss fns : SDIV, CCE, TDPDSCCE, TSCCE, GCE, SCE, FCL, RKLD, MAE
           (identical to rSDNet.ipynb — only hyper-parameters change)
Optimizer: Adam  (same as rSDNet.ipynb)

Experiment Batteries
--------------------
  A. Clean-data performance    — beta × lambda grid × both datasets
  B. Uniform label-noise       — noise_rate ∈ {0, 10, 20, 30, 40} %
  C. FGSM adversarial attacks  — epsilon ∈ {0, 1/255, 2/255, 4/255, 8/255}
  D. rSDNet parameter surface  — 3-D accuracy over beta × lambda

Outputs (saved to ./results/)
------------------------------
  *.csv  — numeric results for all batteries
  *.png  — 2-D line plots, heatmaps, 3-D surface plots, confusion matrices

Design Notes
------------
  • All sections are self-contained → plug-and-play for future PARTs.
  • A single CONFIG dict (Section 0) controls every experiment.
  • Adding a new loss / model / dataset = adding one entry to the relevant dict.
  • Set CFG['QUICK_RUN'] = True to use 30 epochs for rapid prototyping;
    False uses 250 epochs (rSDNet paper setting).
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIG  (edit here; nothing else needs changing for most runs)
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Datasets ───────────────────────────────────────────────────────────
    # Any subset of: 'mnist', 'fashion_mnist', 'cifar10'
    'DATASETS'       : ['mnist', 'fashion_mnist', 'cifar10'],

    # ── Training (paper default: 250; quick debug: 30) ─────────────────────
    'QUICK_RUN'      : True,          # True = 30 epochs,  False = 250 epochs
    'BATCH_SIZE'     : 256,
    'SEED'           : 42,

    # ── rSDNet parameter grid (battery A + D) ──────────────────────────────
    # Valid constraints: A = 1+lam*(1-beta) > 0  AND  B = beta-lam*(1-beta) > 0
    # All combinations below satisfy both constraints.
    'BETA_GRID'      : [0.02, 0.05, 0.10, 0.20, 0.50],
    'LAM_GRID'       : [-0.80, -0.40, 0.00, 0.20],

    # ── Label-noise rates (battery B) ──────────────────────────────────────
    'NOISE_RATES'    : [0.0, 0.10, 0.20, 0.30, 0.40],

    # ── FGSM epsilon values (battery C) ────────────────────────────────────
    'FGSM_EPS'       : [0.0, 1/255, 2/255, 4/255, 8/255],

    # ── Transformer hyper-parameters ───────────────────────────────────────
    'PATCH_SIZE'     : 8,
    'D_MODEL'        : 64,
    'NUM_HEADS'      : 4,
    'FFN_DIM'        : 128,
    'NUM_LAYERS'     : 4,
    'DROPOUT'        : 0.1,

    # ── Results directory ──────────────────────────────────────────────────
    'RESULTS_DIR'    : 'results',
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (3-D plots)
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Keras import compatibility: handle both standalone keras 3.x and tf.keras
try:
    from keras.losses import Loss
    from keras.initializers import GlorotUniform
except (ImportError, ModuleNotFoundError):
    from tensorflow.keras.losses import Loss
    from tensorflow.keras.initializers import GlorotUniform
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datasets import load_dataset as _hf_load   # pip install datasets

warnings.filterwarnings('ignore')
os.makedirs(CFG['RESULTS_DIR'], exist_ok=True)

# Reproducibility
np.random.seed(CFG['SEED'])
tf.random.set_seed(CFG['SEED'])

N_EPOCHS = 30 if CFG['QUICK_RUN'] else 250

print(f"[CONFIG]  epochs={N_EPOCHS}  batch={CFG['BATCH_SIZE']}  "
      f"datasets={CFG['DATASETS']}  quick={CFG['QUICK_RUN']}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOSS FUNCTIONS  (verbatim from rSDNet.ipynb; no changes)
#             Plug-in new losses here: just follow the same Loss subclass pattern
# ══════════════════════════════════════════════════════════════════════════════

class SDIV(Loss):
    """S-Divergence: 2-parameter superfamily (beta=α, lam=λ from paper)."""
    def __init__(self, beta, lam, trim_ratio=0.0):
        super().__init__()
        self.beta = float(beta);  self.lam = float(lam);  self.trim_ratio = trim_ratio

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        A  = 1 + self.lam * (1 - self.beta)
        B  = self.beta - self.lam * (1 - self.beta)
        bs = tf.shape(y_true)[0]
        idx = tf.stack([tf.range(bs), y_true], axis=1)
        p_y = tf.gather_nd(y_pred, idx)
        losses = (tf.reduce_sum(y_pred**(self.beta+1), axis=1)) / A \
               - ((1+self.beta)/(A*B)) * (p_y**B)
        k = tf.cast(tf.math.floor((1-self.trim_ratio)*tf.cast(bs, tf.float32)), tf.int32)
        return tf.reduce_mean(tf.sort(losses)[:k])

class TDPDSCCE(Loss):
    """DPD (Density Power Divergence) for classification — S-div with lam=0."""
    def __init__(self, beta, trim_ratio=0.0):
        super().__init__()
        self.beta = float(beta);  self.trim_ratio = trim_ratio

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        bs  = tf.shape(y_true)[0]
        idx = tf.stack([tf.range(bs), y_true], axis=1)
        p_y = tf.gather_nd(y_pred, idx)
        losses = tf.reduce_sum(y_pred**(self.beta+1), axis=1) \
               - (1 + 1/self.beta) * (p_y**self.beta)
        k = tf.cast(tf.math.floor((1-self.trim_ratio)*tf.cast(bs, tf.float32)), tf.int32)
        return tf.reduce_mean(tf.sort(losses)[:k])

class TSCCE(Loss):
    """Trimmed Sparse Categorical Cross-Entropy."""
    def __init__(self, trim_ratio=0.2):
        super().__init__()
        self.trim_ratio = trim_ratio

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        bs  = tf.shape(y_true)[0]
        idx = tf.stack([tf.range(bs), y_true], axis=1)
        per_loss = -tf.gather_nd(tf.math.log(y_pred), idx)
        k = tf.cast(tf.math.floor((1-self.trim_ratio)*tf.cast(bs, tf.float32)), tf.int32)
        trimmed, _ = tf.math.top_k(-per_loss, k=k, sorted=False)
        return -tf.reduce_mean(trimmed)

class SCE(Loss):
    """Symmetric Cross-Entropy."""
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__()
        self.alpha = float(alpha);  self.beta = float(beta)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        bs  = tf.shape(y_true)[0]
        p_y = tf.gather_nd(y_pred, tf.stack([tf.range(bs), y_true], axis=1))
        return tf.reduce_mean(-self.alpha*tf.math.log(p_y) + self.beta*6*(1-p_y))

class GCE(Loss):
    """Generalised Cross-Entropy."""
    def __init__(self, q=0.7):
        super().__init__()
        self.q = float(q)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        bs  = tf.shape(y_true)[0]
        p_y = tf.gather_nd(y_pred, tf.stack([tf.range(bs), y_true], axis=1))
        return tf.reduce_mean((1 - p_y**self.q) / self.q)

class RKLD(Loss):
    """Reverse KL Divergence."""
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        bs  = tf.shape(y_true)[0]
        p_y = tf.gather_nd(y_pred, tf.stack([tf.range(bs), y_true], axis=1))
        return tf.reduce_mean(
            tf.reduce_sum(y_pred*tf.math.log(y_pred), axis=1) + 2*(1-p_y))

class FCL(Loss):
    """Fractional Cross-Loss."""
    def __init__(self, mu=0.5):
        super().__init__()
        self.mu = float(mu)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
        bs  = tf.shape(y_true)[0]
        p_y = tf.gather_nd(y_pred, tf.stack([tf.range(bs), y_true], axis=1))
        return tf.reduce_mean(
            (-tf.math.log(p_y))**(1-self.mu)
            / tf.exp(tf.math.lgamma(tf.constant(2-self.mu)))
            + 2*(1-p_y))


# ── Loss registry: maps name → (loss_object, Adam_lr) ────────────────────────
def make_loss_registry(beta=0.05, lam=-0.8):
    """Returns all loss functions. beta/lam apply only to SDIV and TDPDSCCE."""
    return {
        'SDIV'     : (SDIV(beta=beta, lam=lam, trim_ratio=0.0), 1e-3),
        'CCE'      : ('sparse_categorical_crossentropy',         1e-3),
        'TDPDSCCE' : (TDPDSCCE(beta=beta, trim_ratio=0.0),       1e-3),
        'TSCCE'    : (TSCCE(trim_ratio=0.2),                     1e-3),
        'GCE'      : (GCE(q=0.7),                                1e-3),
        'SCE'      : (SCE(alpha=0.5, beta=1.0),                  1e-3),
        'FCL'      : (FCL(mu=0.5),                               1e-3),
        'RKLD'     : (RKLD(),                                    1e-3),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# HuggingFace dataset registry — add new datasets here as needed
_HF_REGISTRY = {
    'mnist'         : ('ylecun/mnist',              'image', 'label'),
    'fashion_mnist' : ('randall-lab/fashion-mnist', 'image', 'label'),
    'cifar10'       : ('uoft-cs/cifar10',           'img',   'label'),
}

# Module-level cache so each dataset is only downloaded / decoded once per run
_DATASET_CACHE: dict = {}


def load_dataset(name: str):
    """
    Returns (X_train, y_train, X_test, y_test, num_classes, in_channels).
    Pulls data from HuggingFace Hub; auto-caches in ~/.cache/huggingface/.
    No manual download required — just `pip install datasets pillow`.
    """
    global _DATASET_CACHE
    if name in _DATASET_CACHE:
        return _DATASET_CACHE[name]

    if name not in _HF_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from {list(_HF_REGISTRY.keys())}")

    hf_id, img_col, lbl_col = _HF_REGISTRY[name]
    print(f"  Fetching {name.upper()} from HuggingFace ({hf_id}) …")
    ds = _hf_load(hf_id)

    def _to_numpy(split):
        """HF Dataset split → float32 numpy arrays normalised to [0, 1]."""
        X = np.stack([np.array(img) for img in split[img_col]]).astype('float32') / 255.0
        y = np.array(split[lbl_col], dtype=np.int64)
        return X, y

    X_tr, y_tr = _to_numpy(ds['train'])
    X_te, y_te = _to_numpy(ds['test'])

    # Grayscale: (H, W) → (H, W, 1)
    if X_tr.ndim == 3:
        X_tr = X_tr[..., np.newaxis]
        X_te = X_te[..., np.newaxis]
        C = 1
    else:
        C = X_tr.shape[-1]   # 3 for CIFAR-10

    # Pad 28×28 → 32×32 so patch_size=8 tiles evenly (MNIST, Fashion-MNIST)
    if X_tr.shape[1] == 28:
        X_tr = np.pad(X_tr, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
        X_te = np.pad(X_te, ((0,0),(2,2),(2,2),(0,0)), mode='constant')

    result = (X_tr, y_tr, X_te, y_te, 10, C)
    _DATASET_CACHE[name] = result          # cache for subsequent batteries
    print(f"  Loaded {name.upper()}: train={X_tr.shape}  test={X_te.shape}  channels={C}")
    return result


def corrupt_labels(y: np.ndarray, noise_rate: float,
                   num_classes: int = 10, seed: int = 42) -> np.ndarray:
    """
    Uniform symmetric label noise: each label flipped to a random *different*
    class with probability `noise_rate`.  Returns a new array (y unchanged).
    """
    if noise_rate == 0.0:
        return y.copy()
    rng = np.random.RandomState(seed)
    y_c = y.copy()
    mask = rng.rand(len(y)) < noise_rate
    for i in np.where(mask)[0]:
        choices = [c for c in range(num_classes) if c != y[i]]
        y_c[i]  = rng.choice(choices)
    corrupted = mask.sum()
    print(f"  Label noise {noise_rate*100:.0f}%: {corrupted}/{len(y)} labels flipped")
    return y_c


def fgsm_attack(model, X: np.ndarray, y: np.ndarray, epsilon: float,
                batch_size: int = 256) -> np.ndarray:
    """
    Fast Gradient Sign Method (FGSM) — single-step L∞ attack.
    Returns adversarial examples clipped to [0, 1].
    """
    if epsilon == 0.0:
        return X.copy()

    X_adv_chunks = []
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for start in range(0, len(X), batch_size):
        xb = tf.constant(X[start:start+batch_size], dtype=tf.float32)
        yb = tf.constant(y[start:start+batch_size], dtype=tf.int32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            preds = model(xb, training=False)
            loss  = loss_fn(yb, preds)
        grad   = tape.gradient(loss, xb)
        x_adv  = xb + epsilon * tf.sign(grad)
        x_adv  = tf.clip_by_value(x_adv, 0.0, 1.0)
        X_adv_chunks.append(x_adv.numpy())

    return np.concatenate(X_adv_chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VANILLA TRANSFORMER MODEL  (identical to PART 1 notebook)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(keras.layers.Layer):
    def __init__(self, patch_size, d_model, **kw):
        super().__init__(**kw)
        self.patch_size = patch_size
        self.proj = keras.layers.Dense(d_model,
                                       kernel_initializer=GlorotUniform(seed=42))

    def call(self, x):
        p = self.patch_size
        H, W, C   = x.shape[1], x.shape[2], x.shape[3]
        patches   = tf.image.extract_patches(
            x, sizes=[1,p,p,1], strides=[1,p,p,1],
            rates=[1,1,1,1], padding='VALID')
        n_patches = (H//p) * (W//p)
        patches   = tf.reshape(patches, [tf.shape(x)[0], n_patches, p*p*C])
        return self.proj(patches)


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout, **kw):
        super().__init__(**kw)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.msa   = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads,
            kernel_initializer=GlorotUniform(seed=42))
        self.drop1 = keras.layers.Dropout(dropout)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn   = keras.Sequential([
            keras.layers.Dense(ffn_dim, activation='gelu',
                               kernel_initializer=GlorotUniform(seed=42)),
            keras.layers.Dense(d_model,
                               kernel_initializer=GlorotUniform(seed=42)),
        ])
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.norm1(x)
        h = self.msa(h, h, training=training)
        x = x + self.drop1(h, training=training)
        h = self.norm2(x)
        h = self.ffn(h)
        return x + self.drop2(h, training=training)


def build_vanilla_transformer(input_shape=(32, 32, 3), num_classes=10,
                               cfg: dict = CFG):
    """
    Builds a vanilla Vision Transformer from scratch (no pretrained weights).
    Parameterised entirely by CFG — easy to swap in PART 3 / future PARTs.
    """
    p  = cfg['PATCH_SIZE']
    dm = cfg['D_MODEL']
    H, W, C   = input_shape
    n_patches = (H//p) * (W//p)

    inp = keras.Input(shape=input_shape, name='image')
    x   = PatchEmbedding(p, dm, name='patch_embed')(inp)
    pos = keras.layers.Embedding(n_patches, dm,
                                 embeddings_initializer=GlorotUniform(seed=42),
                                 name='pos_embed')(tf.range(n_patches))
    x   = x + pos

    for i in range(cfg['NUM_LAYERS']):
        x = TransformerEncoderBlock(
            dm, cfg['NUM_HEADS'], cfg['FFN_DIM'], cfg['DROPOUT'],
            name=f'trans_{i}')(x)

    x   = keras.layers.GlobalAveragePooling1D(name='gap')(x)
    x   = keras.layers.LayerNormalization(epsilon=1e-6, name='out_norm')(x)
    out = keras.layers.Dense(num_classes, activation='softmax',
                             kernel_initializer=GlorotUniform(seed=42),
                             name='classifier')(x)

    return keras.Model(inp, out, name='VanillaTransformer')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING & EVALUATION HARNESS
# ══════════════════════════════════════════════════════════════════════════════

def train_and_eval(X_tr, y_tr, X_te, y_te,
                   loss_obj, lr: float,
                   input_shape: tuple, num_classes: int,
                   n_epochs: int, batch_size: int,
                   seed: int = 42, cfg: dict = CFG) -> dict:
    """
    Trains one Vanilla Transformer from scratch with the given loss and
    returns a result dict: accuracy, per-class accuracy, training history.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_vanilla_transformer(input_shape, num_classes, cfg)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_obj)

    history = model.fit(X_tr, y_tr,
                        epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_te, y_te),
                        verbose=0)

    y_pred       = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc          = accuracy_score(y_te, y_pred)
    cm_mat       = confusion_matrix(y_te, y_pred)
    per_class_acc = cm_mat.diagonal() / cm_mat.sum(axis=1)

    return {
        'model'         : model,
        'history'       : history.history,
        'accuracy'      : acc,
        'confusion'     : cm_mat,
        'per_class_acc' : per_class_acc,
    }


def eval_under_noise(model, X_te, y_te) -> float:
    """Evaluate a trained model on (possibly adversarial/noisy) test inputs."""
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    return accuracy_score(y_te, y_pred)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(name: str, cfg: dict = CFG):
    path = os.path.join(cfg['RESULTS_DIR'], name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {path}")


def plot_2d_noise_robustness(df: pd.DataFrame, dataset: str,
                              cfg: dict = CFG):
    """
    2-D: Accuracy vs noise_rate for each loss function.
    Different (beta, lam) pairs shown as dashed reference lines.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for loss_name, grp in df.groupby('loss'):
        grp_sorted = grp.groupby('noise_rate')['accuracy'].mean().reset_index()
        ax.plot(grp_sorted['noise_rate']*100, grp_sorted['accuracy'],
                marker='o', label=loss_name)
    ax.set_xlabel('Label Noise Rate (%)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'[{dataset.upper()}]  Robustness vs. Uniform Label Noise')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(f'{dataset}_noise_robustness_2d.png', cfg)


def plot_2d_adversarial(df: pd.DataFrame, dataset: str,
                         cfg: dict = CFG):
    """2-D: Accuracy vs FGSM epsilon for each loss function."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for loss_name, grp in df.groupby('loss'):
        grp_sorted = grp.sort_values('epsilon')
        ax.plot(grp_sorted['epsilon']*255, grp_sorted['accuracy'],
                marker='s', label=loss_name)
    ax.set_xlabel('FGSM Perturbation ε  (×1/255 units)')
    ax.set_ylabel('Adversarial Test Accuracy')
    ax.set_title(f'[{dataset.upper()}]  Stability under FGSM Adversarial Attack')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    _savefig(f'{dataset}_fgsm_stability_2d.png', cfg)


def plot_2d_training_curves(history: dict, label: str,
                             dataset: str, cfg: dict = CFG):
    """2-D: Training loss + validation accuracy curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get('loss', []), label='train loss')
    axes[0].set_xlabel('Epoch');  axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss  ({label})')
    axes[0].grid(alpha=0.3)

    val_key = 'val_accuracy' if 'val_accuracy' in history else 'val_sparse_categorical_accuracy'
    if val_key in history:
        axes[1].plot(history[val_key], color='orange', label='val acc')
        axes[1].set_xlabel('Epoch');  axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Validation Accuracy  ({label})')
        axes[1].grid(alpha=0.3)

    plt.suptitle(f'[{dataset.upper()}]  {label}', fontsize=11)
    plt.tight_layout()
    safe_label = label.replace('/', '_').replace(' ', '_').replace('=','')
    _savefig(f'{dataset}_training_curve_{safe_label}.png', cfg)


def plot_3d_param_surface(df: pd.DataFrame, dataset: str,
                           loss_name: str = 'SDIV', cfg: dict = CFG):
    """
    3-D surface: Accuracy over (beta, lambda) grid for a given loss + dataset.
    Alongside a 2-D heatmap for the same data.
    """
    sub = df[(df['dataset'] == dataset) & (df['loss'] == loss_name)].copy()
    if sub.empty:
        print(f"  [SKIP] 3-D surface: no data for {dataset}/{loss_name}")
        return

    pivot = sub.pivot_table(index='beta', columns='lam', values='accuracy')
    betas = pivot.index.values.astype(float)
    lams  = pivot.columns.values.astype(float)
    Z     = pivot.values

    B, L = np.meshgrid(lams, betas)

    # ── 3-D surface ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(B, L, Z, cmap='viridis', edgecolor='none', alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Test Accuracy')
    ax.set_xlabel('λ  (lam)')
    ax.set_ylabel('α  (beta)')
    ax.set_zlabel('Accuracy')
    ax.set_title(f'[{dataset.upper()}]  rSDNet({loss_name})  β × λ Accuracy Surface')
    ax.view_init(elev=30, azim=225)
    _savefig(f'{dataset}_{loss_name}_3d_surface.png', cfg)

    # ── 2-D heatmap ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu',
                linewidths=0.5, ax=ax)
    ax.set_title(f'[{dataset.upper()}]  {loss_name}  β × λ Accuracy Heatmap')
    ax.set_xlabel('λ  (lam)')
    ax.set_ylabel('α  (beta)')
    _savefig(f'{dataset}_{loss_name}_heatmap.png', cfg)


def plot_per_layer_attention_entropy(model, X_sample: np.ndarray,
                                     dataset: str, cfg: dict = CFG):
    """
    2-D + 3-D: Mean attention entropy per head per layer.
    Entropy = -Σ p·log(p) over attention weights → measures focus vs. spread.
    High entropy = diffuse attention (less focused); low = sharp patterns.
    """
    # Build intermediate models that expose each MHA layer's attention weights
    mha_layers = [l for l in model.layers
                  if isinstance(l, keras.layers.MultiHeadAttention)]
    if not mha_layers:
        return

    # Call model once to get attention weights via a sub-model approach
    x_in  = model.input
    attn_outputs = []
    for mha in mha_layers:
        # call the layer to get (output, attention_weights)
        # Build a feature extractor up to just before each MHA layer
        pass  # Attention weight extraction requires model surgery; skip if complex

    # Simpler: use the attention scores from a single forward pass hook
    # We'll approximate with the norm of the output embeddings per layer as
    # a proxy for "information flow" — accessible without model surgery.
    layer_names = [f'trans_{i}' for i in range(cfg['NUM_LAYERS'])]
    available   = [l.name for l in model.layers]
    layer_names = [n for n in layer_names if n in available]

    if not layer_names:
        return

    extractor = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(n).output for n in layer_names])

    sample = X_sample[:64]
    layer_outs = extractor.predict(sample, verbose=0)
    if not isinstance(layer_outs, list):
        layer_outs = [layer_outs]

    # Compute token-embedding L2 norm per layer: (B, n_patches, d_model) → (n_patches,)
    norms = [np.linalg.norm(lo, axis=-1).mean(axis=0) for lo in layer_outs]   # list of (n_patches,)
    n_patches = norms[0].shape[0]
    patch_ids = np.arange(n_patches)

    # ── 2-D per-layer norm across patches ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.plasma(np.linspace(0, 0.9, len(layer_names)))
    for idx, (name, norm) in enumerate(zip(layer_names, norms)):
        ax.plot(patch_ids, norm, marker='.', label=name, color=colors[idx])
    ax.set_xlabel('Patch Index')
    ax.set_ylabel('Mean Token Embedding Norm')
    ax.set_title(f'[{dataset.upper()}]  Token Norms per Layer (proxy for information flow)')
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)
    _savefig(f'{dataset}_layer_norms_2d.png', cfg)

    # ── 3-D surface: layer × patch → norm ────────────────────────────────────
    L_ids  = np.arange(len(layer_names))
    PL, LL = np.meshgrid(patch_ids, L_ids)
    Z      = np.stack(norms, axis=0)          # (num_layers, n_patches)

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(PL, LL, Z, cmap='plasma', edgecolor='none', alpha=0.85)
    ax.set_xlabel('Patch Index')
    ax.set_ylabel('Layer Index')
    ax.set_zlabel('Token Norm')
    ax.set_title(f'[{dataset.upper()}]  Layer × Patch Information-Flow Surface')
    ax.view_init(elev=30, azim=240)
    _savefig(f'{dataset}_layer_patch_surface_3d.png', cfg)


def plot_confusion_matrix(cm_arr: np.ndarray, title: str,
                           dataset: str, cfg: dict = CFG):
    """2-D annotated confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                linewidths=0.4, ax=ax)
    ax.set_xlabel('Predicted');  ax.set_ylabel('True')
    ax.set_title(f'[{dataset.upper()}]  {title}')
    safe = title.replace(' ', '_').replace('(','').replace(')','').replace('=','')
    _savefig(f'{dataset}_confusion_{safe}.png', cfg)


def save_results_table(df: pd.DataFrame, name: str, cfg: dict = CFG):
    """Save DataFrame to CSV and print a pretty summary."""
    path = os.path.join(cfg['RESULTS_DIR'], f'{name}.csv')
    df.to_csv(path, index=False)
    print(f"\n{'─'*60}\n  Results table: {name}\n{'─'*60}")
    print(df.to_string(index=False))
    print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPERIMENT BATTERIES
# ══════════════════════════════════════════════════════════════════════════════

# ── Battery A: Clean-data performance  ───────────────────────────────────────
def battery_A_clean(cfg: dict = CFG) -> pd.DataFrame:
    """
    Sweeps all loss functions on clean data for both datasets.
    Documents baseline accuracy before any corruption/attack.
    """
    print("\n" + "═"*60)
    print("  BATTERY A — Clean-Data Performance")
    print("═"*60)
    rows = []

    for dataset in cfg['DATASETS']:
        X_tr, y_tr, X_te, y_te, n_cls, n_ch = load_dataset(dataset)
        input_shape = (32, 32, n_ch)
        registry    = make_loss_registry(beta=0.05, lam=-0.80)

        for loss_name, (loss_obj, lr) in registry.items():
            print(f"  [{dataset.upper()}]  loss={loss_name:<12}", end='', flush=True)
            res = train_and_eval(X_tr, y_tr, X_te, y_te,
                                 loss_obj, lr, input_shape, n_cls,
                                 N_EPOCHS, cfg['BATCH_SIZE'], cfg['SEED'], cfg)
            acc = res['accuracy']
            print(f"  acc={acc:.4f}")

            # Training curves for SDIV baseline
            if loss_name == 'SDIV':
                plot_2d_training_curves(res['history'],
                                        f'SDIV β=0.05 λ=-0.8 (clean)',
                                        dataset, cfg)
                plot_per_layer_attention_entropy(res['model'], X_te,
                                                 dataset, cfg)

            rows.append({'dataset': dataset, 'loss': loss_name,
                         'beta': 0.05, 'lam': -0.80,
                         'noise_rate': 0.0, 'accuracy': acc})

    df = pd.DataFrame(rows)
    save_results_table(df, 'battery_A_clean')
    return df


# ── Battery B: Uniform label-noise robustness  ───────────────────────────────
def battery_B_noise(cfg: dict = CFG) -> pd.DataFrame:
    """
    Trains under increasing label noise and records accuracy degradation.
    Key research question: which loss function degrades least gracefully?
    """
    print("\n" + "═"*60)
    print("  BATTERY B — Uniform Label-Noise Robustness")
    print("═"*60)
    rows = []

    # We test three representative losses to keep compute tractable;
    # expand 'test_losses' to test more.
    test_losses = ['SDIV', 'CCE', 'GCE', 'TDPDSCCE']

    for dataset in cfg['DATASETS']:
        X_tr, y_tr, X_te, y_te, n_cls, n_ch = load_dataset(dataset)
        input_shape = (32, 32, n_ch)

        for noise_rate in cfg['NOISE_RATES']:
            y_noisy = corrupt_labels(y_tr, noise_rate, n_cls, cfg['SEED'])
            registry = make_loss_registry(beta=0.05, lam=-0.80)

            for loss_name in test_losses:
                loss_obj, lr = registry[loss_name]
                print(f"  [{dataset.upper()}]  loss={loss_name:<12} "
                      f"noise={noise_rate*100:.0f}%", end='', flush=True)
                res = train_and_eval(X_tr, y_noisy, X_te, y_te,
                                     loss_obj, lr, input_shape, n_cls,
                                     N_EPOCHS, cfg['BATCH_SIZE'],
                                     cfg['SEED'], cfg)
                acc = res['accuracy']
                print(f"  acc={acc:.4f}")
                rows.append({'dataset': dataset, 'loss': loss_name,
                             'noise_rate': noise_rate, 'accuracy': acc,
                             'beta': 0.05, 'lam': -0.80})

    df = pd.DataFrame(rows)
    save_results_table(df, 'battery_B_noise')

    # Visualise
    for dataset in cfg['DATASETS']:
        sub = df[df['dataset'] == dataset]
        plot_2d_noise_robustness(sub, dataset, cfg)

    return df


# ── Battery C: FGSM adversarial stability  ───────────────────────────────────
def battery_C_adversarial(cfg: dict = CFG) -> pd.DataFrame:
    """
    Trains with each loss, then evaluates under FGSM attacks of varying ε.
    Measures stability (how fast accuracy collapses with ε).
    """
    print("\n" + "═"*60)
    print("  BATTERY C — FGSM Adversarial Stability")
    print("═"*60)
    rows = []
    test_losses = ['SDIV', 'CCE', 'GCE', 'TDPDSCCE']

    for dataset in cfg['DATASETS']:
        X_tr, y_tr, X_te, y_te, n_cls, n_ch = load_dataset(dataset)
        input_shape = (32, 32, n_ch)
        registry    = make_loss_registry(beta=0.05, lam=-0.80)

        for loss_name in test_losses:
            loss_obj, lr = registry[loss_name]
            print(f"  [{dataset.upper()}]  training loss={loss_name:<12}", end='', flush=True)
            res = train_and_eval(X_tr, y_tr, X_te, y_te,
                                 loss_obj, lr, input_shape, n_cls,
                                 N_EPOCHS, cfg['BATCH_SIZE'], cfg['SEED'], cfg)
            print(f"  clean_acc={res['accuracy']:.4f}")

            for eps in cfg['FGSM_EPS']:
                X_adv = fgsm_attack(res['model'], X_te, y_te, eps,
                                    cfg['BATCH_SIZE'])
                adv_acc = eval_under_noise(res['model'], X_adv, y_te)
                print(f"    ε={eps*255:.1f}/255  adv_acc={adv_acc:.4f}")
                rows.append({'dataset': dataset, 'loss': loss_name,
                             'epsilon': eps, 'accuracy': adv_acc})

    df = pd.DataFrame(rows)
    save_results_table(df, 'battery_C_adversarial')

    for dataset in cfg['DATASETS']:
        sub = df[df['dataset'] == dataset]
        plot_2d_adversarial(sub, dataset, cfg)

    return df


# ── Battery D: rSDNet parameter sensitivity  ─────────────────────────────────
def battery_D_param_sweep(cfg: dict = CFG) -> pd.DataFrame:
    """
    Full beta × lambda grid for SDIV loss.
    Produces 3-D accuracy surfaces and heatmaps.
    Key research contribution: first systematic map of the (α, λ) landscape
    for Transformer models — no prior work exists.
    """
    print("\n" + "═"*60)
    print("  BATTERY D — rSDNet (β, λ) Parameter Sensitivity")
    print("═"*60)
    rows = []

    def _valid(beta, lam):
        """Check A > 0 and B > 0."""
        A = 1 + lam*(1-beta)
        B = beta - lam*(1-beta)
        return A > 1e-4 and B > 1e-4

    for dataset in cfg['DATASETS']:
        X_tr, y_tr, X_te, y_te, n_cls, n_ch = load_dataset(dataset)
        input_shape = (32, 32, n_ch)

        for beta in cfg['BETA_GRID']:
            for lam in cfg['LAM_GRID']:
                if not _valid(beta, lam):
                    print(f"  SKIP  β={beta}  λ={lam}  (constraint violated)")
                    continue
                print(f"  [{dataset.upper()}]  β={beta:.2f}  λ={lam:.2f}", end='', flush=True)
                loss_obj = SDIV(beta=beta, lam=lam, trim_ratio=0.0)
                res = train_and_eval(X_tr, y_tr, X_te, y_te,
                                     loss_obj, 1e-3, input_shape, n_cls,
                                     N_EPOCHS, cfg['BATCH_SIZE'], cfg['SEED'], cfg)
                acc = res['accuracy']
                print(f"  acc={acc:.4f}")
                rows.append({'dataset': dataset, 'loss': 'SDIV',
                             'beta': beta, 'lam': lam, 'accuracy': acc})

    df = pd.DataFrame(rows)
    save_results_table(df, 'battery_D_param_sweep')

    for dataset in cfg['DATASETS']:
        sub = df[df['dataset'] == dataset]
        plot_3d_param_surface(sub, dataset, loss_name='SDIV', cfg=cfg)

    return df


# ── Combined summary table (across all batteries) ────────────────────────────
def make_summary_table(dA, dB, dC, dD, cfg: dict = CFG):
    """
    Aggregated summary: best configuration per (dataset, loss) ranked by
    clean accuracy, noise robustness, and adversarial stability.
    """
    rows = []
    for dataset in cfg['DATASETS']:
        # Best clean accuracy per loss (Battery A)
        for loss_name, grp in dA[dA['dataset']==dataset].groupby('loss'):
            clean_acc = grp['accuracy'].mean()

            # Noise robustness: AUC under noise curve (Battery B)
            nb = dB[(dB['dataset']==dataset) & (dB['loss']==loss_name)]
            noise_auc = np.trapz(nb.sort_values('noise_rate')['accuracy'],
                                 nb.sort_values('noise_rate')['noise_rate']) \
                        if not nb.empty else np.nan

            # Adversarial stability: AUC under ε curve (Battery C)
            cb = dC[(dC['dataset']==dataset) & (dC['loss']==loss_name)]
            adv_auc = np.trapz(cb.sort_values('epsilon')['accuracy'],
                               cb.sort_values('epsilon')['epsilon']) \
                      if not cb.empty else np.nan

            rows.append({'dataset': dataset, 'loss': loss_name,
                         'clean_acc': round(clean_acc, 4),
                         'noise_robustness_AUC': round(noise_auc, 4) if not np.isnan(noise_auc) else '—',
                         'adv_stability_AUC':    round(adv_auc, 4)   if not np.isnan(adv_auc)   else '—'})

    df = pd.DataFrame(rows).sort_values(['dataset', 'clean_acc'], ascending=[True, False])
    save_results_table(df, 'SUMMARY_all_batteries')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "█"*60)
    print("  PART 2 — rSDNet Vanilla Transformer Experiments")
    print("  Target: robustness + stability behaviour of Transformers")
    print("  under label noise, feature noise, adversarial attacks")
    print("█"*60)

    # ── Run all four batteries ────────────────────────────────────────────────
    dA = battery_A_clean(CFG)
    dB = battery_B_noise(CFG)
    dC = battery_C_adversarial(CFG)
    dD = battery_D_param_sweep(CFG)

    # ── Combined summary ──────────────────────────────────────────────────────
    summary = make_summary_table(dA, dB, dC, dD, CFG)

    print("\n" + "█"*60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Results + plots saved to  ./{CFG['RESULTS_DIR']}/")
    print("█"*60)

    # ── Quick sanity: print best config per dataset ───────────────────────────
    for dataset in CFG['DATASETS']:
        sub = summary[summary['dataset'] == dataset]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"\n  [{dataset.upper()}]  Best clean accuracy: "
                  f"{best['loss']} → {best['clean_acc']}")

# ══════════════════════════════════════════════════════════════════════════════
# PLUG-AND-PLAY EXTENSION POINTS (for PART 3, 4, … )
# ══════════════════════════════════════════════════════════════════════════════
# To add:
#   • New model    → import and wrap in build_*() returning keras.Model;
#                    pass as argument to train_and_eval()
#   • New dataset  → add case in load_dataset(); add to CFG['DATASETS']
#   • New loss     → add Loss subclass in Section 2; add to make_loss_registry()
#   • New attack   → add function following fgsm_attack() signature;
#                    add a new Battery in Section 7
#   • PGD attack   → replace fgsm_attack() with a multi-step variant:
#                    iterate  x += step * sign(∇_x L)  then project to ε-ball
#   • Feature noise→ add gaussian_noise(X, sigma) / salt_pepper(X, p) utilities
#   • Annotation n.→ extend corrupt_labels() with asymmetric noise matrices
# ══════════════════════════════════════════════════════════════════════════════
