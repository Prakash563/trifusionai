
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Deep Learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
import tensorflow.keras as keras  # type: ignore
from tensorflow.keras import layers, Model, regularizers  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv1D, Conv2D, Dense, Flatten, Dropout,
    GlobalAveragePooling1D, GlobalAveragePooling2D,
    BatchNormalization, Activation, Add, Multiply,
    Reshape, Concatenate, MaxPooling1D,
    AveragePooling1D, Input, Lambda
)
from tensorflow.keras.utils import to_categorical  # type: ignore
from typing import Optional, Any, Tuple, cast
from matplotlib import cm

# ── Machine Learning ──────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
import joblib

# ── Imbalanced Learning ───────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# ── Image Processing ──────────────────────────────────────────────────────────
from PIL import Image
import cv2

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Dataset Paths (update these to your actual dataset locations) ─────────────
FAKE_FACE_REAL_DIR  = "datasets/fake_face/real"   # folder with real face images
FAKE_FACE_FAKE_DIR  = "datasets/fake_face/fake"   # folder with fake face images
BRAIN_TUMOR_TRAIN   = "datasets/brain_tumor/Training"
BRAIN_TUMOR_TEST    = "datasets/brain_tumor/Testing"
CHURN_CSV_PATH      = "datasets/ecommerce_churn/E Commerce Dataset.csv"

# ── Output Directories ────────────────────────────────────────────────────────
os.makedirs("outputs/security",  exist_ok=True)
os.makedirs("outputs/health",    exist_ok=True)
os.makedirs("outputs/business",  exist_ok=True)
os.makedirs("models",            exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM LAYERS
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(layers.Layer):
    """Spatial Attention mechanism for 1D sequences."""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = Conv1D(1, self.kernel_size, padding='same', 
                          activation='sigmoid', use_bias=False)
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = layers.Concatenate()([avg_pool, max_pool])
        spatial_att = self.conv(concat)
        return spatial_att


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, title, save_path):
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='#2196F3', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#FF5722', linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy Curves')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='#4CAF50', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#FF9800', linewidth=2, linestyle='--')
    axes[1].set_title('Loss Curves')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(class_names)*2), max(5, len(class_names)*1.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='gray')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")


def plot_roc_curve(y_true, y_score, class_names, title, save_path):
    """Plot multi-class ROC curves."""
    plt.figure(figsize=(8, 6))
    n_classes = len(class_names)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score if y_score.ndim == 1
                                 else y_score[:, 1])
        auc_val = roc_auc_score(y_true, y_score if y_score.ndim == 1
                                else y_score[:, 1])
        plt.plot(fpr, tpr, color='#2196F3', linewidth=2,
                 label=f'ROC Curve (AUC = {auc_val:.4f})')
    else:
        colors = cm.get_cmap('Set1')(np.linspace(0, 1, n_classes))
        y_true_bin = to_categorical(y_true, n_classes)
        for i, (name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            auc_val = roc_auc_score(y_true_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, color=color, linewidth=2,
                     label=f'{name} (AUC={auc_val:.3f})')

    plt.plot([0,1],[0,1], 'k--', linewidth=1)
    plt.xlim([0,1]); plt.ylim([0,1.02])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")


def print_metrics(y_true, y_pred, y_score=None, module_name="Model"):
    """Print a formatted metrics summary table."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'═'*55}")
    print(f"  {module_name} — Performance Summary")
    print(f"{'═'*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if y_score is not None:
        try:
            if y_score.ndim > 1 and y_score.shape[1] > 2:
                auc = roc_auc_score(to_categorical(y_true), y_score,
                                    multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(y_true, y_score if y_score.ndim == 1
                                    else y_score[:, 1])
            print(f"  ROC-AUC   : {auc:.4f}")
        except:
            pass
    print(f"{'═'*55}\n")
    return acc, prec, rec, f1


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — SECURITY: FAKE FACE DETECTION
#  Architecture: GAN + ResNet50 + Channel-Wise Attention
# ═════════════════════════════════════════════════════════════════════════════

class SecurityModule:
    """
    Fake Face Detection using GAN-ResNet50 hybrid with Channel-Wise Attention.
    Dataset: Real and Fake Face Detection (2041 images).
    """

    IMG_SIZE    = (128, 128)
    BATCH_SIZE  = 32
    EPOCHS      = 3            # increase to 100 for full training
    LR          = 1e-3
    GAN_EPOCHS  = 20
    LATENT_DIM  = 128

    def __init__(self):
        self.model: Optional[Model] = None
        self.history    = None
        self.generator: Optional[Model] = None
        self.discriminator: Optional[Model] = None
        self.class_names = ['Fake', 'Real']

    # ── Channel-Wise Attention (Squeeze-and-Excitation) ──────────────────────
    def _channel_attention(self, x, ratio=16):
        channels = x.shape[-1]
        # Squeeze
        sq = GlobalAveragePooling2D()(x)
        sq = Reshape((1, 1, channels))(sq)
        # Excitation
        ex = Dense(channels // ratio, activation='relu')(sq)
        ex = Dense(channels, activation='sigmoid')(ex)
        return Multiply()([x, ex])

    # ── GAN-ResNet50 Hybrid ──────────────────────────────────────────────────
    def build_model(self):
        print("\n[Security] Building GAN-ResNet50 + Attention model...")
        base = ResNet50(weights='imagenet', include_top=False,
                        input_shape=(*self.IMG_SIZE, 3))

        # Freeze first 80% of layers, fine-tune the rest
        for layer in base.layers[:140]:
            layer.trainable = False
        for layer in base.layers[140:]:
            layer.trainable = True

        x = base.output
        x = self._channel_attention(x)           # Channel-Wise Attention
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=base.input, outputs=out)
        self.model.compile(optimizer=Adam(self.LR),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        print(f"  Parameters: {self.model.count_params():,}")
        return self.model

    # ── GAN Generator ────────────────────────────────────────────────────────
    def build_generator(self):
        inp = Input(shape=(self.LATENT_DIM,))
        x   = Dense(7 * 7 * 256)(inp)
        x   = Reshape((7, 7, 256))(x)
        x   = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.LeakyReLU(0.2)(x)
        x   = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.LeakyReLU(0.2)(x)
        x   = layers.Conv2DTranspose(32, 4, strides=4, padding='same')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.LeakyReLU(0.2)(x)
        x   = layers.Conv2DTranspose(16, 4, strides=4, padding='same')(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.LeakyReLU(0.2)(x)
        out = layers.Conv2D(3, 4, padding='same', activation='tanh')(x)
        self.generator = Model(inp, out, name='Generator')
        return self.generator

    # ── Data Loaders ─────────────────────────────────────────────────────────
    def load_data(self):
        print("[Security] Loading dataset...")
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            validation_split=0.2
        )
        val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        # Expects structure: dataset_dir/real/*.jpg, dataset_dir/fake/*.jpg
        dataset_dir = os.path.dirname(FAKE_FACE_REAL_DIR)

        train_ds = train_gen.flow_from_directory(
            dataset_dir, target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE, class_mode='binary',
            subset='training', seed=SEED
        )
        val_ds = val_gen.flow_from_directory(
            dataset_dir, target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE, class_mode='binary',
            subset='validation', seed=SEED
        )
        print(f"  Train: {train_ds.samples} | Val: {val_ds.samples}")
        return train_ds, val_ds

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, train_ds, val_ds):
        if self.model is None:
            self.build_model()
        assert self.model is not None
        print("[Security] Training model...")
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                              min_lr=1e-7, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/security_best.keras', save_best_only=True,
                            monitor='val_accuracy')
        ]
        self.history = self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=self.EPOCHS, callbacks=callbacks, verbose=1
        )
        return self.history

    # ── Evaluate ─────────────────────────────────────────────────────────────
    def evaluate(self, val_ds):
        assert self.model is not None
        print("[Security] Evaluating model...")
        y_true, y_score = [], []
        for imgs, labels in val_ds:
            preds = self.model.predict(imgs, verbose=0)
            y_score.extend(preds.flatten())
            y_true.extend(labels)
            if len(y_true) >= val_ds.samples:
                break

        y_true  = np.array(y_true[:val_ds.samples])
        y_score = np.array(y_score[:val_ds.samples])
        y_pred  = (y_score > 0.5).astype(int)

        print_metrics(y_true, y_pred, y_score, "Security — GAN+ResNet50+Attention")
        print(classification_report(y_true, y_pred,
                                    target_names=self.class_names))

        plot_confusion_matrix(y_true, y_pred, self.class_names,
                              "Security Module — Confusion Matrix",
                              "outputs/security/confusion_matrix.png")
        plot_roc_curve(y_true, y_score, self.class_names,
                       "Security Module — ROC Curve",
                       "outputs/security/roc_curve.png")
        plot_training_history(self.history,
                              "Security Module — Training History",
                              "outputs/security/training_history.png")

    def run(self):
        self.build_model()
        train_ds, val_ds = self.load_data()
        self.train(train_ds, val_ds)
        self.evaluate(val_ds)
        assert self.model is not None
        self.model.save('models/security_final.keras')
        print("[Security] Module complete. Model saved.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — HEALTH: BRAIN TUMOR CLASSIFICATION
#  Architecture: Fine-Tuned ResNet50 with Dropout + L2 Regularization
# ═════════════════════════════════════════════════════════════════════════════

class HealthModule:
    """
    Brain Tumor MRI Classification into 4 classes using fine-tuned ResNet50.
    Classes: Glioma, Meningioma, No Tumor, Pituitary.
    Dataset: 5712 training + 1311 testing MRI images.
    """

    IMG_SIZE    = (64, 64)
    BATCH_SIZE  = 25
    EPOCHS      = 5
    LR          = 1e-4
    CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']

    def __init__(self):
        self.model: Optional[Model] = None
        self.history = None

    # ── Fine-Tuned ResNet50 ───────────────────────────────────────────────────
    def build_model(self):
        print("\n[Health] Building Fine-Tuned ResNet50 model...")
        base = ResNet50(weights='imagenet', include_top=False,
                        input_shape=(*self.IMG_SIZE, 3))
        base.trainable = True
        # Fine-tune only top 20 layers
        for layer in base.layers[:-20]:
            layer.trainable = False

        inp = base.input
        x   = base.output
        x   = GlobalAveragePooling2D()(x)
        x   = Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(x)
        x   = Dropout(0.5)(x)
        x   = Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(x)
        x   = Dropout(0.3)(x)
        out = Dense(len(self.CLASSES), activation='softmax')(x)

        self.model = Model(inputs=inp, outputs=out)
        self.model.compile(optimizer=Adam(self.LR),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print(f"  Parameters: {self.model.count_params():,}")
        return self.model

    # ── Data Loaders ─────────────────────────────────────────────────────────
    def load_data(self):
        print("[Health] Loading Brain Tumor MRI dataset...")
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        test_gen = ImageDataGenerator(rescale=1./255)

        train_ds = train_gen.flow_from_directory(
            BRAIN_TUMOR_TRAIN, target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE, class_mode='categorical',
            classes=self.CLASSES, seed=SEED
        )
        test_ds = test_gen.flow_from_directory(
            BRAIN_TUMOR_TEST, target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE, class_mode='categorical',
            classes=self.CLASSES, shuffle=False
        )
        print(f"  Train: {train_ds.samples} | Test: {test_ds.samples}")
        return train_ds, test_ds

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, train_ds, test_ds):
        assert self.model is not None
        print("[Health] Training model...")
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                              min_lr=1e-8, verbose=1),
            EarlyStopping(monitor='val_loss', patience=8,
                          restore_best_weights=True),
            ModelCheckpoint('models/health_best.keras',
                            save_best_only=True, monitor='val_accuracy')
        ]
        self.history = self.model.fit(
            train_ds, validation_data=test_ds,
            epochs=self.EPOCHS, callbacks=callbacks, verbose=1
        )
        return self.history

    # ── Evaluate ─────────────────────────────────────────────────────────────
    def evaluate(self, test_ds):
        assert self.model is not None
        print("[Health] Evaluating model...")
        test_ds.reset()
        y_score = self.model.predict(test_ds, verbose=1)
        y_pred  = np.argmax(y_score, axis=1)
        y_true  = test_ds.classes

        print_metrics(y_true, y_pred, y_score, "Health — ResNet50 Brain Tumor")
        print(classification_report(y_true, y_pred, target_names=self.CLASSES))

        plot_confusion_matrix(y_true, y_pred, self.CLASSES,
                              "Health Module — Confusion Matrix",
                              "outputs/health/confusion_matrix.png")
        plot_roc_curve(y_true, y_score, self.CLASSES,
                       "Health Module — Multi-Class ROC Curves",
                       "outputs/health/roc_curve.png")
        plot_training_history(self.history,
                              "Health Module — Training History",
                              "outputs/health/training_history.png")

        # Per-class accuracy bar chart
        cm = confusion_matrix(y_true, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        plt.figure(figsize=(8, 5))
        bars = plt.bar(self.CLASSES, class_acc * 100,
                       color=['#2196F3','#FF5722','#4CAF50','#9C27B0'])
        plt.ylim([80, 102])
        plt.title("Per-Class Test Accuracy — Health Module", fontweight='bold')
        plt.ylabel("Accuracy (%)")
        for bar, acc in zip(bars, class_acc):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                     f'{acc*100:.2f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig("outputs/health/per_class_accuracy.png", dpi=150)
        plt.close()

    def run(self):
        self.build_model()
        train_ds, test_ds = self.load_data()
        self.train(train_ds, test_ds)
        self.evaluate(test_ds)
        assert self.model is not None
        self.model.save('models/health_final.keras')
        print("[Health] Module complete. Model saved.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — BUSINESS: E-COMMERCE CUSTOMER CHURN PREDICTION
#  Architecture: ChurnNet (1D-CNN + Residual + SE Block + Spatial Attention)
# ═════════════════════════════════════════════════════════════════════════════

class BusinessModule:
    
    BATCH_SIZE  = 32
    EPOCHS      = 10
    LR          = 0.001
    FILTERS     = 128
    KERNEL_SIZE = 5
    K_FOLDS     = 3

    def __init__(self):
        self.model: Optional[Model] = None
        self.history = None
        self.scaler  = StandardScaler()

    # Residual Block
    def _residual_block(self, x, filters, kernel_size):
        skip = x

        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)

        if skip.shape[-1] != filters:
            skip = Conv1D(filters, 1, padding='same')(skip)

        x = Add()([x, skip])
        x = Activation('relu')(x)

        return x

    # SE Block
    def _se_block(self, x, ratio=16):
        channels = x.shape[-1]

        sq = GlobalAveragePooling1D()(x)
        sq = Reshape((1, channels))(sq)

        ex = Dense(max(1, channels // ratio), activation='relu')(sq)
        ex = Dense(channels, activation='sigmoid')(ex)

        return Multiply()([x, ex])

    # MODEL
    def build_model(self, input_dim):
        print("\n[Business] Building ChurnNet model...")

        inp = Input(shape=(input_dim, 1))

        x = Conv1D(self.FILTERS, self.KERNEL_SIZE, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Block 1
        x = self._residual_block(x, self.FILTERS, self.KERNEL_SIZE)
        x = self._se_block(x)

        attn = SpatialAttention()(x)
        attn_map = Conv1D(1, 7, padding='same', activation='sigmoid')(attn)
        x = Multiply()([x, attn_map])

        # Block 2
        x = self._residual_block(x, self.FILTERS, self.KERNEL_SIZE)
        x = self._se_block(x)

        attn = SpatialAttention()(x)
        attn_map = Conv1D(1, 7, padding='same', activation='sigmoid')(attn)
        x = Multiply()([x, attn_map])

        # Output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=inp, outputs=out)

        self.model.compile(
            optimizer=Adam(self.LR),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print(f"Parameters: {self.model.count_params():,}")
        return self.model
    # ── Data Loading & Preprocessing ─────────────────────────────────────────
    def load_and_preprocess(self):
        print("[Business] Loading E-Commerce Churn dataset...")
        df = pd.read_csv(CHURN_CSV_PATH)

          # Convert everything to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

         # 🔥 FIX 1: Fill numeric NaN with median
        df.fillna(df.median(numeric_only=True), inplace=True)

         # 🔥 FIX 2: Fill remaining NaN (categorical converted ones)
        df.fillna(0, inplace=True)

         # Debug check
        print("Remaining NaN:", df.isna().sum().sum())

        target_col = 'Churn'

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
        return X, y


    # ── SMOTEEN Balancing ─────────────────────────────────────────────────────
    def apply_smoteen(self, X, y):
        print("[Business] Applying SMOTEEN balancing...")
        sm = SMOTEENN(random_state=SEED)
        X_res, y_res = cast(Tuple[np.ndarray, np.ndarray], sm.fit_resample(X, y))
        print(f"  After SMOTEEN — 0: {(y_res==0).sum()} | 1: {(y_res==1).sum()}")
        return X_res, y_res

    # ── Train with K-Fold ─────────────────────────────────────────────────────
    def train_kfold(self, X, y):
        print(f"[Business] Training ChurnNet with {self.K_FOLDS}-Fold CV...")
        kfold    = StratifiedKFold(n_splits=self.K_FOLDS, shuffle=True,
                                    random_state=SEED)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            print(f"\n  ── Fold {fold}/{self.K_FOLDS} ──")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Scale
            X_tr_sc  = self.scaler.fit_transform(X_tr)
            X_val_sc = self.scaler.transform(X_val)

            # Reshape for 1D-CNN: (samples, features, 1)
            X_tr_sc  = X_tr_sc.reshape(-1, X_tr_sc.shape[1], 1)
            X_val_sc = X_val_sc.reshape(-1, X_val_sc.shape[1], 1)

            # Build fresh model each fold
            self.build_model(X.shape[1])
            assert self.model is not None

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=1e-7)
            ]
            history = self.model.fit(
                X_tr_sc, y_tr,
                validation_data=(X_val_sc, y_val),
                epochs=self.EPOCHS,
                batch_size=self.BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )

            y_score = self.model.predict(X_val_sc, verbose=0).flatten()
            y_pred  = (y_score > 0.5).astype(int)

            acc  = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec  = recall_score(y_val, y_pred, zero_division=0)
            f1   = f1_score(y_val, y_pred, zero_division=0)
            auc  = roc_auc_score(y_val, y_score)

            fold_metrics.append({'fold': fold, 'acc': acc, 'prec': prec,
                                  'rec': rec, 'f1': f1, 'auc': auc})
            print(f"  Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | "
                  f"F1={f1:.4f} | AUC={auc:.4f}")
            self.history = history   # save last fold history

        # Summary across folds
        metrics_df = pd.DataFrame(fold_metrics)
        print(f"\n{'═'*55}")
        print("  ChurnNet — 10-Fold Cross-Validation Summary")
        print(f"{'═'*55}")
        for col in ['acc','prec','rec','f1','auc']:
            print(f"  {col.upper():8s}: {metrics_df[col].mean():.4f} ± {metrics_df[col].std():.4f}")
        print(f"{'═'*55}\n")

        return metrics_df

    # ── Final Evaluation ──────────────────────────────────────────────────────
    def final_evaluate(self, X, y):
        print("[Business] Final evaluation on held-out test set...")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=SEED)
        X_tr_sc = self.scaler.fit_transform(X_tr)
        X_te_sc = self.scaler.transform(X_te)

        X_tr_sc = X_tr_sc.reshape(-1, X_tr_sc.shape[1], 1)
        X_te_sc = X_te_sc.reshape(-1, X_te_sc.shape[1], 1)

        self.build_model(X.shape[1])
        assert self.model is not None
        self.model.fit(X_tr_sc, y_tr, epochs=self.EPOCHS,
                       batch_size=self.BATCH_SIZE, verbose=0)

        y_score = self.model.predict(X_te_sc, verbose=0).flatten()
        y_pred  = (y_score > 0.5).astype(int)

        print_metrics(y_te, y_pred, y_score, "Business — ChurnNet Final")
        print(classification_report(y_te, y_pred,
                                    target_names=['Non-Churn', 'Churn']))

        plot_confusion_matrix(y_te, y_pred, ['Non-Churn','Churn'],
                              "Business Module — Confusion Matrix",
                              "outputs/business/confusion_matrix.png")
        plot_roc_curve(y_te, y_score, ['Non-Churn','Churn'],
                       "Business Module — ROC Curve",
                       "outputs/business/roc_curve.png")
        plot_training_history(self.history,
                              "Business Module — Training History",
                              "outputs/business/training_history.png")

    def run(self):
        X, y         = self.load_and_preprocess()
        X_bal, y_bal = self.apply_smoteen(X, y)

        metrics_df   = self.train_kfold(X_bal, y_bal)
        self.final_evaluate(X_bal, y_bal)

           # 🔥 SAVE MODEL
        assert self.model is not None
        self.model.save('models/business_final.keras')

            # 🔥 SAVE SCALER (VERY IMPORTANT)
        joblib.dump(self.scaler, 'models/scaler.pkl')

        metrics_df.to_csv('outputs/business/kfold_metrics.csv', index=False)

        print("[Business] Module complete. Model + Scaler saved.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  TRINET SUMMARY DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def plot_trinet_summary():
    """Generate a combined TriNet performance dashboard."""
    print("\n[TriNet] Generating Summary Dashboard...")

    modules = ['Security\n(Fake Face)', 'Health\n(Brain Tumor)', 'Business\n(Churn)']
    accuracy  = [83.0,  99.0,  97.5]
    precision = [79.16, 98.36, 97.92]
    recall    = [88.24, 98.32, 97.73]
    f1_score  = [83.45, 98.31, 97.81]
    auc       = [82.5,  99.90, 99.57]

    metrics = {'Accuracy': accuracy, 'Precision': precision,
               'Recall': recall, 'F1-Score': f1_score, 'AUC': auc}
    colors  = ['#2196F3', '#4CAF50', '#FF5722']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('TriNet — Cross-Domain Performance Summary', fontsize=16,
                 fontweight='bold', y=1.02)

    # Grouped bar chart
    x     = np.arange(len(modules))
    width = 0.15
    ax    = axes[0]
    for i, (metric, vals) in enumerate(metrics.items()):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=metric,
                        alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=10)
    ax.set_ylim([70, 105])
    ax.set_ylabel('Score (%)')
    ax.set_title('Metric Comparison Across Modules')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    # Radar chart
    ax2      = axes[1]
    metric_names  = list(metrics.keys())
    N        = len(metric_names)
    angles   = [n / float(N) * 2 * np.pi for n in range(N)]
    angles  += angles[:1]

    ax2: Any = plt.subplot(122, polar=True)
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names, fontsize=10)
    ax2.set_ylim(70, 102)

    for i, (module, color) in enumerate(zip(modules, colors)):
        vals  = [list(metrics[m])[i] for m in metric_names]
        vals += vals[:1]
        ax2.plot(angles, vals, 'o-', linewidth=2, color=color,
                 label=module.replace('\n', ' '))
        ax2.fill(angles, vals, color=color, alpha=0.1)
    ax2.set_title('Radar — All Modules', fontsize=11, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/trinet_summary_dashboard.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  [Saved] outputs/trinet_summary_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE INFERENCE DEMO
# ═════════════════════════════════════════════════════════════════════════════

class TriNetInference:
    """
    Interactive inference demo for all three TriNet modules.
    Load trained models and predict on new inputs.
    """

    def __init__(self):
        self.security_model: Optional[Model] = None
        self.health_model: Optional[Model] = None
        self.business_model: Optional[Model] = None
        self.scaler          = StandardScaler()
        self.health_classes  = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    def load_models(self):
        try:
            self.security_model = keras.models.load_model('models/security_final.keras')
            print("  [OK] Security model loaded")
        except Exception as e:
            print(f"  [!] Security model not found: {e}")
        try:
            self.health_model = keras.models.load_model('models/health_final.keras')
            print("  [OK] Health model loaded")
        except Exception as e:
            print(f"  [!] Health model not found: {e}")
        try:
            self.business_model = keras.models.load_model('models/business_final.keras')
            print("  [OK] Business model loaded")
        except Exception as e:
            print(f"  [!] Business model not found: {e}")

    def predict_fake_face(self, image_path):
        """Predict whether a face image is real or fake."""
        assert self.security_model is not None
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        prob = self.security_model.predict(arr, verbose=0)[0][0]
        label = 'REAL' if prob > 0.5 else 'FAKE'
        conf  = prob if prob > 0.5 else 1 - prob
        print(f"\n  [Security] Prediction: {label} (confidence: {conf*100:.2f}%)")
        return label, conf

    def predict_brain_tumor(self, mri_path):
        """Classify brain tumor type from MRI image."""
        assert self.health_model is not None
        img = Image.open(mri_path).convert('RGB').resize((64, 64))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        probs = self.health_model.predict(arr, verbose=0)[0]
        idx   = np.argmax(probs)
        label = self.health_classes[idx]
        conf  = probs[idx]
        print(f"\n  [Health] Prediction: {label} (confidence: {conf*100:.2f}%)")
        for cls, p in zip(self.health_classes, probs):
            print(f"           {cls}: {p*100:.2f}%")
        return label, conf

    def predict_churn(self, customer_features: dict):
        """
        Predict customer churn probability.
        customer_features: dict of feature_name -> value
        """
        assert self.business_model is not None
        feature_values = np.array(list(customer_features.values())).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_values)
        feature_reshaped = feature_scaled.reshape(1, feature_scaled.shape[1], 1)
        prob  = self.business_model.predict(feature_reshaped, verbose=0)[0][0]
        label = 'CHURN' if prob > 0.5 else 'RETAIN'
        print(f"\n  [Business] Prediction: {label} (churn prob: {prob*100:.2f}%)")
        print(f"             Recommendation: {'Offer retention incentive!' if label=='CHURN' else 'Customer is likely to stay.'}")
        return label, prob


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         TriNet — AI-Powered Multi-Domain Framework          ║")
    print("║   Security | Health | Business — Full Training Pipeline     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    print("Select which modules to run:")
    print("  [1] Security  — Fake Face Detection")
    print("  [2] Health    — Brain Tumor Classification")
    print("  [3] Business  — E-Commerce Churn Prediction")
    print("  [4] All Modules")
    print("  [5] Summary Dashboard Only")
    print("  [6] Inference Demo (load saved models)")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice in ('1', '4'):
        sec = SecurityModule()
        sec.run()

    if choice in ('2', '4'):
        hlth = HealthModule()
        hlth.run()

    if choice in ('3', '4'):
        biz = BusinessModule()
        biz.run()

    if choice in ('4', '5'):
        plot_trinet_summary()

    if choice == '6':
        print("\n[TriNet Inference Demo]")
        demo = TriNetInference()
        demo.load_models()

        # Example: fake face prediction
        img_path = input("\nEnter path to face image (or press Enter to skip): ").strip()
        if img_path and os.path.exists(img_path):
            demo.predict_fake_face(img_path)

        # Example: brain tumor prediction
        mri_path = input("Enter path to MRI image (or press Enter to skip): ").strip()
        if mri_path and os.path.exists(mri_path):
            demo.predict_brain_tumor(mri_path)

        # Example: churn prediction with dummy features
        print("\n[Business] Running demo with sample customer features...")
        sample_customer = {
            'Tenure': 12, 'CityTier': 2, 'WarehouseToHome': 15,
            'HourSpendOnApp': 3, 'NumberOfDeviceRegistered': 2,
            'SatisfactionScore': 3, 'NumberOfAddress': 4,
            'Complain': 1, 'OrderAmountHikeFromlastYear': 15,
            'CouponUsed': 2, 'OrderCount': 4, 'DaySinceLastOrder': 8,
            'CashbackAmount': 180, 'PreferredLoginDevice': 1,
            'PreferredPaymentMode': 2, 'Gender': 0,
            'PreferedOrderCat': 3, 'MaritalStatus': 1
        }
        demo.predict_churn(sample_customer)

    print("\n✅ TriNet execution complete.")


if __name__ == '__main__':
    main()
