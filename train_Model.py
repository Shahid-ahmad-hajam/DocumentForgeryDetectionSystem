# ONLY TO GET THE RESULTS

"""
train_forgery_model_ready_augmented_pretrain_finetune.py

Two-phase training:
  1) Optional pretrain on CASIA (general image forgery) to learn universal tamper cues
  2) Fine-tune on your document dataset (document-specific adaptation)

Features:
 - ELA preprocessing
 - Data sanity checks (duplicates / missing folders)
 - Pretrained backbone (EfficientNetB0) used as feature extractor
 - L2 regularization, Dropout, BatchNorm
 - Data augmentation
 - Class weighting
 - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 - Saves accuracy & loss plots for each phase and final report
 - Saves training histories (.npz), ROC/PR curves, confusion matrix plot, hyperparameters and model summary
"""

import os
import hashlib
from io import BytesIO
from collections import Counter
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

# seaborn used only for confusion matrix heatmap
import seaborn as sns

# ----------------------------
# CONFIGURATION
# ----------------------------
DATASET_DIR = "dataset"            # folder containing 'real' and 'fake' (your document dataset)
CASIA_DIR = "casia2"               # set to path where CASIA v2 is extracted (Au/ and Tp/). If missing -> skip pretrain
OUTPUT_DIR = "outputs"
ELA_QUALITY = 90
SCALE = 15
IMAGE_SIZE = (224, 224)            # (width, height) for PIL; EfficientNet expects 224x224
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 12
FINETUNE_EPOCHS = 25
MODEL_NAME_PREFIX = "ela_efficientnet"
RANDOM_STATE = 42
L2_REG = 1e-4
PRETRAIN_LEARNING_RATE = 1e-4
FINETUNE_LEARNING_RATE = 1e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Optional: allow GPU memory growth (non-intrusive)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ----------------------------
# HELPERS: hashing + checks
# ----------------------------
def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def data_checks(dataset_dir, folder_map=None):
    """
    Count files and check duplicates.
    folder_map (optional) maps logical labels to actual folder names, e.g.
      {'real':'real','fake':'fake'}  OR  {'real':'Au','fake':'Tp'}
    If folder_map is None it will try to auto-detect common layouts.
    Returns: counts dict, dupes dict
    """
    # Auto-detect if not provided
    if folder_map is None:
        if os.path.isdir(os.path.join(dataset_dir, 'real')) and os.path.isdir(os.path.join(dataset_dir, 'fake')):
            folder_map = {'real': 'real', 'fake': 'fake'}
        elif os.path.isdir(os.path.join(dataset_dir, 'Au')) and os.path.isdir(os.path.join(dataset_dir, 'Tp')):
            folder_map = {'real': 'Au', 'fake': 'Tp'}
        else:
            raise ValueError(f"Could not auto-detect dataset layout in {dataset_dir}. "
                             "Expected 'real'/'fake' or 'Au'/'Tp' or provide folder_map explicitly.")

    print("[DATA CHECK] Using folder_map:", folder_map)
    counts = {}
    hashes = {}

    for lbl, folder_name in folder_map.items():
        folder = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder):
            raise ValueError(f"Missing folder: {folder} (tried mapping label '{lbl}')")

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[lbl] = len(files)
        print(f"  {lbl} ({folder_name}): {counts[lbl]} images")

        for f in files:
            p = os.path.join(folder, f)
            try:
                h = file_hash(p)
            except Exception as e:
                print(f"  Could not hash {p}: {e}")
                continue
            hashes.setdefault(h, []).append(p)

    dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    print(f"[DATA CHECK] Duplicate file groups found: {len(dupes)} (exact binary duplicates)")
    if len(dupes) > 0:
        print("  Example duplicates (up to 10 groups):")
        for i, (h, ps) in enumerate(dupes.items()):
            print("   ", ps)
            if i >= 9:
                break
    return counts, dupes

# ----------------------------
# ELA conversion
# ----------------------------
def convert_to_ela_image(image_path, quality=ELA_QUALITY, scale=SCALE):
    image = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert('RGB')
    ela_image = ImageChops.difference(image, resaved)

    def scale_channel(c):
        return c.point(lambda i: min(255, int(i * scale)))

    if ela_image.mode == 'RGB':
        r, g, b = ela_image.split()
        r, g, b = scale_channel(r), scale_channel(g), scale_channel(b)
        ela_image = Image.merge('RGB', (r, g, b))
    else:
        ela_image = scale_channel(ela_image)
    return ela_image

# ----------------------------
# Dataset loader (applies ELA)
# ----------------------------
def load_dataset_ela(dataset_dir, image_size=IMAGE_SIZE, limit=None, verbose=True, folder_map=None):
    """
    Load images applying ELA. Accepts either:
      - default folder_map {'real':'real','fake':'fake'}
      - CASIA style {'real':'Au','fake':'Tp'}
      - or auto-detects based on what's present in dataset_dir.
    Returns: X (np.array), y (np.array)
    """
    X, y = [], []
    counts = {'real': 0, 'fake': 0}

    # If an explicit map provided, use it
    if folder_map is None:
        # auto-detect common layouts
        if os.path.isdir(os.path.join(dataset_dir, 'real')) and os.path.isdir(os.path.join(dataset_dir, 'fake')):
            folder_map = {'real': 'real', 'fake': 'fake'}
        elif os.path.isdir(os.path.join(dataset_dir, 'Au')) and os.path.isdir(os.path.join(dataset_dir, 'Tp')):
            folder_map = {'real': 'Au', 'fake': 'Tp'}
        else:
            # fallback to expecting 'real'/'fake' (will raise below)
            folder_map = {'real': 'real', 'fake': 'fake'}

    for label, folder_name in folder_map.items():
        folder = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder):
            raise ValueError(f"Missing folder: {folder} (tried mapping label '{label}')")
        if verbose:
            print(f"[LOAD] Processing {label} images from {folder} ...")
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if limit:
            files = files[:limit]
        for fname in tqdm(files):
            path = os.path.join(folder, fname)
            try:
                ela_img = convert_to_ela_image(path)
                ela_img = ela_img.resize(image_size)
                arr = np.array(ela_img).astype(np.float32)
                # Ensure shape is (H, W, 3)
                if arr.ndim == 2:
                    # convert grayscale to 3-channel
                    arr = np.stack([arr]*3, axis=-1)
                if arr.shape[:2] != (image_size[1], image_size[0]):
                    arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(image_size)).astype(np.float32)
                # store
                X.append(arr)
                # label mapping: real -> 1, fake -> 0
                label_val = 1 if label == 'real' else 0
                y.append(label_val)
                counts[label] += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    if verbose:
        print(f"[LOAD] Loaded {len(X)} images. (real: {counts['real']}, fake: {counts['fake']})")
    return X, y

# ----------------------------
# Model builder (EfficientNetB0 backbone)
# ----------------------------
def build_model(input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), l2_reg=L2_REG, base_trainable=False):
    # EfficientNetB0 backbone
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base.trainable = base_trainable  # False for pretrain freeze, True to fine-tune later if needed

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg))(x)

    model = Model(inputs=base.input, outputs=out)
    return model

# ----------------------------
# Plotting utility
# ----------------------------
def save_plots(history, prefix):
    acc_path = os.path.join(OUTPUT_DIR, f'{prefix}_accuracy_plot.png')
    loss_path = os.path.join(OUTPUT_DIR, f'{prefix}_loss_plot.png')

    plt.figure(figsize=(8,5))
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
    plt.title(f'{prefix} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title(f'{prefix} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

# ----------------------------
# EXTRA: Save training history arrays
# ----------------------------
def save_history_npz(history, prefix):
    out_path = os.path.join(OUTPUT_DIR, f"{prefix}_history.npz")
    # history.history is a dict of lists/arrays
    np.savez(out_path, **history.history)
    print(f"[SAVE] Training history saved to {out_path}")

# ----------------------------
# EXTRA: Confusion Matrix Plot
# ----------------------------
def save_confusion_matrix(cm, prefix, class_names=['Fake','Real']):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{prefix} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVE] Confusion matrix plot saved to {out_path}")

# ----------------------------
# EXTRA: ROC and PR Curves
# ----------------------------
def save_roc_pr_curves(y_true, pred_probs, prefix):
    try:
        fpr, tpr, _ = roc_curve(y_true, pred_probs)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_true, pred_probs):.4f})")
        plt.plot([0,1], [0,1], '--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix} ROC Curve")
        plt.grid(True)
        roc_path = os.path.join(OUTPUT_DIR, f"{prefix}_ROC.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save ROC curve: {e}")
        roc_path = None

    try:
        precision, recall, _ = precision_recall_curve(y_true, pred_probs)
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{prefix} Precision–Recall Curve")
        plt.grid(True)
        pr_path = os.path.join(OUTPUT_DIR, f"{prefix}_PR.png")
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save PR curve: {e}")
        pr_path = None

    print(f"[SAVE] ROC saved → {roc_path}")
    print(f"[SAVE] PR curve saved → {pr_path}")

# ----------------------------
# Training helper (numpy arrays)
# ----------------------------
def train_phase(model, X_train, y_train, X_val, y_val, epochs, lr, prefix, class_weight=None):
    # preprocess for EfficientNet: expects pixels in -1..1 if using preprocess_input
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.03,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_STATE)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    # compile
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        ModelCheckpoint(os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_{prefix}.h5'),
                        monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    steps = max(1, len(X_train) // BATCH_SIZE)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    save_plots(history, prefix)
    save_history_npz(history, prefix)
    return history

# ----------------------------
# EVALUATION helper
# ----------------------------
def evaluate_model(model, X, y, prefix):
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    Xp = preprocess(X.copy())
    preds = model.predict(Xp, batch_size=BATCH_SIZE).ravel()
    y_pred = (preds > 0.5).astype(int)
    auc = roc_auc_score(y, preds) if len(np.unique(y)) > 1 else float('nan')
    report = classification_report(y, y_pred, target_names=['Fake','Real'])
    cm = confusion_matrix(y, y_pred)
    print(f"\n[{prefix}] AUC: {auc:.4f}")
    print(report)
    print("Confusion matrix:\n", cm)
    return preds, y_pred, auc, report, cm

# ----------------------------
# MAIN
# ----------------------------
def main():
    start_time = time.time()
    # Data checks for your document dataset
    counts, dupes = data_checks(DATASET_DIR)
    if counts.get('real',0) == 0 or counts.get('fake',0) == 0:
        raise SystemExit("Dataset folders missing or empty. Ensure dataset/real and dataset/fake exist with images.")

    # LOAD your document dataset (ELA)
    print("[STEP] Loading document dataset (ELA)...")
    X_doc, y_doc = load_dataset_ela(DATASET_DIR, image_size=IMAGE_SIZE, verbose=True)

    # Split doc dataset into train/val/test (these are used for finetune and final eval)
    X_train_doc, X_temp_doc, y_train_doc, y_temp_doc = train_test_split(
        X_doc, y_doc, test_size=0.20, random_state=RANDOM_STATE, stratify=y_doc
    )
    X_val_doc, X_test_doc, y_val_doc, y_test_doc = train_test_split(
        X_temp_doc, y_temp_doc, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp_doc
    )
    print(f"[DOC SPLIT] Train: {len(X_train_doc)} | Val: {len(X_val_doc)} | Test: {len(X_test_doc)}")
    print("Train class counts:", Counter(y_train_doc))
    print("Val class counts:", Counter(y_val_doc))

    # Pretrain on CASIA if available
    pretrained_model_path = None
    if os.path.isdir(CASIA_DIR):
        # check CASIA structure
        casia_au = os.path.join(CASIA_DIR, 'Au')
        casia_tp = os.path.join(CASIA_DIR, 'Tp')
        if os.path.isdir(casia_au) and os.path.isdir(casia_tp):
            print("[STEP] CASIA found — starting pretraining phase (CASIA -> learn universal tamper cues)")
            X_casia, y_casia = load_dataset_ela(CASIA_DIR, image_size=IMAGE_SIZE, verbose=True)
            # split casia into train/val
            X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
                X_casia, y_casia, test_size=0.20, random_state=RANDOM_STATE, stratify=y_casia
            )
            print(f"[CASIA SPLIT] Train: {len(X_train_c)} | Val: {len(X_val_c)}")
            # Build model with frozen backbone
            model = build_model(input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), l2_reg=L2_REG, base_trainable=False)
            # class weights computed on CASIA train
            cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_c), y=y_train_c)
            class_weight = {int(c): float(w) for c,w in zip(np.unique(y_train_c), cw)}
            print("[CASIA] class weights:", class_weight)
            history_pre = train_phase(model, X_train_c, y_train_c, X_val_c, y_val_c,
                                      epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LEARNING_RATE,
                                      prefix='pretrain_casia', class_weight=class_weight)
            pretrained_model_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_pretrain_casia.h5')
            # ModelCheckpoint saved best weights under outputs, we copy/ensure the path variable
            print(f"[PRETRAIN] Pretraining phase finished. Best weights under outputs with prefix pretrain_casia.")
        else:
            print("[INFO] CASIA directory found but missing 'Au' or 'Tp' subfolders. Skipping pretrain.")
    else:
        print("[INFO] CASIA dir not found. Skipping pretrain phase.")

    # FINETUNE on your document data
    print("[STEP] Fine-tuning on document dataset...")
    # Build fresh model: load weights from pretrained model if exists
    model_ft = build_model(input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), l2_reg=L2_REG, base_trainable=False)

    # if pretrained saved model exists in outputs, load weights
    pretrain_candidate = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_pretrain_casia.h5')
    if os.path.exists(pretrain_candidate):
        try:
            model_ft.load_weights(pretrain_candidate)
            print("[FINETUNE] Loaded pretrained weights from:", pretrain_candidate)
        except Exception as e:
            print("Could not load pretrained weights:", e)

    # Optionally unfreeze some of the base layers for finetuning (fine-grained adaptation)
    # We'll unfreeze all layers (you may choose to unfreeze only some layers in practice)
    for layer in model_ft.layers:
        layer.trainable = True  # allow fine-tuning for better adaptation (we use low LR)

    # compile and prepare class weights
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_doc), y=y_train_doc)
    class_weight_doc = {int(c): float(w) for c,w in zip(np.unique(y_train_doc), cw)}
    print("[DOC] class weights:", class_weight_doc)

    history_ft = train_phase(model_ft, X_train_doc, y_train_doc, X_val_doc, y_val_doc,
                             epochs=FINETUNE_EPOCHS, lr=FINETUNE_LEARNING_RATE,
                             prefix='finetune_docs', class_weight=class_weight_doc)

    # Final evaluation on held-out test set
    print("[EVALUATION] Evaluating on document test set...")
    preds, y_pred, auc, report, cm = evaluate_model(model_ft, X_test_doc, y_test_doc, prefix='DOCUMENT_TEST')

    # Save final artifacts
    final_model_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_final_finetuned.h5')
    model_ft.save(final_model_path)
    print("[SAVE] Final model saved to:", final_model_path)

    # Save summary report
    with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w') as f:
        f.write(f'Final Test AUC: {auc:.4f}\n\n')
        f.write('Classification Report:\n')
        f.write(report + '\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(cm) + '\n')
    print("[SAVE] Text report saved to report.txt")

    # Save predictions, test arrays (X_test may be large; consider removing X_test if not needed)
    np.savez(os.path.join(OUTPUT_DIR, 'test_results.npz'),
             X_test=X_test_doc, y_test=y_test_doc, y_pred=y_pred, y_pred_probs=preds)
    print("[SAVE] test_results.npz saved")

    # Save confusion matrix figure
    save_confusion_matrix(cm, prefix="DOCUMENT_TEST")

    # Save ROC & PR curves
    save_roc_pr_curves(y_test_doc, preds, prefix="DOCUMENT_TEST")

    # Save hyperparameters and training setup
    with open(os.path.join(OUTPUT_DIR, 'hyperparameters.txt'), 'w') as f:
        f.write("IMAGE_SIZE: {}\n".format(IMAGE_SIZE))
        f.write("BATCH_SIZE: {}\n".format(BATCH_SIZE))
        f.write("PRETRAIN_EPOCHS: {}\n".format(PRETRAIN_EPOCHS))
        f.write("FINETUNE_EPOCHS: {}\n".format(FINETUNE_EPOCHS))
        f.write("L2_REG: {}\n".format(L2_REG))
        f.write("PRETRAIN_LEARNING_RATE: {}\n".format(PRETRAIN_LEARNING_RATE))
        f.write("FINETUNE_LEARNING_RATE: {}\n".format(FINETUNE_LEARNING_RATE))
        f.write("CLASS_WEIGHTS: {}\n".format(class_weight_doc))
        f.write("RANDOM_STATE: {}\n".format(RANDOM_STATE))
    print("[SAVE] Hyperparameters saved to hyperparameters.txt")

    # Save model summary
    with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
        model_ft.summary(print_fn=lambda x: f.write(x + "\n"))
    print("[SAVE] Model summary saved to model_summary.txt")

    elapsed = time.time() - start_time
    print(f"\nAll outputs saved to folder: {OUTPUT_DIR}")
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")
    print("Files include plots, models, report.txt, test_results.npz, histories (.npz), ROC/PR plots, confusion matrix image, hyperparameters and model summary.")
    print("Done.")

if __name__ == "__main__":
    main()
