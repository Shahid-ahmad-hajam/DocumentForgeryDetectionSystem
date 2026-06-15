"""
report_accuracy.py
Extract and print:
 - Final Train Accuracy
 - Best Validation Accuracy
 - Test Accuracy (recommended overall metric)
"""

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import tensorflow as tf

HISTORY_FILE = "outputs/finetune_docs_history.npz"
TEST_FILE = "outputs/test_results.npz"
MODEL_FILE = "outputs/ela_efficientnet_final_finetuned.h5"  # change if needed


def load_history():
    print("\n=== LOADING TRAINING HISTORY ===")
    h = np.load(HISTORY_FILE)
    train_acc = h["accuracy"]
    val_acc = h["val_accuracy"]

    final_train = float(train_acc[-1])
    best_val = float(val_acc.max())
    final_val = float(val_acc[-1])

    print(f"Final Train Accuracy       : {final_train:.4f}")
    print(f"Final Validation Accuracy  : {final_val:.4f}")
    print(f"Best Validation Accuracy   : {best_val:.4f}")

    return final_train, final_val, best_val


def load_test_results():
    print("\n=== LOADING TEST RESULTS ===")
    data = np.load(TEST_FILE)
    y_test = data["y_test"]
    y_pred = data["y_pred"]

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy              : {test_acc:.4f}")

    return test_acc


def evaluate_model_directly():
    """
    Optional:
    Evaluate the saved final model directly on X_test in case future runs need recalculation.
    """
    data = np.load(TEST_FILE)
    X_test = data["X_test"]
    y_test = data["y_test"]

    model = load_model(MODEL_FILE)
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    Xp = preprocess(X_test.copy())
    loss, acc, auc = model.evaluate(Xp, y_test, verbose=0)

    print("\nModel Evaluate() Results:")
    print(f"  Accuracy (Evaluate): {acc:.4f}")
    print(f"  AUC                : {auc:.4f}")


if __name__ == "__main__":
    print("\n====================================")
    print("       MODEL ACCURACY REPORT")
    print("====================================")

    final_train, final_val, best_val = load_history()
    test_acc = load_test_results()

    print("\n============ SUMMARY ============")
    print(f"Training Accuracy     : {final_train:.4f}")
    print(f"Validation Accuracy   : {final_val:.4f}  (best = {best_val:.4f})")
    print(f"Test Accuracy         : {test_acc:.4f}")
    print("=================================\n")
