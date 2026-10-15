import sys, time, random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


def load():
    root = Path.cwd().parent
    data = np.load(root / "data" / "processed" / "processed.npz")

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    y_train_cat, y_test_cat = data["y_train_cat"], data["y_test_cat"]

    encoder = joblib.load(root / "models" / "label_encoder.pkl")
    class_names = list(encoder.classes_)
    num_classes = len(class_names)

    print("X_train:", X_train.shape, "| y_train_cat:", y_train_cat.shape)
    print("X_test :", X_test.shape,  "| y_test_cat :", y_test_cat.shape)
    print("Classes:", class_names)
    print("Pixel range:", X_train.min(), X_train.max())

    return (
        X_train, X_test, 
        y_train, y_test,
        y_train_cat, y_test_cat,
        encoder, class_names,
    )


def sanityCheck(X_train, y_train_cat, y_train, y_test):
    assert X_train.max() <= 1.0 and X_train.min() >= 0.0, "Pixels are not normalized to [0, 1]"
    assert np.allclose(y_train_cat.sum(axis=1), 1), "y_train_cat is not one-hot"
    assert (np.argmax(y_train_cat, axis=1) == y_train).all(), "Integer and one-hot labels disagree"

    print("Train class counts:", np.bincount(y_train))
    print("Test class counts :", np.bincount(y_test))
    print("All checks passed")

def calculateWeights(y_tr_int):
    weights = compute_class_weight("balanced", classes=np.unique(y_tr_int), y=y_tr_int)
    class_weight = {i: w for i, w in enumerate(weights)}
    print("Class weights:", class_weight)

    return class_weight


