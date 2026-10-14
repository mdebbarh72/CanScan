import sys 
from pathlib import Path

root =  Path.cwd().parent
sys.path.append(str(root))

from src.data.load import ImageLoader
from src.data.visualize import Painter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib




class Processor:

    @classmethod
    def shuffleData(cls, images, labels):
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(images))

        # Apply the same indexes to both arrays
        images = images[indices]
        labels = labels[indices]

        return (images, labels)

    @classmethod
    def encodeLabels(cls, labels):
        encoder = LabelEncoder()

        labels_encoded = encoder.fit_transform(labels)

        print("Classes:", encoder.classes_)
        print("Encoded labels:", labels_encoded[:10])

        return encoder, labels_encoded

    @classmethod
    def encodeTarget(cls, encoder, y_train, y_test):
        num_classes = len(encoder.classes_)

        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat  = to_categorical(y_test, num_classes)

        print("y_train_cat:", y_train_cat.shape)   # (4916, 4)
        print("y_test_cat:", y_test_cat.shape)     # (2107, 4)
        print("Example:", y_train[0], "->", y_train_cat[0])

        return (y_train_cat, y_test_cat)

    @classmethod
    def scaleData(cls, x_train, x_test):
        return ( 
            x_train.astype("float32") / 255.0, 
            x_test.astype("float32") / 255.0
        )

    @classmethod
    def saveData(cls, encoder, X_train, y_train, X_test, y_test, y_train_cat, y_test_cat):
        np.savez_compressed(
        root / "data" / "processed" / "processed.npz",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_train_cat=y_train_cat, y_test_cat=y_test_cat,
)
        joblib.dump(encoder, root / "models" / "label_encoder.pkl")