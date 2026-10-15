import sys, json
from pathlib import Path
root = Path.cwd().parent
sys.path.append(str(root))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


from src.models.cnn import build_model


def train(configs, X_tr, y_tr, X_val, y_val, EPOCHS):
    results = []

    for cfg in configs:
        print("Training:", cfg["name"])

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        model = build_model(cfg["learning_rate"], cfg["dropout"], cfg["dense_units"])

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=cfg["batch_size"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            verbose=0,
        )

        best_val_acc = max(history.history["val_accuracy"])
        print(f"   validation accuracy: {best_val_acc:.4f}")
        results.append({**cfg, "val_accuracy": best_val_acc})

    return results