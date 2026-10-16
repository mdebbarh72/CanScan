import io
import os

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "models/brain_tumor_cnn.keras")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "models/class_names.pkl")
IMAGE_SIZE = (224, 224)

model = None
class_names = None


def loadModel():
    global model, class_names
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = joblib.load(CLASS_NAMES_PATH)
    return model


def preprocessImage(imageBytes: bytes):
    # Same steps as in training: RGB, 224x224, pixels scaled to [0, 1]
    image = Image.open(io.BytesIO(imageBytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)  # shape (1, 224, 224, 3)


def predictTumor(imageBytes: bytes):
    if model is None:
        raise RuntimeError("Model has not been loaded")

    batch = preprocessImage(imageBytes)
    probs = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probs))

    return {
        "prediction": class_names[best],
        "confidence": float(probs[best]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }