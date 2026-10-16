from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def conv_block(model, filters):
    for _ in range(2):
        model.add(layers.Conv2D(filters, 3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())


def build_model(num_classes=4):
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        # Light augmentation (training only)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
    ])

    for filters in [32, 64, 128, 256, 256]:
        conv_block(model, filters)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model