import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
import h5py
import cv2

DATA_DIR = "/workspace/data"

print("Cargando dataset")

X, Y = [], []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".mat"):
        continue

    path = os.path.join(DATA_DIR, file)
    with h5py.File(path, "r") as f:
        img = np.array(f["cjdata"]["image"]).astype("float32")
        label = int(f["cjdata"]["label"][0][0]) - 1   # 1–3 → 0–2

    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, None)

    if img.max() > 0:
        img = img / img.max()

    img = cv2.resize(img, (128, 128))

    X.append(img)
    Y.append(label)

X = np.array(X).reshape(-1, 128, 128, 1)
Y = np.array(Y)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print("Dataset cargado. Tamaño:", X.shape)

def build_model(hp):
    model = models.Sequential()

    # filtros variables
    model.add(layers.Conv2D(
        filters=hp.Choice("filters1", [32, 48, 64]),
        kernel_size=3,
        activation="relu",
        input_shape=(128, 128, 1)
    ))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(
        filters=hp.Choice("filters2", [64, 96, 128]),
        kernel_size=3,
        activation="relu"
    ))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())

    model.add(layers.Dense(
        hp.Choice("dense_units", [64, 128, 256]),
        activation="relu"
    ))

    model.add(layers.Dropout(hp.Choice("dropout", [0.3, 0.4, 0.5])))

    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("lr", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# HYPERBAND TUNER
tuner = Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=6,                # límite de epochs
    factor=3,
    directory="tuner_mid_results",
    project_name="mri_fast_tuning"
)

# Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

print("Iniciando búsqueda de hiperparámetros")

tuner.search(
    X_train, y_train,
    epochs=6,
    validation_data=(X_val, y_val),
    callbacks=[stop_early]
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nMejores hiperparámetros encontrados")
print(best_hp.values)

# Guardar
with open("/workspace/mejores_hiperparametros.txt", "w") as f:
    f.write(str(best_hp.values))

print("Guardado en /workspace/mejores_hiperparametros.txt")
