import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# --- Paths ---
data_dir = "C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\data"
#jpg_tumor_dir = "/home/leonel/Documents/learning/sexto/sistemas/proyecto/archive/training"
jpg_tumor_dir = "C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\archive\\Training"
# files inside jpg_tumor_dir
tumor_file_names = ["glioma", "meningioma", "notumor", "pituitary"]

# --- Label dictionary ---
tumor_types = {0: "notumor", 1: "meningioma", 2: "glioma", 3: "pituitary"}

X, Y = [], []

# --- Load tumor data ---
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".mat"):
            continue

        filepath = os.path.join(folder_path, file)
        try:
            with h5py.File(filepath, 'r') as f:
                image = np.array(f['cjdata']['image']).astype(np.float32)
                label = int(f['cjdata']['label'][0][0])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        image = cv2.resize(image, (128, 128))
        image = image / 255.0

        X.append(image)
        Y.append(label)

        print(f"Loaded tumor image {file} -> label {label}")

# --- Load only half of no-tumor data ---

for i in tumor_file_names:

    divisor = 1 if i == "notumor" else 2

    folder_path = os.path.join(jpg_tumor_dir, i)
    jpg_tumor_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    random.shuffle(jpg_tumor_files)
    half_count = len(jpg_tumor_files) // divisor
    jpg_tumor_files = jpg_tumor_files[:half_count]
    print(f"Explorando carpeta: {folder_path} con {len(jpg_tumor_files)} archivos seleccionados")

    for file in jpg_tumor_files:
        filepath = os.path.join(folder_path, file)

        try:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Invalid image file")

            image = cv2.resize(image, (128, 128))
            image = image.astype(np.float32) / 255.0

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        X.append(image)
        Y.append(tumor_file_names.index(i))  # integer label


# --- Finalize data ---
X = np.array(X).reshape(-1, 128, 128, 1)
Y = np.array(Y)

print("Data loaded successfully")
print(f"Shapes: X={X.shape}, Y={Y.shape}")

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# --- Build CNN classifier ---
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(4, activation='softmax')  # 4 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# --- Train ---
history = model.fit(
    X_train, y_train,
    epochs=10, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks

)

# --- Save model ---
os.makedirs("C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\models", exist_ok=True)
model.save("C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\models\\tumor_classifier_4class_half_no_tumor.h5")

# --- Evaluate ---
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=list(tumor_types.values())))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=tumor_types.values(),
            yticklabels=tumor_types.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - 4 Class (Half No-Tumor)")
plt.show()


# --- Curvas de entrenamiento ---
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Curva de Accuracy')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Curva de Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
