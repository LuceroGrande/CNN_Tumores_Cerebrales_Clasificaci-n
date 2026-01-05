import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from itertools import cycle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

data_dir = "C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\data"
jpg_tumor_dir = "C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\archive\\Training"

tumor_file_names = ["glioma", "meningioma", "notumor", "pituitary"]
tumor_types = {0: "notumor", 1: "meningioma", 2: "glioma", 3: "pituitary"}
LIMITE_EXTRA = 500  # Cantidad máxima a aumentar

X, Y = [], []

# 1. CARGA DE DATOS SOLO ORIGINALES
print("Paso 1: Cargando datos originales")

#  A. Cargar MAT 
if os.path.exists(data_dir):
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path): continue
        for file in os.listdir(folder_path):
            if not file.endswith(".mat"): continue
            filepath = os.path.join(folder_path, file)
            try:
                with h5py.File(filepath, 'r') as f:
                    image = np.array(f['cjdata']['image']).astype(np.float32)
                    label = int(f['cjdata']['label'][0][0])
                image = cv2.resize(image, (128, 128)) / 255.0
                X.append(image)
                Y.append(label)
            except: continue

#  B. Cargar JPG
if os.path.exists(jpg_tumor_dir):
    for class_name in tumor_file_names:
        folder_path = os.path.join(jpg_tumor_dir, class_name)
        if not os.path.exists(folder_path): continue
        
        if class_name == "notumor": label_id = 0
        elif class_name == "meningioma": label_id = 1
        elif class_name == "glioma": label_id = 2
        elif class_name == "pituitary": label_id = 3

        jpg_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        
        for file in jpg_files:
            filepath = os.path.join(folder_path, file)
            try:
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None: continue
                image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0
                X.append(image)
                Y.append(label_id)
            except: continue

X = np.array(X).reshape(-1, 128, 128, 1)
Y = np.array(Y)
print(f"Total de imágenes originales cargadas: {len(X)}")

# 2. DIVISIÓN TRAIN / TEST
print("Paso 2: Dividiendo en Train y Test")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Guardamos conteo original del TRAIN para la gráfica
unique, counts = np.unique(y_train, return_counts=True)
counts_train_orig = dict(zip(unique, counts)) 

# 3. DATA AUGMENTATION SOLO A TRAIN
print(f"Paso 3: Aumentando Train (Max {LIMITE_EXTRA} para 'notumor')")

X_train_aug = list(X_train)
y_train_aug = list(y_train)
contador_aug = 0
counts_added = {0: 0, 1: 0, 2: 0, 3: 0} 

indices_notumor = [i for i, x in enumerate(y_train) if x == 0]
random.shuffle(indices_notumor)

for idx in indices_notumor:
    if contador_aug >= LIMITE_EXTRA: break
    img_flip = cv2.flip(X_train[idx], 1)
    X_train_aug.append(img_flip.reshape(128, 128, 1))
    y_train_aug.append(0)
    
    counts_added[0] += 1
    contador_aug += 1

X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)

# GRAFICA 1: BALANCE DEL SET DE ENTRENAMIENTO
labels = [tumor_types[i] for i in range(4)]
orig_vals = [counts_train_orig.get(i, 0) for i in range(4)]
aug_vals = [counts_added.get(i, 0) for i in range(4)]
total_vals = [o + a for o, a in zip(orig_vals, aug_vals)]
x = np.arange(len(labels))
width = 0.6

plt.figure(figsize=(10, 6))
plt.bar(x, orig_vals, width, label='Originales (Train)', color='#3498db')
plt.bar(x, aug_vals, width, bottom=orig_vals, label='Aumentadas (Flip)', color='#e74c3c', hatch='//')
plt.ylabel('Cantidad de Imágenes')
plt.title('Distribución final del Set de Entrenamiento')
plt.xticks(x, labels)
plt.legend()
for i in range(len(labels)):
    plt.text(i, total_vals[i] + 15, str(total_vals[i]), ha='center', fontweight='bold')
plt.ylim(0, max(total_vals) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 4. ENTRENAMIENTO
print(f"Iniciando entrenamiento con {len(X_train)} imágenes")

model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                    validation_data=(X_test, y_test), callbacks=callbacks)

# Guardar modelo
os.makedirs("C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\models", exist_ok=True)
model.save("C:\\Users\\Luce\\Documents\\SEXTO SEMESTRE\\software\\Proyecto_final\\models\\tumor_classifier_final.h5")

# 5. EVALUACIÓN Y GRÁFICAS FINALES
y_pred_proba = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1)

print("\nReporte de Clasificación")
print(classification_report(y_test, y_pred_classes, target_names=list(tumor_types.values())))

# GRAFICA 2: MATRIZ DE CONFUSIÓN
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=tumor_types.values(), yticklabels=tumor_types.values())
plt.title("Matriz de Confusión")
plt.xlabel("Predicción del Modelo")
plt.ylabel("Clase Real")
plt.show()

# GRAFICA 3: CURVAS ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 4

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC {0} (AUC = {1:0.2f})'.format(tumor_types[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# GRAFICA 4: CURVAS DE APRENDIZAJE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Loss')
plt.legend()
plt.show()
