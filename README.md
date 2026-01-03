# Clasificacion por medio de CNN para tumores cerebrales
Una página web completa para realizar un análisis automatizado de imágenes de resonancia magnética (MRI) cerebral. El sistema utiliza Deep Learning para clasificar el tipo de tumor y segmentar la región afectada, generando además una reconstrucción 3D del tumor para mejor visualización médica.

-> Características Principales
Hace una clasificación multiclase, debido a que identifica el tipo de tumor entre cuatro categorías, las cuales son meningioma, glioma, pituitary y un cerebro sano.

Realiza una segmentación semántica, donde se utiliza una red UNet para generar máscaras precisas del área tumoral.

Posteriormente se hacen unas visualizaciones avanzadas como cortes axiales (2D) con superposición del tumor y contornos definidos, además de una reconstrucción volumétrica (3D) del tumor.

-> Dataset
El modelo fue entrenado utilizando el siguiente conjunto de datos de acceso público:

Nombre: Brain Tumor Dataset de Figshare

Enlace: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

Descripción: Contiene 3064 imágenes de resonancia magnética ponderadas en T1 con contraste de 233 pacientes.

Nombre: Brain Tumor MRI Dataset de Kaggle

Enlace: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Nota: Para este dataset se utilizó la clase de notumor y la mitad de las demás clases.

Imágenes usadas exactamente:
https://drive.google.com/drive/folders/1Jo5bjOuRnje_Eb4fExw0dwDru1TTOGjj

-> Estructura del Proyecto
El código está organizado de la siguiente manera:

├── app.py             
├── pipeline.py          
├── requirements.txt     
├── templates/
│   └── pagina.html     
│── segmentation_model_nopr.pth 
│── classifier_tumor.h5            

-> Modelos implementados con IA
La segmentación con UNet donde el modelo recibe imágenes de 256x256 y genera una máscara binaria. Y la clasificación con CNN, donde se utiliza una red neuronal convolucional para categorizar el tipo de tumor existente.
