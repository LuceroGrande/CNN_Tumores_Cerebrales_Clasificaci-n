import h5py
import os

# Mapa de etiquetas a tipos de tumor según el dataset de Figshare
LABELS_MAP = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary"
}

def get_mat_metadata_and_tumor(mat_file_path):
    """
    Lee un archivo .mat (v7.3 HDF5), devuelve metadatos de cada dataset
    y el tipo de tumor según el campo 'label'.
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {mat_file_path}")
    
    metadata = {}
    tumor_type = "Desconocido"
    
    with h5py.File(mat_file_path, 'r') as f:
        # Función recursiva para recorrer todos los datasets
        def visit(name, obj):
            metadata[name] = {
                'type': type(obj).__name__,
                'shape': getattr(obj, 'shape', 'N/A'),
                'dtype': getattr(obj, 'dtype', 'N/A')
            }
        f.visititems(visit)
        
        # Leer el label y determinar tipo de tumor
        if 'cjdata/label' in f:
            label = f['cjdata/label'][()]
            # Algunos labels vienen como float o array escalar
            if hasattr(label, "__len__"):
                label = int(label[0]) if len(label) > 0 else int(label)
            else:
                label = int(label)
            tumor_type = LABELS_MAP.get(label, "Desconocido")
    
    return metadata, tumor_type

# Ejemplo de uso
#mat_file = "ruta/a/tu/archivo.mat"
mat_file = r"C:\Users\Luce\Documents\SEXTO SEMESTRE\computo paralelo\proyecto 2.0\brainTumorDataPublic_767-1532\446.mat"
metadata, tumor_type = get_mat_metadata_and_tumor(mat_file)

print(f"Tipo de tumor: {tumor_type}")
print("\nMetadatos del archivo .mat:")
for key, info in metadata.items():
    print(f"{key}: tipo={info['type']}, shape={info['shape']}, dtype={info['dtype']}")



