import scipy.io
import os

def extract_mat_metadata(mat_file_path):
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {mat_file_path}")
    
    # Cargar el archivo .mat
    mat_contents = scipy.io.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    
    # Filtrar las entradas del sistema de MATLAB (__header__, __version__, __globals__)
    metadata_keys = [key for key in mat_contents.keys() if not key.startswith('__')]
    
    metadata = {}
    for key in metadata_keys:
        value = mat_contents[key]
        metadata[key] = {
            'type': type(value).__name__,
            'shape': getattr(value, 'shape', 'N/A'),
            'size': getattr(value, 'size', 'N/A'),
        }
    
    return metadata

# Ejemplo de uso
mat_file = "C:/Users/Luce/Downloads/proyecto_final/proyecto_final/patient_100360/3/443.mat"
metadata = extract_mat_metadata(mat_file)

print("Metadatos del archivo .mat:")
for key, info in metadata.items():
    print(f"{key}: tipo={info['type']}, shape={info['shape']}, size={info['size']}")
