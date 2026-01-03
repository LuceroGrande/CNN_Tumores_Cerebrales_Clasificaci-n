from flask import Flask, render_template, request, jsonify
from pipeline import run_pipeline
import cv2
import os
import base64
import matplotlib
matplotlib.use("Agg") # Backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes, label, regionprops


app = Flask(__name__)

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

@app.route("/")
def home():
    return render_template("pagina.html")

@app.route("/procesar", methods=["POST"])
def procesar():
    archivos = request.files.getlist("folder")
    if not archivos:
        return jsonify({"error": "No se recibieron archivos"}), 400

    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)

    # Limpiar carpeta temporal de forma segura
    for root, _, files in os.walk(temp_dir):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except Exception as e:
                print(f"Error borrando archivo temporal: {e}")

    # Guardar archivos
    for file in archivos:
        path = os.path.join(temp_dir, file.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file.save(path)

    # Ejecutar Pipeline
    volume, mask_volume, _, _, label_pred, confidences = run_pipeline(temp_dir)

    slices_b64 = []
    
    # CONFIGURACIÓN
    THRESHOLD = 0.05         
    ALPHA = 0.4              
    TARGET_SIZE = (800, 800) 

    # 1. PROCESAMIENTO 2D 
    for i in range(volume.shape[0]):
        slice_i = volume[i]
        mask_prob = mask_volume[i]

        # A. Binarizar
        mask_bin = (mask_prob > THRESHOLD).astype(np.uint8)

        # B. Redimensionar
        if mask_bin.shape != slice_i.shape:
            mask_bin = cv2.resize(
                mask_bin.astype(np.float32),
                (slice_i.shape[1], slice_i.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8) 

        # C. Limpieza 2D
        if np.max(mask_bin) > 0:
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                mask_clean = np.zeros_like(mask_bin)
                cv2.drawContours(mask_clean, [largest_c], -1, 1, thickness=cv2.FILLED)
                mask_bin = mask_clean

        # D. Imagen Base
        img_base = ((slice_i - slice_i.min()) /
                 (slice_i.max() - slice_i.min() + 1e-8) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_base, cv2.COLOR_GRAY2BGR)

        # E. Pintar
        if np.max(mask_bin) > 0:
            overlay = img_bgr.copy()
            overlay[mask_bin == 1] = [0, 0, 255]
            cv2.addWeighted(overlay, ALPHA, img_bgr, 1 - ALPHA, 0, img_bgr)
            
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, contours, -1, (0, 255, 255), 2)

        # F. Guardar
        img_final = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        _, buffer = cv2.imencode(".png", img_final)
        slices_b64.append(base64.b64encode(buffer).decode("utf-8"))

    # 3. PROCESAMIENTO 3D
    img3d_b64 = ""
    
    if np.sum(mask_volume > THRESHOLD) > 10:
        try:
            # A. Float32
            vol_3d = np.array(mask_volume, dtype=np.float32)

            # B. Suavizado
            vol_smooth = gaussian_filter(vol_3d, sigma=0.5)

            # C. Padding
            vol_padded = np.pad(vol_smooth, pad_width=1, mode='constant', constant_values=0)

            # D. Limpieza de Islas 3D
            vol_bin_temp = vol_padded > THRESHOLD
            if np.any(vol_bin_temp):
                lbl_vol, num_features = label(vol_bin_temp, return_num=True)
                
                if num_features > 1:
                    regions = regionprops(lbl_vol)
                    largest_region = max(regions, key=lambda r: r.area)
                    mask_keep = (lbl_vol == largest_region.label)
                    vol_padded = vol_padded * mask_keep

            # E. Marching Cubes
            if np.max(vol_padded) > THRESHOLD:
                verts, faces, _, _ = marching_cubes(vol_padded, level=THRESHOLD)

                # F. Visualización
                fig3 = plt.figure(figsize=(6,6))
                ax = fig3.add_subplot(111, projection="3d")
                
                mesh = Poly3DCollection(verts[faces], alpha=0.6)
                mesh.set_facecolor('#d62728') 
                mesh.set_edgecolor('none')
                
                ax.add_collection3d(mesh)

                ax.set_xlim(np.min(verts[:,0]), np.max(verts[:,0]))
                ax.set_ylim(np.min(verts[:,1]), np.max(verts[:,1]))
                ax.set_zlim(np.min(verts[:,2]), np.max(verts[:,2]))

                ax.axis('off')
                ax.view_init(elev=30, azim=45)
                
                img3d_b64 = fig_to_base64(fig3)
                plt.close()

        except Exception as e:
            print(f"Error al generar la imagen 3D: {e}")
            pass
    else:
        print("Volumen tumoral insuficiente para generar modelo 3D.")

    return jsonify({
        "label": label_pred,
        "img2d_slices": slices_b64,
        "img3d": img3d_b64,
        "confidences": confidences
    })

if __name__ == "__main__":
    app.run(debug=True)
