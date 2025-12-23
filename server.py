from flask import Flask, jsonify, send_file, request
from tkinter import filedialog, Tk
from pruebas3 import run_pipeline
import numpy as np
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

selected_folder = None
volume = None
verts = None
faces = None
label = None
slice_index = 0

@app.route("/select-folder")
def select_folder():
    global selected_folder
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    selected_folder = folder
    return jsonify({"folder": folder})

@app.route("/run")
def run():
    global volume, verts, faces, label, slice_index
    slice_index = 0

    volume, verts, faces, label = run_pipeline(selected_folder)

    return jsonify({"label": label})

@app.route("/slice/<int:i>")
def send_slice(i):
    global volume
    slice_img = volume[i]
    plt.imshow(slice_img, cmap="gray")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/slice-change")
def slice_change():
    global slice_index, volume
    d = int(request.args.get("d"))
    slice_index = max(0, min(slice_index + d, len(volume)-1))
    return jsonify({"slice": slice_index})

@app.route("/render3d")
def render_3d():
    global verts, faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.8)
    ax.add_collection3d(mesh)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


app.run(debug=True)
