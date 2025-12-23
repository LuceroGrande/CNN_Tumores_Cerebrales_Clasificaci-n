import os
import numpy as np
from PIL import Image
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================
#  PARAMETERS
# ============================================================

pixel_spacing_x = 0.49
pixel_spacing_y = 0.49
slice_spacing   = 7.0

GT_DIR = r"C:\Users\Luce\Downloads\proyecto_final\proyecto_final\output_predictions"
OUTPUT = "tumor_3d_corrected.png"

# ============================================================
#  PACKS USING RANGES
# ============================================================
def lol(a,b,c,d,e,f):
    import matplotlib.pyplot as plt
    packs = [
        {
            "prefix": "",
            "start": a,
            "end": b,      # creates p1_001_gt.png → p1_003_gt.png
            "pad_before": 0,
            "pad_after": 0
        },
        {
            "prefix": "",
            "start": c,
            "end": d,
            "pad_before": 0,
            "pad_after": 0
        },
        {
            "prefix": "",
            "start": e,
            "end": f,
            "pad_before": 0,
            "pad_after": 0
        }
    ]

    # ============================================================
    #  LOAD + PAD + STACK
    # ============================================================

    # ============================================================
    #  LOAD + COMBINE (VOXEL-WISE FUSION)
    # ============================================================

    # Load shape from first pack
    first_fname = f"{packs[0]['prefix']}{packs[0]['start']}_mask.png"
    sample_img = Image.open(os.path.join(GT_DIR, first_fname)).convert("L")
    H, W = np.array(sample_img).shape

    # Final 3D volume with max number of slices across all packs
    max_slices = max([p["end"] - p["start"] + 1 + p["pad_before"] + p["pad_after"] for p in packs])

    # 3D volume initially empty
    volume = np.zeros((max_slices, H, W), dtype=np.uint8)

    # Fill (mix) using OR fusion
    for pack in packs:
        filenames = [f"{pack['prefix']}{i}_mask.png" for i in range(pack["start"], pack["end"] + 1)]

        slice_index = pack["pad_before"]  # starting Z index inside this pack

        for fname in filenames:
            path = os.path.join(GT_DIR, fname)
            img = Image.open(path).convert("L")
            arr = (np.array(img) > 127).astype(np.uint8)

            # OR fusion → mix, don't stack
            volume[slice_index] = np.logical_or(volume[slice_index], arr).astype(np.uint8)

            slice_index += 1

    print("Mixed volume shape:", volume.shape)


    # ============================================================
    #  3D RECONSTRUCTION
    # ============================================================
    # ============================================================
    #  3D RECONSTRUCTION (with z-shift -20 mm for visualization)
    # ============================================================

    spacing = (slice_spacing, pixel_spacing_y, pixel_spacing_x)
    verts, faces, normals, values = marching_cubes(volume, level=0.5, spacing=spacing)

    # --- shift Z by -20 mm (move tumor down in plot) ---
    # verts columns are (x, y, z) after spacing applied; subtract 20 from z
    verts_shifted = verts.copy()
    verts_shifted[:, 2] -= 50.0

    # ============================================================
    #  PLOT RESULT (use shifted verts)
    # ============================================================

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts_shifted[faces], alpha=0.75)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # update axis limits to reflect the z-shift
    ax.set_xlim(0, volume.shape[2] * pixel_spacing_x)
    ax.set_ylim(0, volume.shape[1] * pixel_spacing_y)

    z_min = -20.0
    z_max = volume.shape[0] * slice_spacing - 20.0
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    plt.title("Corrected 3D Tumor Reconstruction (Z shifted -20 mm)")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300)
    plt.close()

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(OUTPUT)

    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray")
    plt.title("Combined Packs")
    plt.axis("off")
    plt.show()

    print("Saved:", OUTPUT)
    return volume, verts, faces

