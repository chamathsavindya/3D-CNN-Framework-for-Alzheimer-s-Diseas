import os
import glob
import numpy as np
import nibabel as nib

# ====== EDIT THESE PATHS ======
LEFT_ROOT  = r"C:\Users\chama\OneDrive\Desktop\research\Left hippocampus"
RIGHT_ROOT = r"C:\Users\chama\OneDrive\Desktop\research\Right hippocampus"
OUT_ROOT   = r"C:\Users\chama\OneDrive\Desktop\research\patches_out"
PATCH_SIZE = (64, 64, 64)  # (D, H, W) in array order: (X, Y, Z) from nibabel data
# ==============================

def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data

def center_of_nonzero(vol):
    coords = np.argwhere(vol != 0)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)  # (x, y, z) float center

def crop_with_padding(vol, center, patch_size):
    px, py, pz = patch_size
    cx, cy, cz = [int(round(c)) for c in center]

    # half sizes
    hx, hy, hz = px // 2, py // 2, pz // 2

    # desired crop bounds in source
    x1, x2 = cx - hx, cx - hx + px
    y1, y2 = cy - hy, cy - hy + py
    z1, z2 = cz - hz, cz - hz + pz

    # compute padding needed
    pad_x1 = max(0, -x1); pad_x2 = max(0, x2 - vol.shape[0])
    pad_y1 = max(0, -y1); pad_y2 = max(0, y2 - vol.shape[1])
    pad_z1 = max(0, -z1); pad_z2 = max(0, z2 - vol.shape[2])

    # clamp crop bounds to valid range
    x1c, x2c = max(0, x1), min(vol.shape[0], x2)
    y1c, y2c = max(0, y1), min(vol.shape[1], y2)
    z1c, z2c = max(0, z1), min(vol.shape[2], z2)

    cropped = vol[x1c:x2c, y1c:y2c, z1c:z2c]

    # pad to exact patch size
    cropped = np.pad(
        cropped,
        pad_width=((pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2)),
        mode="constant",
        constant_values=0
    )

    # safety: force exact size
    if cropped.shape != patch_size:
        # rare edge-case: if patch_size odd/even rounding causes 1-voxel mismatch
        cropped = cropped[:px, :py, :pz]
        cropped = np.pad(
            cropped,
            ((0, px - cropped.shape[0]), (0, py - cropped.shape[1]), (0, pz - cropped.shape[2])),
            mode="constant",
            constant_values=0
        )

    return cropped

def zscore_nonzero(patch, eps=1e-6):
    nonzero = patch[patch != 0]
    if nonzero.size < 20:
        # fallback: normalize using whole patch if too few voxels (very rare)
        mean = patch.mean()
        std = patch.std()
    else:
        mean = nonzero.mean()
        std = nonzero.std()

    if std < eps:
        return patch  # avoid divide-by-zero; keep as-is
    return (patch - mean) / std

def process_side(side_name, root_dir):
    for cls in ["AD", "CN"]:
        in_dir = os.path.join(root_dir, cls)
        out_dir = os.path.join(OUT_ROOT, side_name, cls)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(in_dir, "*.nii")) + glob.glob(os.path.join(in_dir, "*.nii.gz")))
        print(f"{side_name}/{cls}: {len(files)} files")

        for fp in files:
            vol = load_nii(fp)

            center = center_of_nonzero(vol)
            if center is None:
                print("❌ No hippocampus voxels (all zeros):", os.path.basename(fp))
                continue

            patch = crop_with_padding(vol, center, PATCH_SIZE)
            patch = zscore_nonzero(patch)

            # add channel dimension for CNN: (C, D, H, W)
            patch = patch[np.newaxis, ...].astype(np.float32)

            base = os.path.basename(fp)
            base = base.replace(".nii.gz", "").replace(".nii", "")
            out_path = os.path.join(out_dir, base + "_patch.npy")

            np.save(out_path, patch)

process_side("left", LEFT_ROOT)
process_side("right", RIGHT_ROOT)

print("\n✅ Done. Patches saved to:", OUT_ROOT)
