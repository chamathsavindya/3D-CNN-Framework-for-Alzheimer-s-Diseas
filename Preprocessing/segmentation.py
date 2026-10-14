import os
import glob
import numpy as np
import nibabel as nib

BRAIN_ROOT = r"C:\Users\chama\OneDrive\Desktop\research\output_brain"
MASK_ROOT  = r"C:\Users\chama\OneDrive\Desktop\research\hippocampus masks"
OUT_ROOT   = r"C:\Users\chama\OneDrive\Desktop\research\hippo_only"

def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return img, data

def save_nii(data, ref_img, out_path):
    out = nib.Nifti1Image(data.astype(np.float32), affine=ref_img.affine, header=ref_img.header)
    nib.save(out, out_path)

def binarize(mask):
    return (mask > 0.5).astype(np.float32)

def extract_subject_id(name):
    # Extracts like: ADNI_005_S_0814
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "ADNI":
        return "_".join(parts[:4])
    return None

def find_mask(mask_dir, brain_base_no_ext, side):
    """
    Try multiple naming styles:
    1) exact: <base>_mask_L.nii.gz
    2) exact: <base>_brain_mask_L.nii.gz
    3) glob:  <base>*mask_L*.nii*
    4) fallback by subject id (ADNI_###_S_####)*mask_L*.nii*
    """
    side_tag = "_L" if side == "L" else "_R"

    # 1) exact
    p1 = os.path.join(mask_dir, brain_base_no_ext + f"_mask{side_tag}.nii.gz")
    if os.path.exists(p1):
        return p1

    # 2) exact brain_mask
    p2 = os.path.join(mask_dir, brain_base_no_ext + f"_brain_mask{side_tag}.nii.gz")
    if os.path.exists(p2):
        return p2

    # 3) glob with full base
    candidates = sorted(glob.glob(os.path.join(mask_dir, brain_base_no_ext + f"*mask{side_tag}*.nii*")))
    if candidates:
        return candidates[0]

    # 4) fallback by subject id
    sid = extract_subject_id(brain_base_no_ext)
    if sid:
        candidates = sorted(glob.glob(os.path.join(mask_dir, sid + f"*mask{side_tag}*.nii*")))
        if candidates:
            return candidates[0]

    return None

def strip_nii_gz(name):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name

def process_label(label):
    brain_dir = os.path.join(BRAIN_ROOT, label)
    mask_dir  = os.path.join(MASK_ROOT, label)
    out_dir   = os.path.join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    # Accept both .nii.gz and .nii if any slipped in
    brain_files = sorted(glob.glob(os.path.join(brain_dir, "*_brain.nii.gz"))) + \
                  sorted(glob.glob(os.path.join(brain_dir, "*_brain.nii")))

    print(f"\n[{label}] brain files found: {len(brain_files)}")

    ok, missing = 0, 0

    for brain_path in brain_files:
        base = os.path.basename(brain_path)
        brain_base = strip_nii_gz(base)             # remove extension
        brain_base = brain_base.replace("_brain", "") if brain_base.endswith("_brain") else brain_base
        # IMPORTANT: your brain filenames are like <original>_brain.nii.gz
        # So brain_base_no_ext should be the original base WITHOUT "_brain"
        brain_base_no_ext = strip_nii_gz(base).replace("_brain", "")

        left_mask_path  = find_mask(mask_dir, brain_base_no_ext, "L")
        right_mask_path = find_mask(mask_dir, brain_base_no_ext, "R")

        if not left_mask_path or not right_mask_path:
            print("❌ Missing masks for:", brain_base_no_ext)
            missing += 1
            continue

        brain_img, brain = load_nii(brain_path)
        _, left_mask  = load_nii(left_mask_path)
        _, right_mask = load_nii(right_mask_path)

        if brain.shape != left_mask.shape or brain.shape != right_mask.shape:
            print(f"❌ Shape mismatch for {brain_base_no_ext}: brain{brain.shape}, L{left_mask.shape}, R{right_mask.shape}")
            continue

        left_mask_bin  = binarize(left_mask)
        right_mask_bin = binarize(right_mask)

        left_hippo  = brain * left_mask_bin
        right_hippo = brain * right_mask_bin

        out_left  = os.path.join(out_dir, brain_base_no_ext + "_left_hippo.nii.gz")
        out_right = os.path.join(out_dir, brain_base_no_ext + "_right_hippo.nii.gz")

        save_nii(left_hippo, brain_img, out_left)
        save_nii(right_hippo, brain_img, out_right)

        ok += 1

    print(f"[{label}] ✅ done: {ok} | ❌ missing masks: {missing}")

if __name__ == "__main__":
    process_label("AD")
    process_label("CN")
    print("\nAll finished ✅")
