import os
import glob
import random
import shutil

PATCHES_ROOT = r"C:\Users\chama\OneDrive\Desktop\research\patches_out"
SPLIT_ROOT   = r"C:\Users\chama\OneDrive\Desktop\research\split_subject"
SEED = 42

TRAIN = 0.70
VAL   = 0.15
TEST  = 0.15

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def subject_id_from_filename(fname: str) -> str:
    """
    Expected patch filename contains ADNI subject id like:
    ADNI_005_S_0553_...._patch.npy
    -> returns: ADNI_005_S_0553
    """
    base = os.path.basename(fname)
    base = base.replace("_patch.npy", "")
    parts = base.split("_")
    if len(parts) >= 4 and parts[0] == "ADNI":
        return "_".join(parts[:4])  # ADNI_XXX_S_XXXX
    # fallback: if pattern unexpected, use everything before first double underscore
    return parts[0]

def build_subject_map(side: str, cls: str):
    """
    Returns dict: subject_id -> filepath
    """
    d = os.path.join(PATCHES_ROOT, side, cls)
    files = sorted(glob.glob(os.path.join(d, "*.npy")))
    m = {}
    for f in files:
        sid = subject_id_from_filename(f)
        # If multiple patches per subject exist, keep first and warn (rare for your setup)
        if sid not in m:
            m[sid] = f
    return m

def split_subjects(subject_ids):
    random.shuffle(subject_ids)
    n = len(subject_ids)
    n_train = int(n * TRAIN)
    n_val   = int(n * VAL)
    train = subject_ids[:n_train]
    val   = subject_ids[n_train:n_train+n_val]
    test  = subject_ids[n_train+n_val:]
    return train, val, test

def copy_pair(left_path, right_path, split_name, cls):
    out_left  = os.path.join(SPLIT_ROOT, "left",  split_name, cls)
    out_right = os.path.join(SPLIT_ROOT, "right", split_name, cls)
    ensure_dir(out_left)
    ensure_dir(out_right)
    shutil.copy2(left_path,  os.path.join(out_left,  os.path.basename(left_path)))
    shutil.copy2(right_path, os.path.join(out_right, os.path.basename(right_path)))

def process_class(cls: str):
    left_map  = build_subject_map("left", cls)
    right_map = build_subject_map("right", cls)

    # Keep only subjects that exist on BOTH sides
    common = sorted(set(left_map.keys()) & set(right_map.keys()))
    missing_left  = sorted(set(right_map.keys()) - set(left_map.keys()))
    missing_right = sorted(set(left_map.keys()) - set(right_map.keys()))

    print(f"\n[{cls}] left={len(left_map)} right={len(right_map)} common={len(common)}")
    if missing_left:
        print(f"  ⚠ Missing LEFT for {len(missing_left)} subjects (will be ignored)")
    if missing_right:
        print(f"  ⚠ Missing RIGHT for {len(missing_right)} subjects (will be ignored)")

    train_ids, val_ids, test_ids = split_subjects(common)

    def handle(split_name, ids):
        for sid in ids:
            copy_pair(left_map[sid], right_map[sid], split_name, cls)

    handle("train", train_ids)
    handle("val",   val_ids)
    handle("test",  test_ids)

    print(f"  -> train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")

def main():
    random.seed(SEED)
    process_class("AD")
    random.seed(SEED)  # keep deterministic
    process_class("CN")
    print("\n✅ Done. Split saved to:", SPLIT_ROOT)

if __name__ == "__main__":
    main()
