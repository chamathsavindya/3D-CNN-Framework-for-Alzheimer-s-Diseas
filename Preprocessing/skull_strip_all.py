import os
import subprocess

input_dirs = ["data/AD", "data/CN"]
output_root = "output_brain"

os.makedirs(output_root, exist_ok=True)

for input_dir in input_dirs:
    label = os.path.basename(input_dir)
    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            in_path = os.path.join(input_dir, file)

            base = file
            if base.endswith(".nii.gz"):
                base = base[:-7]
            elif base.endswith(".nii"):
                base = base[:-4]

            out_path = os.path.join(out_dir, base + "_brain.nii.gz")

            cmd = ["hd-bet", "-i", in_path, "-o", out_path, "-device", "cpu"]

            print("Processing:", in_path)
            subprocess.run(cmd, check=False)
