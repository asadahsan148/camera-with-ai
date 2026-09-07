"""
Package original + augmented frames and labels into a Kaggle-ready zip.

Creates a flat directory structure:
  scos-corners-v2/
    cam1_frame_01.jpg
    cam1_frame_01.txt
    cam1_frame_01_aug00.jpg
    cam1_frame_01_aug00.txt
    ...

Then zips it for upload to Kaggle.
"""

import os
import shutil
import zipfile
import glob

OUTPUT_DIR = "scos-corners-v2"
ZIP_NAME = "scos-corners-v2.zip"


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 1. Copy original annotated frames + labels
    print("Copying original frames + labels...")
    orig_count = 0
    for label_file in glob.glob("labels/*.txt"):
        base = os.path.splitext(os.path.basename(label_file))[0]
        # Find matching image
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join("frames", f"{base}{ext}")
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(OUTPUT_DIR, f"{base}{ext}"))
                shutil.copy2(label_file, os.path.join(OUTPUT_DIR, f"{base}.txt"))
                orig_count += 1
                break

    print(f"  Original frames: {orig_count}")

    # 2. Copy augmented frames + labels
    print("Copying augmented frames + labels...")
    aug_count = 0
    for img_file in glob.glob("augmented/images/*.jpg"):
        base = os.path.splitext(os.path.basename(img_file))[0]
        lbl_file = os.path.join("augmented", "labels", f"{base}.txt")
        if os.path.exists(lbl_file):
            shutil.copy2(img_file, os.path.join(OUTPUT_DIR, f"{base}.jpg"))
            shutil.copy2(lbl_file, os.path.join(OUTPUT_DIR, f"{base}.txt"))
            aug_count += 1

    print(f"  Augmented frames: {aug_count}")

    total = orig_count + aug_count
    print(f"  Total: {total} frames")

    # 3. Create zip
    print(f"\nCreating {ZIP_NAME}...")
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, fname)
            zf.write(fpath, os.path.join(OUTPUT_DIR, fname))

    zip_size = os.path.getsize(ZIP_NAME) // (1024 * 1024)
    print(f"  {ZIP_NAME}: {zip_size} MB")
    print(f"\nDone! Upload {ZIP_NAME} to Kaggle as a new dataset version.")
    print(f"Dataset path in Kaggle: /kaggle/input/datasets/asadahsan148/scos-corners-v2")


if __name__ == "__main__":
    main()
