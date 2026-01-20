import os
import urllib.request

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
CLASSES = ["car", "tree", "house", "cat", "bicycle"]

os.makedirs("data/quickdraw", exist_ok=True)

for cls in CLASSES:
    url = BASE_URL + cls.replace(" ", "%20") + ".npy"
    out_path = f"data/quickdraw/{cls}.npy"

    if os.path.exists(out_path):
        print(f"✅ already exists: {out_path}")
        continue

    print(f"⬇️ downloading {cls} ...")
    urllib.request.urlretrieve(url, out_path)
    print(f"✅ saved: {out_path}")

print("✅ Done.")
