import kagglehub
import zipfile
import os
import shutil
from pathlib import Path

# 원하는 최종 경로 설정
TARGET_ROOT = Path.home() / ".cache/c2bm/CelebA/celeba"
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

print(">> Target path:", TARGET_ROOT)

# -----------------------------
# Step 1. Kaggle에서 다운로드
# -----------------------------
print(">> Downloading CelebA from Kaggle...")
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print(">> Kagglehub downloaded path:", path)

# zip 파일 및 txt 파일 찾기
download_dir = Path(path)

# img_align_celeba.zip
zip_file = download_dir / "img_align_celeba.zip"
attr_file = download_dir / "list_attr_celeba.txt"
bbox_file = download_dir / "list_bbox_celeba.txt"
landmark_file = download_dir / "list_landmarks_align_celeba.txt"

# -----------------------------
# Step 2. 이미지 압축 풀기
# -----------------------------
extract_dir = download_dir / "img_align_celeba"

if not extract_dir.exists():
    print(">> Extracting image zip...")
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_dir)
else:
    print(">> Already extracted, skipping.")

# -----------------------------
# Step 3. 원하는 경로로 복사
# -----------------------------
print(">> Copying to target directory...")

# 이미지 폴더 복사
dst_img_dir = TARGET_ROOT / "img_align_celeba"
if dst_img_dir.exists():
    print(">> Removing old img_align_celeba...")
    shutil.rmtree(dst_img_dir)

shutil.copytree(extract_dir, dst_img_dir)

# 텍스트 파일 복사
for f in [attr_file, bbox_file, landmark_file]:
    if f.exists():
        shutil.copy(f, TARGET_ROOT)

print(">> CelebA prepared successfully!")
print("Final CelebA path:", TARGET_ROOT)
