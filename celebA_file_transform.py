import pandas as pd
import os

ROOT = os.path.expanduser("~/.cache/c2bm/CelebA/celeba")

# 1) list_attr_celeba.csv -> list_attr_celeba.txt
df = pd.read_csv(f"{ROOT}/list_attr_celeba.csv")
# 첫줄에 attribute 개수 추가 + header 출력
with open(f"{ROOT}/list_attr_celeba.txt", "w") as f:
    f.write(f"{len(df.columns)-1}\n")    # 40
    f.write(" ".join(df.columns) + "\n")
    for i,row in df.iterrows():
        values = " ".join(str(v) for v in row.values)
        f.write(values + "\n")

print("Converted list_attr_celeba.txt")

# 2) list_eval_partition.csv -> list_eval_partition.txt
df = pd.read_csv(f"{ROOT}/list_eval_partition.csv")
df.to_csv(f"{ROOT}/list_eval_partition.txt", sep=" ", index=False, header=False)
print("Converted list_eval_partition.txt")

# 3) list_bbox_celeba.csv -> list_bbox_celeba.txt
df = pd.read_csv(f"{ROOT}/list_bbox_celeba.csv")
df.to_csv(f"{ROOT}/list_bbox_celeba.txt", sep=" ", index=False)
print("Converted list_bbox_celeba.txt")

# 4) list_landmarks_align_celeba.csv -> list_landmarks_align_celeba.txt
df = pd.read_csv(f"{ROOT}/list_landmarks_align_celeba.csv")
df.to_csv(f"{ROOT}/list_landmarks_align_celeba.txt", sep=" ", index=False)
print("Converted list_landmarks_align_celeba.txt")

# 5) identity_CelebA.csv -> identity_CelebA.txt (있으면)
id_path = f"{ROOT}/identity_CelebA.csv"
if os.path.exists(id_path):
    df = pd.read_csv(id_path)
    df.to_csv(f"{ROOT}/identity_CelebA.txt", sep=" ", index=False, header=False)
    print("Converted identity_CelebA.txt")
else:
    print("identity_CelebA.csv not found, skipping.")
