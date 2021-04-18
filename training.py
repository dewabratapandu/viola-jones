import os
from src.CascadeClassifiers import CascadeClassifier

## DATASET
model_path = "model.txt"
P_path = 'dataset/face'
N_path = 'dataset/non-face'
imsize = (24, 24)

P_paths = [os.path.join(P_path, s) for s in sorted(os.listdir(P_path))]
N_paths = [os.path.join(N_path, s) for s in sorted(os.listdir(N_path))]

model = CascadeClassifier()
model.feature_extracting(P_paths, N_paths)
model.fit(Ftarget=0.1, Dtarget=0.9, f=0.1, d=0.9)
model.save(model_path)
