from src.CascadeClassifiers import CascadeClassifier

## DATASET FACE
model_path = "model_face.txt"
P_paths = '/media/pandu/Dewabrata/natural images/face'
N_paths = '/media/pandu/Dewabrata/natural images/non-face'
imsize = (24,24)

model = CascadeClassifier()
model.feature_extracting(P_paths, N_paths)
model.fit(Ftarget=0.2, Dtarget=0.8, f=0.2, d=0.8)
model.save(model_path)
