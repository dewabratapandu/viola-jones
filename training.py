from src.CascadeClassifiers import CascadeClassifier

## DATASET GEDUNG REKTORAT UB
# model_path = "model_ub.txt"
# P_paths = 'training_ub/true'
# N_paths = 'training_ub/false'
# imsize = (25,50)

## DATASET MATA KATARAK
model_path = "model_katarak.txt"
P_paths = 'training_katarak/true'
N_paths = 'training_katarak/false'
imsize = (24, 24)


## DATASET FACE
# model_path = "model_face.txt"
# P_paths = '/media/pandu/Dewabrata/natural images/face'
# N_paths = '/media/pandu/Dewabrata/natural images/non-face'
# imsize = (24,24)

model = CascadeClassifier()
model.feature_extracting(P_paths, N_paths, step=2)
model.fit(f=0.3, d=0.7)
model.save(model_path)