from src.CascadeClassifiers import CascadeClassifier

## DATASET GEDUNG REKTORAT UB
model_path = "model_ub.txt"
P_paths = 'training_ub/true'
N_paths = 'training_ub/false'
imsize = (25,50)
features_paths = '/media/pandu/Dewabrata/ViolaJones-Data-UB/haar_features'
hypothesis_paths = '/media/pandu/Dewabrata/ViolaJones-Data-UB/hypothesis'

## DATASET MATA KATARAK
# model_path = "model_katarak.txt"
# P_paths = 'training_katarak/true'
# N_paths = 'training_katarak/false'
# imsize = (15, 15)
# features_paths = '/media/pandu/Dewabrata/ViolaJones-Data-Katarak/haar_features'
# hypothesis_paths = '/media/pandu/Dewabrata/ViolaJones-Data-Katarak/hypothesis'


## DATASET FACE
# model_path = "model_face.txt"
# P_paths = '/media/pandu/Dewabrata/natural images/face'
# N_paths = '/media/pandu/Dewabrata/natural images/non-face'
# imsize = (24,24)
# features_paths = '/media/pandu/Dewabrata/natural images/haar_features'
# hypothesis_paths = '/media/pandu/Dewabrata/natural images/hypothesis'

model = CascadeClassifier()
model.feature_extracting(P_paths, N_paths, step=2, featureTypes=("type_one",))

feat = model.allImgFeatures
j = 10
for i in range(len(feat)):
    feat[i].pop(j)
    print(len(feat[i]))


#model.fit()
#model.save('model_face.txt')
"""CARI CARA UNTUK HAPUS FEATURE DARI ALLIMGFEATURES,
    BUKAN CUMA INDEXNYA YG HILANG"""