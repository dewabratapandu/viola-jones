import src.cascade as cascade
import data_extract as extract
import numpy as np

## DATASET GEDUNG REKTORAT UB
# model_path = "model_ub.txt"
# P_paths = 'training_ub/true'
# N_paths = 'training_ub/false'
# imsize = (25,50)
# features_paths = '/media/pandu/Dewabrata/ViolaJones-Data-UB/haar_features'
# hypothesis_paths = '/media/pandu/Dewabrata/ViolaJones-Data-UB/hypothesis'

## DATASET MATA KATARAK
model_path = "model_katarak.txt"
P_paths = 'training_katarak/true'
N_paths = 'training_katarak/false'
imsize = (15, 15)
features_paths = '/media/pandu/Dewabrata/ViolaJones-Data-Katarak/haar_features'
hypothesis_paths = '/media/pandu/Dewabrata/ViolaJones-Data-Katarak/hypothesis'


## DATASET FACE
# model_path = "model_face.txt"
# P_paths = '/media/pandu/Dewabrata/natural images/face'
# N_paths = '/media/pandu/Dewabrata/natural images/non-face'
# imsize = (24,24)
# features_paths = '/media/pandu/Dewabrata/natural images/haar_features'
# hypothesis_paths = '/media/pandu/Dewabrata/natural images/hypothesis'

# Mempersiapkan file model.txt
model = open(model_path, "w")
model.write(""); model.close()

extract.prepare_data(P_paths, N_paths, '/media/pandu/Dewabrata/ViolaJones-Data-UB', imsize)
# cascade.trainCascade(model_path, P_paths, N_paths, imsize, 0.001, 0.9, 0.1, 0.9, features_paths, hypothesis_paths, 30)