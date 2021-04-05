import numpy as np
import cv2
import os
import csv
import random
from src.CascadeClassifiers import CascadeClassifier
import src.haar_extractor as haar

def test(model_path, P_path, N_path, imsize):
    # LOAD MODEL
    model = CascadeClassifier()
    model.load(model_path)

    P_paths = [os.path.join(P_path, s) for s in sorted(os.listdir(P_path))]
    N_paths = [os.path.join(N_path, s) for s in sorted(os.listdir(N_path))]
    len_P = len(P_paths); len_N = len(N_paths)

    posResult = [model.predict(p, imsize=imsize) for p in P_paths]
    negResult = [model.predict(n, imsize=imsize) for n in N_paths]
    TP, FN, TN, FP = model.confusionMatrix(len_P, len_N, posResult, negResult)

    print("\nTP", TP, "TN", TN, "FP", FP, "FN", FN)
    print("Accuracy :", (TP+TN)/(TP+TN+FP+FN))

# model_path = 'model_ub.txt'
# P_path = 'training_ub/true'
# N_path = 'training_ub/false'
# test(model_path, P_path, N_path, (25,50))

# model_path = 'model_katarak.txt'
# P_path = 'training_katarak/true'
# N_path = 'training_katarak/false'
# test(model_path, P_path, N_path, (24, 24))

model_path = "model_face.txt"
P_path = '/media/pandu/Dewabrata/natural images/face'
N_path = '/media/pandu/Dewabrata/natural images/non-face'
test(model_path, P_path, N_path, (24,24))
