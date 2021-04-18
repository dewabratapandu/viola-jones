import cv2
import os
import numpy as np
from src.CascadeClassifiers import CascadeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def test(model_path, P_paths, N_paths, imsize):
    # LOAD MODEL
    model = CascadeClassifier()
    model.load(model_path)

    len_P = len(P_paths); len_N = len(N_paths)

    posResult = [model.predict(cv2.imread(p,0), imsize=imsize) for p in P_paths]
    negResult = [model.predict(cv2.imread(n,0), imsize=imsize) for n in N_paths]
    y_true = np.hstack((np.zeros(len_N), np.ones(len_P)))
    y_pred = np.array(negResult + posResult)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[1,0]).ravel()

    print("\nTP", TP, "TN", TN, "FP", FP, "FN", FN)
    print("Accuracy :", (TP+TN)/(TP+TN+FP+FN))


## DATASET
model_path = "model.txt"
P_path = 'dataset/face'
N_path = 'dataset/non-face'
imsize = (24, 24)

P_paths = [os.path.join(P_path, s) for s in sorted(os.listdir(P_path))]
N_paths = [os.path.join(N_path, s) for s in sorted(os.listdir(N_path))]
P_train, P_test = train_test_split(P_paths, test_size=0.2, shuffle=False)
N_train, N_test = train_test_split(N_paths, test_size=0.2, shuffle=False)

model = CascadeClassifier()
model.feature_extracting(P_train, N_train)
model.fit(Ftarget=0.1, Dtarget=0.9, f=0.1, d=0.9)
model.save(model_path)

print('TESTING...')
test(model_path, P_test, N_test, (24, 24))
