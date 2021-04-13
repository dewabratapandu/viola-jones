import cv2
import os
from src.CascadeClassifiers import CascadeClassifier
import src.haar_extractor as haar
from sklearn.model_selection import train_test_split

def test(model_path, P_paths, N_paths, imsize):
    # LOAD MODEL
    model = CascadeClassifier()
    model.load(model_path)

    len_P = len(P_paths); len_N = len(N_paths)

    posResult = [model.predict(cv2.imread(p,0), imsize=imsize) for p in P_paths]
    negResult = [model.predict(cv2.imread(n,0), imsize=imsize) for n in N_paths]
    TP, FN, TN, FP = model.confusionMatrix(len_P, len_N, posResult, negResult)

    print("\nTP", TP, "TN", TN, "FP", FP, "FN", FN)
    print("Accuracy :", (TP+TN)/(TP+TN+FP+FN))


## DATASET
model_path = "model.txt"
P_path = 'Dataset/face'
N_path = 'Dataset/non-face'
imsize = (24, 24)

P_paths = [os.path.join(P_path, s) for s in sorted(os.listdir(P_path))]
N_paths = [os.path.join(N_path, s) for s in sorted(os.listdir(N_path))]
P_train, P_test = train_test_split(P_paths, test_size=0.2, shuffle=False)
N_train, N_test = train_test_split(N_paths, test_size=0.2, shuffle=False)

model = CascadeClassifier()
model.feature_extracting(P_train, N_train)
model.fit(Ftarget=0.1, Dtarget=0.9, f=0.2, d=0.8)
model.save(model_path)

print('TESTING...')
test(model_path, P_test, N_test, (24, 24))
