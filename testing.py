import numpy as np
import cv2
import os
import csv
import random
import src.haar_extractor as haar
import src.model as m

def test(model_path, test_path, imsize):
    # LOAD MODEL
    model = m.loadModel(model_path)

    TP = 0; TN = 0; FP = 0; FN = 0

    listdir = os.listdir(test_path)
    listdir = random.sample(listdir, len(listdir))

    for filename in listdir:
        im = cv2.imread(os.path.join(test_path, filename), 0)
        im = cv2.resize(im, imsize)
        im = haar.integralImage(im)
        label = m.cascadeClassifier(im, model)

        text = 'true' if label==1 else 'false'
        # print(text)
        # cv2.imshow(filename, im)
        # cv2.waitKey(0)

        if filename[:4] == 'true' and text == 'true':
            TP += 1
        elif filename[:4] == 'true' and text != 'true':
            FN += 1
        elif filename[:5] == 'false' and text == 'false':
            TN += 1
        elif filename[:5] == 'false' and text != 'false':
            FP += 1
    print("\nTP", TP, "TN", TN, "FP", FP, "FN", FN)
    print("Accuracy :", (TP+TN)/(TP+TN+FP+FN))

model_path = 'model_katarak.txt'
test_path = '../ViolaJones/training_image'
test(model_path, test_path, (50,25))

model_path = 'model.txt'
test_path = '../ViolaJones/training_ub'
test(model_path, test_path, (25,50))