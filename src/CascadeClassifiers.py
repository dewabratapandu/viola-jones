import os
import csv
import time
import numpy as np
import cv2
import src.adaboost as ada
import src.haar_extractor as haar
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class CascadeClassifier:
    def __init__(self):
        self.cascadeClf = []

    def __del__(self):
        haar.FISRTRUN = True

    def save(self, filename):
        # prepare .txt file
        model = open(filename, "w")
        model.write("")

        # Write cascadeClf to .txt file
        for strClf in self.cascadeClf:
            for weakClf in strClf:
                model.write(str(weakClf[0])); model.write(' ')
                model.write(str(weakClf[1])); model.write(' ')
                model.write(str(weakClf[2])); model.write(' ')
                model.write(str(weakClf[3])); model.write(' ')
                model.write(str(weakClf[4])); model.write(' ')
                model.write(str(weakClf[5])); model.write(' ')
                model.write(str(weakClf[6]))
                model.write("\n")
            model.write("end of cascade\n")
        model.close()

    def load(self, filename):
        self.cascadeClf = []
        strClf = []
        with open(filename) as csvfile:
            modeltxt = csv.reader(csvfile, delimiter=' ')
            for line in modeltxt:
                if(line[0] == 'end'):
                    cp = strClf.copy()
                    self.cascadeClf.append(cp)
                    strClf.clear()
                elif(line is None):
                    continue
                else:
                    for i in range(1, 6):
                        line[i] = int(line[i])
                    line[6] = float(line[6])
                    strClf.append(line)

    def feature_extracting(self, P_paths, N_paths, imsize=(24,24), featureTypes = ("type_one", "type_two", "type_three", "type_four", "type_five"), step=1):
        self.imsize = imsize
        self.P_paths = P_paths
        self.N_paths = N_paths
        self.features = []

        # Haar Feature Extraction
        training_paths = self.N_paths + self.P_paths
        print('Extracting Haar Feature...')
        for i, filename in enumerate(tqdm(training_paths)):
            im = cv2.imread(filename, 0)
            im = cv2.resize(im, self.imsize)
            
            # Calculate integral image
            im = haar.integralImage(im)
            
            # Calculate haar features
            features, features_list = haar.getFeatures(im, featureTypes=featureTypes, step=step)
            self.features.append(features)
            if len(features_list) != 0:
                self.features_list = features_list
        
            
        print('Haar Extracted...')

    def predict(self, im, cascadeClf=None, imsize=None):
        if cascadeClf == None:
            cascadeClf = self.cascadeClf
        if imsize == None:
            imsize = self.imsize
        im = cv2.resize(im, imsize)
        im = haar.integralImage(im)
        for strClf in cascadeClf:
            ahx = 0 # sum of alfa * hx
            sumAlfa = 0 # sum of alfa
            for weakClf in strClf:
                fx = haar.computeFeature(im, weakClf[0], weakClf[1], weakClf[2], weakClf[3], weakClf[4])
                hx = 1 if fx < weakClf[5] else 0
                alfa = float(weakClf[6])
                ahx += alfa * hx
                sumAlfa += alfa
            result = 1 if ahx >= 0.5*sumAlfa else 0
            if(result == 0):
                break
        return result

    def fit(self, Ftarget=0.2, Dtarget=0.8, f=0.2, d=0.8, max_cascade=50):
        F = np.ones(max_cascade, dtype=float)
        D = np.ones(max_cascade, dtype=float)
        remove_indices = []
        i = 0 # cascade index

        while F[i] > Ftarget or D[i] < Dtarget:
            if(i >= max_cascade):
                break
            print('\nCascade :',i)

            n = 0
            i+=1
            falserate = []; detrate = []
            F[i] = F[i-1]; D[i] = D[i-1]
            len_P = len(self.P_paths); len_N = len(self.N_paths)
            thres = len_N + len_P
            strClf = []
            while F[i] > (F[i-1]*f) or D[i] < (D[i-1]*d):
                start = time.time()
                n += 1
                print("\nNumber of feature (weak classfifier) :", n)

                # Train adaboost
                thres -= 1
                if abs(thres) > len_N + len_P:
                    break
                if n == 1:
                    prevCond, self.features, self.features_list = ada.adaboost(len_P, len_N, self.features, self.features_list, thres)
                else:
                    prevCond, self.features, self.features_list = ada.adaboost(len_P, len_N, self.features, self.features_list, thres, prevCond=prevCond)
                strClf = prevCond[0]

                # Evaluate F and D
                clf = [strClf]
                posResult = [self.predict(cv2.imread(p,0), clf) for p in self.P_paths]
                negResult = [self.predict(cv2.imread(n,0), clf) for n in self.N_paths]
                y_true = np.hstack((np.zeros(len_N), np.ones(len_P)))
                y_pred = np.array(negResult + posResult)
                TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[1,0]).ravel()
                print("TP", TP, "TN", TN, "FP", FP, "FN", FN)
    
                F[i] = FP/(FP+TN)
                D[i] = TP/(TP+FN)
                # falserate.append(FP/(FP+TN))
                # detrate.append(TP/(TP+FN))
                # F[i] = np.prod(falserate)
                # D[i] = np.prod(detrate)
                
                print("F[i]", F[i], "   F[i-1]*f", F[i-1]*f)
                print("D[i]", D[i], "   D[i-1]*d", D[i-1]*d)
                print("Time :", time.time()-start)

            if len(strClf) == 0:
                break

            # Append strong classifier to cascade classifier
            self.cascadeClf.append(strClf)

            # Leaving negative images that is wrongly detected (False Positive)
            self.N_paths = [self.N_paths[i] for i in range(len_N) if negResult[i]==1]
            negResult = np.array(negResult)
            remove_indices = np.argwhere(negResult == 0)
            self.features = [i for j, i in enumerate(self.features) if j not in remove_indices]
