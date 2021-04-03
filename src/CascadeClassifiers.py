import os
import csv
import time
import numpy as np
import cv2
import src.adaboost as ada
import src.haar_extractor as haar
import src.hypothesis as hypo
from tqdm import tqdm

class CascadeClassifier:
    def __init__(self):
        self.cascadeClf = []

    def save(self, filename):
        # Mempersiapkan file model.txt
        model = open(filename, "w")
        model.write("")

        # Write hasil pelatihan cascade ke file txt
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
                    line[0] = int(line[0])
                    line[2] = int(line[2])
                    line[3] = int(line[3])
                    line[4] = int(line[4])
                    line[5] = int(line[5])
                    line[6] = float(line[6])
                    strClf.append(line)

    def feature_extracting(self, P_path, N_path, imsize=(24,24), featureTypes = ("type_one", "type_two", "type_three", "type_four", "type_five"), step=1):
        self.imsize = imsize
        self.P_path = P_path
        self.N_path = N_path
        self.allImgFeatures = []

        # Haar Feature Extraction
        self.P_paths = [os.path.join(self.P_path, s) for s in sorted(os.listdir(self.P_path))]
        self.N_paths = [os.path.join(self.N_path, s) for s in sorted(os.listdir(self.N_path))]
        training_paths = self.N_paths + self.P_paths

        for i, filename in enumerate(tqdm(training_paths)):
            im = cv2.imread(filename, 0)
            im = cv2.resize(im, self.imsize)
            
            # Hitung integral image
            im = haar.integralImage(im)
            
            # Cari haar fitur
            features = haar.getFeatures(im, featureTypes=featureTypes, step=step)
            self.allImgFeatures.append(features)
            
        print('Haar Extracted...')

    def predict(self, src, cascadeClf=None, imsize=None):
        if cascadeClf == None:
            cascadeClf = self.cascadeClf
        if imsize == None:
            imsize = self.imsize
        im = cv2.imread(src, 0)
        im = cv2.resize(im, imsize)
        im = haar.integralImage(im)
        for strClf in cascadeClf:
            ahx = 0 #merupakan sigma hasil perkalian alfa * hx
            sumAlfa = 0 #merupakan sigma alfa
            for weakClf in strClf:
                fx = haar.computeFeature(im, weakClf[1], weakClf[2], weakClf[3], weakClf[4], weakClf[5])
                # hx = hypo.h(fx, int(weakClf[6]))
                hx = 1 if fx < weakClf[0] else 0
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
        i = 0 # index cascade

        while F[i] > Ftarget:
            if(i >= max_cascade):
                break
            print('\nCascade ke-',i)

            n = 0
            i+=1
            F[i] = F[i-1]; D[i] = D[i-1]
            falserate = []; detrate = []
            len_P = len(self.P_paths); len_N = len(self.N_paths)
            thres = len_N + len_P - 1
            while F[i] > (F[i-1] * f):
                start = time.time()
                n += 1
                print("\nJumlah fitur :", n)
                # Latih adaboost
                if n == 1:
                    prevCond, self.allImgFeatures = ada.adaboost(len_P, len_N, self.allImgFeatures, thres)
                else:
                    prevCond, self.allImgFeatures = ada.adaboost(len_P, len_N, self.allImgFeatures, thres, prevCond=prevCond)
                strClf = prevCond[0]

                # Evaluasi F dan D
                clf = [strClf]
                posResult = [self.predict(p, clf) for p in self.P_paths]
                negResult = [self.predict(n, clf) for n in self.N_paths]
                TP, FN, TN, FP = self.confusionMatrix(len_P, len_N, posResult, negResult)
                print("TP", TP, "TN", TN, "FP", FP, "FN", FN)
    
                # F[i] = FP/(FP+TN)
                # D[i] = TP/(TP+FN)
                falserate.append(FP/(FP+TN))
                detrate.append(TP/(TP+FN))
                F[i] = np.prod(falserate)
                D[i] = np.prod(detrate)
                
                print("F[i]", F[i], "   F[i-1]*f", F[i-1]*f)
                print("D[i]", D[i], "   D[i-1]*d", D[i-1]*d)
                print("Time :", time.time()-start)

                thres -= 1

            # Tambahkan strong classifier ini ke cascade classifier
            self.cascadeClf.append(strClf)

            # Menyisakan citra negatif yang salah dideteksi (False Positive)
            self.N_paths = self.eliminateTrueNeg(self.N_paths, negResult)
            negResult = np.array(negResult)
            remove_indices = np.argwhere(negResult == 0)
            self.allImgFeatures = [i for j, i in enumerate(self.allImgFeatures) if j not in remove_indices]

    def confusionMatrix(self, P, N, posResult, negResult):
        TP = sum([1 for h in posResult if h==1])
        FN = P - TP

        TN = sum([1 for h in negResult if h==0])
        FP = N - TN

        return TP, FN, TN, FP

    def eliminateTrueNeg(self, Nimg, hasilClf):
        NimgBaru = list()
        for i in range(len(hasilClf)):
            if(hasilClf[i] == 1):
                NimgBaru.append(Nimg[i])
        return NimgBaru