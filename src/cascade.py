import os
import time
import numpy as np
import cv2
import pandas as pd
import src.adaboost as ada
import src.haar_extractor as haar
import src.hypothesis as hypo

def trainCascade(model_path, P_paths, N_paths, imsize, Ftarget, Dtarget, f, d, features_paths, hypothesis_paths, computed_features, max_cascade=50):
    """ Argumen --> Pimg = dataset gambar positif, Nimg = dataset gambar negatif
                    Ftarget = float overall false positive rate, Dtarget = float detection rate
                    f = max acceptable false positive rate, d = min acceptable detection rate
                    features = list 3D berisikan hasil hitung haar-features. Dimensi 1 terkait gambar ke-. Dimensi 2 terkait fitur ke-. Dimensi 3 terkiat detail fitur (hitungan, type, x, y, w, h)
        Return --> cascade = list berisikan sejumlah strong classifier (adaboost)."""
    
    # Mempersiapkan file model.txt
    model = open(model_path, "w")
    model.write(""); model.close()

    cascade = []
    P_paths = [os.path.join(P_paths, s) for s in os.listdir(P_paths)]
    N_paths = [os.path.join(N_paths, s) for s in os.listdir(N_paths)]
    features = [os.path.join(features_paths, s) for s in os.listdir(features_paths)]
    hypothesis = [os.path.join(hypothesis_paths, s) for s in os.listdir(hypothesis_paths)]
    print(len(os.listdir(hypothesis_paths)))
    F = np.ones(max_cascade, dtype=float)
    D = np.ones(max_cascade, dtype=float)
    i = 0 #index cascade

    # print('Jumlah gambar tersisa : ', len(features))
    remove_indices = []
    # while F[i] > Ftarget or D[i] < Dtarget:
    while F[i] > Ftarget:
        if(i >= max_cascade):
            break
        print('\nCascade ke-',i)

        n = 0
        i+=1
        F[i] = F[i-1]
        D[i] = D[i-1]
        falseposrate = []; detectrate = []; last_F = 9999
        # while F[i] > (F[i-1] * f) or D[i] < (D[i-1]*d):
        while F[i] > (F[i-1] * f):
            start = time.time()
            n += 1
            print("\nJumlah fitur :", n)
            # Latih adaboost
            if n == 1:
                prevCond = ada.adaboost(len(P_paths), len(N_paths), features, hypothesis, remove_indices=remove_indices)
            else:
                prevCond = ada.adaboost(len(P_paths), len(N_paths), features, hypothesis, prevCond=prevCond)
            strClf = prevCond[0]
            # print(strClf)

            # Evaluasi F dan D
            posResult = [testCascade(p, imsize, strClf) for p in P_paths]
            negResult = [testCascade(n, imsize, strClf) for n in N_paths]
            TP, FN, TN, FP = confusionMatrix(len(P_paths), len(N_paths), posResult, negResult)
            print("TP", TP, "TN", TN, "FP", FP, "FN", FN)
            
            # falseposrate.append(FP / (FP+TN))
            # F[i] = np.prod(np.array(falseposrate))
            # detectrate.append(TP/(TP+FN))
            # D[i] = np.prod(np.array(detectrate))
            F[i] = FP/(FP+TN)
            D[i] = TP/(TP+FN)
            
            print("F[i]", F[i], "   F[i-1]*f", F[i-1]*f)
            print("D[i]", D[i], "   D[i-1]*d", D[i-1]*d)
            print("Time :", time.time()-start)

            if F[i] > last_F:
                strClf.pop()
                break
            last_F = F[i]

        # Tambahkan strong classifier ini ke cascade classifier
        cascade.append(strClf)
        #Write hasil pelatihan cascade ke file txt
        print("\nMenyimpan Cascade Classifier pada model.txt")
        model = open(model_path, "a+")
        for weakClf in strClf:
            model.write(weakClf[1]); model.write(' ')
            model.write(str(weakClf[2])); model.write(' ')
            model.write(str(weakClf[3])); model.write(' ')
            model.write(str(weakClf[4])); model.write(' ')
            model.write(str(weakClf[5])); model.write(' ')
            model.write(str(weakClf[6])); model.write(' ')
            model.write(str(weakClf[7]))
            model.write("\n")
        model.write("end of cascade\n")
        model.close()

        # Evaluasi cascade detector saat ini
        """ dipake untuk menggugurkan gambar2 yg mudah diklasifikasikan negatif (dalam kasus ini, hilangkan features)
        sehingga hanya menyisakan gambar negatif yg sulit dideteksi saja"""
        N_paths = eliminateTrueNeg(N_paths, negResult)
        negResult = np.array(negResult)
        remove_indices = np.argwhere(negResult == 0)
        # features = [i for j, i in enumerate(features) if j not in remove_indices]
        print('P = {}, N = {}'.format(len(P_paths), len(N_paths)))

def testCascade(im_path, imsize, strClf):
    """ Argumen --> imgs = dataset gambar yang akan diuji
                    strClf = strong classifier yang akan diuji
        Return --> hasilClf = hasil klasifikasi pada im dengan strClf"""
    im = cv2.imread(im_path, 0)
    im = cv2.resize(im, imsize)
    im = haar.integralImage(im)
    ahx = 0 #merupakan sigma hasil perkalian alfa * hx
    sumAlfa = 0 #merupakan sigma alfa
    for weakClf in strClf:
        fx = haar.computeFeature(im, weakClf[1], int(weakClf[2]), int(weakClf[3]), int(weakClf[4]), int(weakClf[5]))
        hx = hypo.h(fx, int(weakClf[7]))
        alfa = float(weakClf[6])
        ahx += alfa * hx
        sumAlfa += alfa
    return 1 if ahx >= 0.5*sumAlfa else 0

def confusionMatrix(P, N, posResult, negResult):
    TP = sum([1 for h in posResult if h==1])
    FN = P - TP

    TN = sum([1 for h in negResult if h==0])
    FP = N - TN

    return TP, FN, TN, FP

def eliminateTrueNeg(Nimg, hasilClf):
    """ Argumen --> hasilClf = hasil klasifikasi pada dataset gambar negatif
        Return --> Nimg = dataset gambar negatif baru"""
    NimgBaru = list()
    for i in range(len(hasilClf)):
        if(hasilClf[i] == 1):
            NimgBaru.append(Nimg[i])
    return NimgBaru