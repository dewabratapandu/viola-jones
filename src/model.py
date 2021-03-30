import csv
import src.hypothesis as hypo
import src.haar_extractor as haar

def loadModel(filename):
    cascadeClf = []
    strClf = []
    with open(filename) as csvfile:
        model = csv.reader(csvfile, delimiter=' ')
        for line in model:
            if(line[0] == 'end'):
                cp = strClf.copy()
                cascadeClf.append(cp)
                strClf.clear()
            elif(line is None):
                continue
            else:
                for i in range(1, 4):
                    line[i] = int(line[i])
                for i in range(5, 6):
                    line[i] = float(line[i])
                strClf.append(line)
    print(cascadeClf)
    return cascadeClf

def cascadeClassifier(im, model):
    for strClf in model:
        ahx = 0 #merupakan sigma hasil perkalian alfa * hx
        sumAlfa = 0 #merupakan sigma alfa
        for weakClf in strClf:
            fx = haar.computeFeature(im, weakClf[0], int(weakClf[1]), int(weakClf[2]), int(weakClf[3]), int(weakClf[4]))
            hx = hypo.h(fx, int(weakClf[6]))
            alfa = float(weakClf[5])
            ahx += alfa * hx
            sumAlfa += alfa
        hasil = 1 if ahx >= 0.5*sumAlfa else 0
        if(hasil == 0):
            break
    return hasil