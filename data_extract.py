import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
import src.haar_extractor as haar
import src.hypothesis as hypo

def prepare_data(P_path, N_path, destination_path, imsize):
    allImgFeatures = []
    computed_features = []

    # Haar Feature Extraction
    P_paths = [os.path.join(P_path, s) for s in sorted(os.listdir(P_path))]
    N_paths = [os.path.join(N_path, s) for s in sorted(os.listdir(N_path))]
    training_paths = N_paths + P_paths

    haar_dir = os.path.join(destination_path, 'haar_features')
    if not os.path.isdir(haar_dir):
        os.mkdir(haar_dir)

    for i, filename in enumerate(tqdm(training_paths)):
        im = cv2.imread(filename, 0)
        im = cv2.resize(im, imsize)
        
        # Hitung integral image
        im = haar.integralImage(im)
        
        # Cari haar fitur
        features = haar.getFeatures(im, step=2, featureTypes=("type_one",))
        allImgFeatures.append(features)
        
        # Save feature
        df = pd.DataFrame(features)
        path = os.path.join(haar_dir, 'features_{}.csv'.format(i))
        df.to_csv(path, index=False)
    print('Haar Extracted...')

    # Compute Hypothesis
    hypo_dir = os.path.join(destination_path, 'hypothesis')
    if not os.path.isdir(hypo_dir):
        os.mkdir(hypo_dir)
    for thres in tqdm(range(len(allImgFeatures))):
        hipotesis = hypo.hypothesis(thres, allImgFeatures)
        filename = "hypo_{}.npy".format(thres)
        path = os.path.join(hypo_dir, filename)
        np.save(path, hipotesis)
    print('Hypothesis calculated...')