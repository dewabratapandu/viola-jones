import numpy as np
import math

def adaboost(P, N, features, features_list, thres, prevCond=None):
      
    if prevCond is None:
        strClf = []
        # initialize weight for each positive and negative image in 1D array
        weight = np.hstack((np.ones(N)*1/(2*N), np.ones(P)*1/(2*P)))
    else:
        strClf, weight = prevCond

    # positif and negatif label in 1D array
    label = np.hstack((np.zeros(N), np.ones(P)))
    # number of images and features
    num_images = len(features)
    num_features = len(features[0])
    # weight normalization
    weight = weight / np.sum(weight)


    # calculate hypothesis
    f = np.array(features)
    hipo = np.where(f < f[thres, :], 1, 0)


    # calculate error
    error_classification = np.sum(weight * np.abs(label-np.transpose(hipo)), axis=1)

    
    # best feature (feature with minimum error)
    min_error_index = np.argmin(error_classification)
    best_error = error_classification[min_error_index]
    print('best error =', best_error)

    # calculate alpha
    a = alpha(best_error)

    # weight update
    b = beta(best_error)
    weight = np.where(hipo[:,min_error_index]==label, weight*b, weight)

    # save feature and alpha
    best_feature = list(features_list[min_error_index])
    best_feature.extend([features[thres][min_error_index], a])
    strClf.append(best_feature)

    # remove best feature from features list
    for i in range(num_images):
        features[i].pop(min_error_index)
    features_list.pop(min_error_index)

    prevCond = (strClf, weight)
    return prevCond, features, features_list

def alpha(error):
    return math.log(1/beta(error))

def beta(error): 
    return error/(1 - error)
