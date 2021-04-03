import numpy as np
import cv2

def integralImage(im):
    """ Argumen --> im = numpy array normal image
        Return --> im_itg = numpy array integral image"""
    im_itg = im.copy()
    im_itg = np.zeros_like(im, dtype='uint8').astype(int)
    # Menggunakan Summed Area Table
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if (x > 0) and (y > 0):
                im_itg[x, y] = im[x, y] + im_itg[x-1, y] + \
                    im_itg[x, y-1] - im_itg[x-1, y-1]
            elif x > 0:
                im_itg[x, y] = im[x, y] + im_itg[x-1, y]
            elif y > 0:
                im_itg[x, y] = im[x, y] + im_itg[x, y-1]
            else:
                im_itg[x, y] = im[x, y]
    return im_itg

def computeArea(A, B, C, D):
    return ((D+A) - (B-C))

def computeFeature(im, featureType, x, y, w, h):
    if(featureType == "type_one"):
        putih = computeArea(im[y,x], im[y, x+w], im[y+h, x], im[y+h, x+w])
        hitam = computeArea(im[y+h, x], im[y+h, x+w], im[y+2*h, x], im[y+2*h, x+w])
        return (hitam-putih)
    elif(featureType == "type_two"):
        putih = computeArea(im[y,x], im[y, x+w], im[y+h, x], im[y+h, x+w])
        hitam = computeArea(im[y, x+w], im[y, x+2*w], im[y+h, x+w], im[y+h, x+2*w])
        return (hitam-putih)
    elif(featureType == "type_three"):
        putih1 = computeArea(im[y,x], im[y, x+w], im[y+h, x], im[y+h, x+w])
        hitam = computeArea(im[y, x+w], im[y, x+2*w], im[y+h, x+w], im[y+h, x+2*w])
        putih2 = computeArea(im[y, x+2*w], im[y, x+3*w], im[y+h, x+2*w], im[y+h, x+3*w])
        return (hitam-putih1-putih2)
    elif(featureType == "type_four"):
        putih1 = computeArea(im[y,x], im[y, x+w], im[y+h, x], im[y+h, x+w])
        hitam = computeArea(im[y+h, x], im[y+h, x+w], im[y+2*h, x], im[y+2*h, x+w])
        putih2 = computeArea(im[y+2*h, x], im[y+2*h, x+w], im[y+3*h, x], im[y+3*h, x+w])
        return (hitam-putih1-putih2)
    elif(featureType == "type_five"):
        putih1 = computeArea(im[y,x], im[y, x+w], im[y+h, x], im[y+h, x+w])
        hitam1 = computeArea(im[y, x+w], im[y, x+2*w], im[y+h, x+w], im[y+h, x+2*w])
        putih2 = computeArea(im[y+h, x+w], im[y+h, x+2*w], im[y+2*h, x+w], im[y+2*h, x+2*w])
        hitam2 = computeArea(im[y+h, x], im[y+h, x+w], im[y+2*h, x], im[y+2*h, x+w])
        return (hitam1+hitam2-putih1-putih2)

def getFeatures(im, featureTypes = ("type_one", "type_two", "type_three", "type_four", "type_five"), step=1):
    """ Argumen --> im = numpy array integral image
        Return --> features = list/array 1D berisikan fitur-fitur. Dimensi 1 terkait fitur keberapa"""
    # type_one : edge feature horizontal
    # type_two : edge feature vertical
    # type_three : line feature vertical
    # type_four : line feature horizontal
    # type_five : four rectangle feature

    winHe, winWi = im.shape

    height_Limit = {"type_one": (int)(winHe/2 - 1),
                    "type_two": (int)(winHe - 1),
                    "type_three": (int)(winHe - 1),
                    "type_four": (int)(winHe/3 - 1),
                    "type_five": (int)(winHe/2 - 1)}
    width_Limit = {"type_one": (int)(winWi - 1),
                   "type_two": (int)(winWi/2 - 1),
                   "type_three": (int)(winWi/3 - 1),
                   "type_four": (int)(winWi - 1),
                   "type_five": (int)(winWi/2 - 1)}

    allFeatureTypes = ("type_one", "type_two", "type_three", "type_four", "type_five")
    type_removed = list(set(allFeatureTypes) - set(featureTypes))
    for key in type_removed:
        del height_Limit[key]
        del width_Limit[key]

    features = []
    for types in featureTypes:
        for h in range(1, height_Limit[types], step):
            for w in range(1, width_Limit[types], step):
                if w == 1 and h == 1:
                    continue

                if types == "type_one":
                    x_limit = winWi - w
                    y_limit = winHe - 2*h
                    for x in range(1, x_limit, step):
                        for y in range(1, y_limit, step):
                            fitur = computeFeature(im, "type_one", x, y, w, h)
                            features.append((fitur, types, x,  y, w, h))

                elif types == "type_two":
                    x_limit = winWi - 2*w
                    y_limit = winHe - h
                    for x in range(1, x_limit, step):
                        for y in range(1, y_limit, step):
                            fitur = computeFeature(im, "type_two", x, y, w, h)
                            features.append((fitur, types, x, y, w, h))

                elif types == "type_three":
                    x_limit = winWi - 3*w
                    y_limit = winHe - h
                    for x in range(1, x_limit, step):
                        for y in range(1, y_limit, step):
                            fitur = computeFeature(im, "type_three", x, y, w, h)
                            features.append((fitur, types, x, y, w, h))

                elif types == "type_four":
                    x_limit = winWi - w
                    y_limit = winHe - 3*h
                    for x in range(1, x_limit, step):
                        for y in range(1, y_limit, step):
                            fitur = computeFeature(im, "type_four", x, y, w, h)
                            features.append((fitur, types, x, y, w, h))

                elif types == "type_five":
                    x_limit = winWi - 2*w
                    y_limit = winHe - 2*h
                    for x in range(1, x_limit, step):
                        for y in range(1, y_limit, step):
                            fitur = computeFeature(im, "type_five", x, y, w, h)
                            features.append((fitur, types, x, y, w, h))
    return features