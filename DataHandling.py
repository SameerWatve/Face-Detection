import numpy as np
import cv2
import os
from numpy import array

train_size = 1000
test_size = 100

train_folder_face = 'train_face_images/'
train_folder_nonface = 'train_nonface_images/'
test_folder_face = 'test_face_images/'
test_folder_nonface = 'test_nonface_images/'

files_train_face = os.listdir(train_folder_face)
files_train_nonface = os.listdir(train_folder_nonface)
files_test_face = os.listdir(test_folder_face)
files_test_nonface = os.listdir(test_folder_nonface)


def get_image_train(index, typ):
    dat = []
    if typ == 'face':
        dat = cv2.imread(train_folder_face + files_train_face[index])
    elif typ == 'nonface':
        dat = cv2.imread(train_folder_nonface + files_train_nonface[index])
    if dat is None:
        return array([0]*100)
    dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float64')/255
    return dat


def get_image_CV(index, typ):
    dat = []
    if typ == 'face':
        dat = cv2.imread(test_folder_face + files_test_face[index])
    elif typ == 'nonface':
        dat = cv2.imread(test_folder_nonface + files_test_nonface[index])
    if dat is None:
        return array([0]*100)
    dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float64')/255
    return dat


def load_train_data(typ):
    X_train = []
    for i in range(train_size):
        if typ == 'face':
            dat = get_image_train(i, 'face')
            dat = dat.flatten()
            X_train.append(dat)
        elif typ == 'nonface':
            dat = get_image_train(i, 'nonface')
            dat = dat.flatten()
            X_train.append(dat)
    X_train = array(X_train)
    return X_train
#    Each ROW is an OBSERVATION and COLUMN a FEATURES


def load_CV_data(typ):
    X_test = []
    for i in range(test_size):
        if typ == 'face':
            dat = get_image_CV(i, 'face')
            dat = dat.flatten()
            X_test.append(dat)
        elif typ == 'nonface':
            dat = get_image_CV(i, 'nonface')
            dat = dat.flatten()
            X_test.append(dat)
    X_test = array(X_test)
    return X_test


def get_MC(X):
    meanX = np.mean(X, axis=0)
    covar = np.zeros((100, 100), dtype='float64')
    np.fill_diagonal(covar, np.cov(X, rowvar=False).diagonal())
    return [meanX, covar]


