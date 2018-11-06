from skimage import exposure
from skimage import feature
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report

import os
from os import listdir
from os.path import isfile, isdir, join

import cv2#opencv

train_path = "./CSL/training/"
test_path = "./CSL/test/"

label_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
scoring = ['precision_macro', 'recall_macro', 'accuracy']


def readimage(path):
    image_list = []
    y_list = [] #label list

    files = listdir(path)

    for file in files:
        fname, ftype = os.path.splitext(file)
        #avoid to read temp file, ex:.DS_store
        if ftype != ".jpg":
            continue

        fullpath = path+file
        #print(fullpath)
        image = cv2.imread(fullpath)
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_list.append(grayimg)
        y_list.append(file[0])#first vocab of file

    return image_list, y_list

def getHogFeature(image_list):
    feature_list = []
    for image in image_list:
        img_feature = feature.hog(image, orientations=9, pixels_per_cell=(64, 64),cells_per_block=(1,1), transform_sqrt=True,feature_vector=True)
        feature_list.append(img_feature)

    
    return feature_list

def runSVM_RBF(train_feature_list, train_target_list, test_feature_list, test_target_list, label_names, scoring):
    accuracy = 0
    recall = 0
    print("--------------RBF-----------------")
    clf = svm.SVC(kernel='rbf')
    clf.fit(train_feature_list, train_target_list)
    scores = cross_validate(clf, train_feature_list, train_target_list, scoring=scoring, cv=5)
    print(sorted(scores.keys()))
    print("cross_validation scores")
    #print(scores)
    for key, value in scores.items():
        print(key, value)
        if key == "test_accuracy":
            for i in  value:
                accuracy += i
        if key == "test_recall_macro":
            for i in  value:
                recall += i
    print("Avg accuracy : ", accuracy/5)
    print("Avg recall : ", recall/5)
    result = clf.predict(test_feature_list)
    print(classification_report(test_target_list, result, target_names=label_names))
    print("--------------END-----------------")
    
    return

def runSVM_linear(train_feature_list, train_target_list, test_feature_list, test_target_list, label_names, scoring):
    accuracy = 0
    recall = 0
    print("--------------svm with linear-----------------")
    clf = svm.SVC(kernel='linear')
    clf.fit(train_feature_list, train_target_list)
    scores = cross_validate(clf, train_feature_list, train_target_list, scoring=scoring, cv=5)
    print(sorted(scores.keys()))
    print("cross_validation scores")
    #print(scores)
    for key, value in scores.items():
        print(key, value)
        if key == "test_accuracy":
            for i in  value:
                accuracy += i
        if key == "test_recall_macro":
            for i in  value:
                recall += i
    print("Avg accuracy : ", accuracy/5)
    print("Avg recall : ", recall/5)

    result = clf.predict(test_feature_list)
    print(classification_report(test_target_list, result, target_names=label_names))
    print("--------------END-----------------")
    return

def runLinearSVC(train_feature_list, train_target_list, test_feature_list, test_target_list, label_names, scoring):
    accuracy = 0
    recall = 0
    print("--------------linearSVC-----------------")
    clf = svm.LinearSVC()
    clf.fit(train_feature_list, train_target_list)
    scores = cross_validate(clf, train_feature_list, train_target_list, scoring=scoring, cv=5)
    print(sorted(scores.keys()))
    print("cross_validation scores")
    #print(scores)
    for key, value in scores.items():
        print(key, value)
        if key == "test_accuracy":
            for i in  value:
                accuracy += i
        if key == "test_recall_macro":
            for i in  value:
                recall += i
    print("Avg accuracy : ", accuracy/5)
    print("Avg recall : ", recall/5)

    result = clf.predict(test_feature_list)
    print(classification_report(test_target_list, result, target_names=label_names))
    print("--------------END-----------------")
    return


train_image, y_train = readimage(train_path)
test_image, y_test = readimage(test_path)

x_train = getHogFeature(train_image)
x_test = getHogFeature(test_image)

print(len(x_train[0]))


runSVM_RBF(x_train, y_train, x_test, y_test, label_names, scoring)
runSVM_linear(x_train, y_train, x_test, y_test, label_names, scoring)
runLinearSVC(x_train, y_train, x_test, y_test, label_names, scoring)
