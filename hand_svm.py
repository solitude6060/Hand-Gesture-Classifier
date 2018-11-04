from skimage import exposure
#exposure可用來調整影像中像素的強度
from skimage import feature
import cv2
from os import listdir
from os.path import isfile, isdir, join
from sklearn import svm, metrics
import numpy as np
from sklearn.model_selection import cross_val_score
train_path = "CSL/training/"
test_path = "CSL/test"
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report


def readfile(filepath):
   imageList=[]
   features=[]
   # 取得所有檔案與子目錄名稱
   files = listdir(filepath)
   # 以迴圈處理
   for f in files:
       # 產生檔案的絕對路徑
       fullpath = join(filepath, f)
       # 判斷 fullpath 是檔案還是目錄
       #print(fullpath)
       image = cv2.imread(fullpath)
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       imageList.append(gray)
       (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(64, 64),cells_per_block=(2,2), transform_sqrt=True, visualise=True)
       #print(len(H))
       features.append(H)
#        cv2.imshow("123",gray)
#        cv2.waitKey(0)
   return imageList

def getFeature(imageList):
   features=[]
   for i in imageList:
        (H, hogImage) = feature.hog(i, orientations=9, pixels_per_cell=(32,32),cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        #print(H)
        features.append(H)
#         #調整影像強度範為介於0~255之間（rescale_intensity可將影像的像素強度進行壓縮或放大）
#         hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#         #數值類型更改為Unsigned Integer 8 bits
#         hogImage = hogImage.astype("uint8")
#         #顯示HOG視覺圖
#         cv2.imshow("HOG Image", hogImage)
#         imageList = readfile("training/")
   return features

def getFeature_moments(imageList):
   feature=[]
   for i in imageList:
       feature.append(cv2.HuMoments(cv2.moments(i)).flatten())
   return feature

def getLabel(path):
    LabelList = []

    files = listdir(path)
    for f in files:
        LabelList.append(f[0])

    return LabelList



#imageList = readfile(path)
#feature = getFeature_moments(imageList)
y_train = np.array(getLabel(train_path))
imageTrainList = readfile(train_path)
x_train = np.array(getFeature_moments(imageTrainList))
x_train_hog = np.array(getFeature(imageTrainList))

print("Feature : ", x_train.shape, "Label : ",  y_train.shape)
#np.savetxt('x_train_hu.csv', x_train, fmt='%.8e')
#np.savetxt('y_train_hu.csv', y_train, fmt="%s")
#np.save('x_train_hu', x_train, fmt='%.18e')
#np.save('y_train_hu', y_train, fmt="%s")
np.save('x_train_hog', x_train_hog)

y_test = np.array(getLabel(test_path))
imageTestList = readfile(test_path)
x_test = np.array(getFeature_moments(imageTestList))
x_test_hog = np.array(getFeature(imageTestList))
print("HOG : ", len(x_test_hog[0]))
print("HU : ", len(x_test[0]))
print("Test Feature : ", x_test.shape, "Label : ",  y_test.shape)
#np.savetxt('x_test_hu.csv', x_test, fmt='%.8e')
#np.savetxt('y_test_hu.csv', y_test, fmt='%s')
#np.save('x_test_hu', x_test, fmt='%.18e')
#np.save('y_test_hu', y_test, fmt='%s')
np.save('x_test_hog', x_test_hog)

target_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)

print("---------------kernel=linear----------------")

classifier = svm.SVC(gamma=0.001, kernel="linear")
classifier.fit(x_train_hog, y_train)
y_predict = classifier.predict(x_test_hog)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, y_predict)))
#print("Classification report for AUC :\n" % (metrics.auc(y_test, y_predict)))

scores = cross_validate(classifier, x_train_hog, y_train, cv=5, scoring='accuracy')

print("cross_validation scores")
print(scores)


result = classifier.predict(x_test_hog)
print(classification_report(y_test, result, target_names=target_names))

print("---------------kernel=rbf(default)----------------")

classifier_rbf = svm.SVC(gamma=0.001)
classifier_rbf.fit(x_train_hog, y_train)
y_predict = classifier_rbf.predict(x_test_hog)
print("Classification report for classifier %s:\n%s\n" % (classifier_rbf, metrics.classification_report(y_test, y_predict)))
#print("Classification report for AUC :\n" % (metrics.auc(y_test, y_predict)))

scores = cross_validate(classifier_rbf, x_train_hog, y_train, cv=5, scoring='accuracy')

print("cross_validation scores")
print(scores)


result = classifier_rbf.predict(x_test_hog)
print(classification_report(y_test, result, target_names=target_names))


