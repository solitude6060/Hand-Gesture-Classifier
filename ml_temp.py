from skimage import exposure    #exposure可用來調整影像中像素的強度
from skimage import feature
import cv2
import os
from os import listdir, getcwd
from os.path import isfile, isdir, join
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score # K折交叉验证模块
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

def readfile(filepath):
    imageList=[]
    features=[]
    targes = []
    # 取得所有檔案與子目錄名稱
    
    files = listdir(filepath)
    # 以迴圈處理
    for f in files:
        
        fname, ftype = os.path.splitext(f)
        if ftype != ".jpg":
            continue
        #產生檔案的絕對路徑
        fullpath = join(filepath, f)
        # 判斷 fullpath 是檔案還是目錄
        #print(fullpath)
        image = cv2.imread(fullpath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageList.append(gray)
        H = feature.hog(gray, orientations=9, pixels_per_cell=(100, 100),cells_per_block=(1,1), transform_sqrt=True,feature_vector=True)
        #features.append(cv2.HuMoments(cv2.moments(gray)).flatten())
        #print(len(H))
        features.append(H)
        targes.append(f[0])
    return features,targes

def getFeature(imageList):
    features=[]
    for i in imageList:
        (H, hogImage) = feature.hog(i, orientations=9, pixels_per_cell=(32,32),cells_per_block=(1, 1), transform_sqrt=True, visualise=True)
#         print(H)
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
    features=[]
    for i in imageList:
        features.append(cv2.HuMoments(cv2.moments(i)).flatten())
    return features

target_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#current_dir = getcwd()
features,targes = readfile("./CSL/training/")
features_t,targes_t = readfile("./CSL/test/")
scoring = ['precision_macro', 'recall_macro']

print("~~~~~~~~~~~~~~~linear kernel~~~~~~~~~~~~~~~~~")
clf = svm.SVC(kernel='linear')
clf.fit(features, targes)
scores = cross_validate(clf, features,targes, scoring=scoring, cv=5, return_train_score=False)

print(sorted(scores.keys()))
print("cross_validation scores")
print(scores)
for s in scores:
    print(s)

result = clf.predict(features_t)
#print(clf.score(features_t,targes_t))
print(classification_report(targes_t, result, target_names=target_names))


print("~~~~~~~~~~~~~~~RBF kernel~~~~~~~~~~~~~~~~~")
clf2 = svm.SVC()
clf2.fit(features, targes)
scores = cross_validate(clf2,features,targes, cv=5, scoring='accuracy')
print("cross_validation scores")
print(scores)
for s in scores:
    print(s)


result = clf2.predict(features_t)
#print(clf.score(features_t,targes_t))
print(classification_report(targes_t, result, target_names=target_names))


print("~~~~~~~~~~~~~~~LinearSVC~~~~~~~~~~~~~~~~~")
clf3 = svm.LinearSVC()
clf3.fit(features, targes)
scores = cross_validate(clf3,features,targes, cv=5, scoring='accuracy')
print("cross_validation scores")
print(scores)
for s in scores:
    print(s)



result = clf3.predict(features_t)
#print(clf.score(features_t,targes_t))
print(classification_report(targes_t, result, target_names=target_names))
