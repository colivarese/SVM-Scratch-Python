from SVM import SVM
from utils import loadData

train_x, train_y, test_x, test_y = loadData('./dataset/gender_classification_v7.csv', 'gender')

svm = SVM()

svm.fit(train_x, train_y)

svm.predictTest(test_x, test_y)