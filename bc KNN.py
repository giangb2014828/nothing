from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import numpy as np
iris_dataset = load_iris()

import pandas as pd
data = pd.read_csv('marketing.dat', header=None, names=['Sex', 'MaritalStatus', 'Age', 'Education', 'Occupation', 'YearsInSf', 'DualIncome', 'HouseholdMembers', 'Under18', 'HouseholdStatus', 'TypeOfHome', 'EthnicClass', 'Language', 'Income'])
data.dropna(inplace=True)
data.fillna(data.mean(), inplace=True)
dulieu_X=data.iloc[:,0:-1]
dulieu_Y=data.iloc[:,-1]
X = data.drop('Income', axis=1)
y = data['Income']
kf = KFold(n_splits=100,shuffle=True,random_state=None)
for idTrain, idTest in kf.split(data):
    X_train = dulieu_X.iloc[idTrain,]
    X_test = dulieu_X.iloc[idTest,]
    y_train = dulieu_Y.iloc[idTrain]
    y_test = dulieu_Y.iloc[idTest]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
MoHinhDT = KNeighborsClassifier(n_neighbors=7)
MoHinhDT.fit(X_train, y_train)
Y_Dudoan = MoHinhDT.predict(X_test)
print("Ket qua du doan:",Y_Dudoan)
accuracy = accuracy_score(y_test,Y_Dudoan)
print("Độ chính xác trên tập kiểm tra: {:.2f}%".format(accuracy * 100))

# Đánh giá hiệu suất của mô hình
accuracy = accuracy_score(y_test,Y_Dudoan)
print("Độ chính xác trên tập kiểm tra: {:.2f}%".format(accuracy * 100))


#danh gia cho 8 phan tu
X_test1 = X_test[0:8]
Y_test1 = y_test[0:8]
print("8 phan tu dau tien:",X_test1)
Y_Dudoan = MoHinhDT.predict(X_test1)
print("Ket qua du doan:",Y_Dudoan)
Ketqua_Dochinhxac = accuracy_score(Y_test1,Y_Dudoan)*100
print("Do chinh xac la:",Ketqua_Dochinhxac)

prec, rec,fsco,sup = precision_recall_fscore_support(Y_test1,Y_Dudoan)
print("Precision:",np.mean(prec))
print("Recall",np.mean(rec))
print("fscore",np.mean(fsco))
print("sup",np.mean(sup))