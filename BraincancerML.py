import pandas as pd

dataset_path = 'braincancer.csv'
X = pd.read_csv(dataset_path)

X.dropna()

y=X.pop('y')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=0) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf.fit(X_train,y_train)

y_predSVC = clf.predict(X_test)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_predLogistic = classifier.predict(X_test)


from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=9)
KNN.fit(X_train,y_train)

pred = KNN.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_predLogistic)
print(cm)
acc1 = accuracy_score(y_test, y_predLogistic)
print(acc1)
## To print the accuracy score for Logistic Regression

cm1 = confusion_matrix(y_test, y_predSVC)
print(cm1)
acc2 = accuracy_score(y_test,y_predSVC)
print(acc2)
## To print the accuracy score for SVC

cm2 = confusion_matrix(y_test, pred)
print(cm2)
acc3 = accuracy_score(y_test,pred)
print(acc3)
## To print the accuracy score for KNN
