import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
dulieu = pd.read_csv('dataset.csv')
dulieu.head()

# Hiển thị dữ liệu
sns.countplot(x='target', data=dulieu, palette='bwr')
plt.show()

# hiên thị tất cả các cột
dulieu.columns

# Tách dữ liệu
x= dulieu.iloc[:,0:13].values
y= dulieu['target'].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

# Tạo mô hình
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Dự đoán
y_pred = classifier.predict(x_test)

# Đánh giá mô hình
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Độ chính xác của mô hình là: ",accuracy_score(y_test, y_pred))

import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# dự đoán một giá trị mới
new_prediction = classifier.predict(st_x.transform(np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])))
print(new_prediction)
