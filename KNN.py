import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

# kiểm tra giá trị k tôt nhất
error = []
# Calculating error for K values between 1 and 30
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Tỉ lệ lỗi theo giá trị K')
plt.xlabel('Giá trị K')
plt.ylabel('Trung bình sai số')
plt.show()
print("Giá trị k tốt nhất là: ",error.index(min(error))+1)

# Tạo mô hình
knn = KNeighborsClassifier(n_neighbors= 7, p=2, metric='euclidean')
knn.fit(x_train, y_train)

# Dự đoán
y_pred = knn.predict(x_test)
print(y_pred)

# Đánh giá mô hình
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Độ chính xác của mô hình là: ",accuracy_score(y_test, y_pred))

# Lưu mô hình
import pickle
pickle.dump(knn, open('model.pkl','wb'))

# Dự đoán một giá trị mới
model = pickle.load(open('model.pkl','rb'))
# a = input("Age: ")
# b = input("Sex: ")
# c = input("cp: ")
# d = input("trestbps: ")
# e = input("chol: ")
# f = input("fbs: ")
# g = input("restecg: ")
# h = input("thalach: ")
# i = input("exang: ")
# j = input("oldpeak: ")
# k = input("slope: ")
# l = input("ca: ")
# m = input("thal: ")
# print(model.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m]]))
# Ví dụ: 63,1,3,145,233,1,0,150,0,2.3,0,0,1

# Kết quả: 1

print(model.predict([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]))

