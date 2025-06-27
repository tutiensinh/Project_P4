import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
df = pd.read_csv(r'C:\Dethi_python\BaiTap1\KNN_CayTrong.csv')

# Giả sử cột cuối là "Lớp mục tiêu"
X = df.iloc[:, :-1].values      # Các đặc trưng
y = df.iloc[:, -1].values       # Nhãn

# Mã hóa nhãn nếu là chữ
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Huấn luyện mô hình KNN với k = 5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y_encoded)

# Dự đoán cho mẫu: {5.67, 874, 34.5, 8.6}
sample = np.array([[5.67, 874, 34.5, 8.6]])
pred = model.predict(sample)
pred_label = label_encoder.inverse_transform(pred)
print("Dự đoán nhóm cho mẫu [5.67, 874, 34.5, 8.6]:", pred_label[0])

# Nhập từ bàn phím
input_features = []
feature_names = ['pH', 'Độ ẩm', 'Nhiệt độ', 'Chất dinh dưỡng']
for feature in feature_names:
    value = float(input(f"Nhập {feature}: "))
    input_features.append(value)

# Dự đoán từ đầu vào
input_array = np.array([input_features])
pred_input = model.predict(input_array)
print("Phân nhóm cho dữ liệu nhập vào:", label_encoder.inverse_transform(pred_input)[0])

# Vẽ biểu đồ (ví dụ đơn giản cho 2 đặc trưng đầu tiên)
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='viridis', label=y)
plt.xlabel('Đặc trưng 1 (pH)')
plt.ylabel('Đặc trưng 2 (Độ ẩm)')
plt.title('Biểu đồ phân nhóm cây trồng')
plt.colorbar(label='Lớp mục tiêu (mã hóa)')
plt.grid(True)
plt.show()
