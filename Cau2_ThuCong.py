import pandas as pd
import math
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv(r"C:\Dethi_python\BaiTap1\KNN_CayTrong.csv")
data.columns = data.columns.str.strip()  # Xử lý khoảng trắng thừa

# Chia X, y
features = [
    "Độ pH của đất",
    "Lượng mưa trung bình hàng năm (milimét)",
    "Nhiệt độ trung bình hàng năm (độ C)",
    "Số giờ nắng trung bình mỗi ngày"
]
X = data[features].values.tolist()
y = data["Lớp mục tiêu"].tolist()

# -------------------------------
# Chuẩn hóa dữ liệu thủ công (min-max scaling)
def normalize(data):
    transposed = list(zip(*data))
    normalized = []
    for i in range(len(data)):
        row = []
        for j in range(len(data[0])):
            col = transposed[j]
            min_val = min(col)
            max_val = max(col)
            if max_val != min_val:
                norm_val = (data[i][j] - min_val) / (max_val - min_val)
            else:
                norm_val = 0
            row.append(norm_val)
        normalized.append(row)
    return normalized

X_norm = normalize(X)

# -------------------------------
# Hàm tính khoảng cách Euclid
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# -------------------------------
# Hàm dự đoán KNN thủ công
def knn_predict(X_train, y_train, input_point, k=5):
    input_norm = normalize([input_point])[0]  # chuẩn hóa input theo min/max của toàn bộ tập
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], input_norm)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    classes = [label for _, label in k_neighbors]
    prediction = max(set(classes), key=classes.count)
    return prediction

# -------------------------------
# Dự đoán thử một mẫu
new_point = [5.67, 874, 34.5, 8.6]
pred = knn_predict(X_norm, y, new_point, k=5)
print("Kết quả phân nhóm cho dữ liệu mới:", pred)

# -------------------------------
# Nhập từ bàn phím
print("Nhập 4 thông số:")
user_point = [float(input(f"Thông số {i+1}: ")) for i in range(4)]
user_pred = knn_predict(X_norm, y, user_point, k=5)
print("Kết quả phân nhóm cho dữ liệu nhập vào:", user_pred)

# -------------------------------
# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
for label in set(y):
    subset = data[data["Lớp mục tiêu"] == label]
    plt.scatter(
        subset["Độ pH của đất"],
        subset["Lượng mưa trung bình hàng năm (milimét)"],
        label=label
    )
plt.xlabel("Độ pH của đất")
plt.ylabel("Lượng mưa trung bình hàng năm (mm)")
plt.title("Biểu đồ phân bố cây trồng")
plt.legend()
plt.grid(True)
plt.show()
