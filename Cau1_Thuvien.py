import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Đọc dữ liệu
data = pd.read_csv(r"C:\Dethi_python\BaiTap1\LinearRegression_DienNang.csv")

print("Các cột trong file csv:")
print(data.columns.tolist())

X = data[['Số giờ sử dụng thiết bị']]  # Đầu vào
y = data['Lượng điện tiêu thụ (kWh)']  # Đầu ra

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)

# Hệ số góc và hệ số chặn
slope = model.coef_[0]
intercept = model.intercept_
print(f"Hệ số góc (slope): {slope}")
print(f"Hệ số chặn (intercept): {intercept}")

# Vẽ biểu đồ đường hồi quy
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, model.predict(X), color='red', label='Đường hồi quy')
plt.xlabel('Số giờ sử dụng thiết bị')
plt.ylabel('Lượng điện tiêu thụ (kWh)')
plt.legend()
plt.title('Hồi quy tuyến tính: Lượng điện tiêu thụ theo số giờ sử dụng')
plt.grid(True)
plt.show()

# Dự đoán với số giờ sử dụng 2
predict_2 = model.predict(pd.DataFrame({'Số giờ sử dụng thiết bị': [2]}))
print("Dự đoán lượng điện tiêu thụ với 2h sử dụng:", round(predict_2[0], 2))

# Nhập số giờ sử dụng từ bàn phím
time = float(input("Nhập số giờ sử dụng thiết bị: "))
predicted_output = model.predict(pd.DataFrame({'Số giờ sử dụng thiết bị': [time]}))
print(f"Dự đoán lượng điện tiêu thụ với {time} giờ: {round(predicted_output[0], 2)} kWh")