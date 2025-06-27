# Đọc dữ liệu thủ công từ file CSV
def doc_du_lieu(filename):
    so_cong_nhan = []
    san_luong = []
    with open(filename, 'r', encoding='utf-8') as f:
        next(f)  # Bỏ qua dòng tiêu đề
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    so_cong_nhan.append(x)
                    san_luong.append(y)
                except:
                    continue  # Bỏ qua dòng không hợp lệ
    return so_cong_nhan, san_luong

# Tính trung bình
def tinh_trung_binh(ds):
    return sum(ds) / len(ds)

# Tính hệ số hồi quy tuyến tính
def hoi_quy_tuyen_tinh(x, y):
    n = len(x)
    mean_x = tinh_trung_binh(x)
    mean_y = tinh_trung_binh(y)

    tu = 0
    mau = 0
    for i in range(n):
        tu += (x[i] - mean_x) * (y[i] - mean_y)
        mau += (x[i] - mean_x) ** 2
    slope = tu / mau
    intercept = mean_y - slope * mean_x
    return slope, intercept

# Dự đoán sản lượng
def du_doan(slope, intercept, x):
    return slope * x + intercept

# Chương trình chính
def main():
    file = "C:\Dethi_python\BaiTap1\LinearRegression_CongNghiep.csv"
    x, y = doc_du_lieu(file)

    if not x or not y:
        print("Không có dữ liệu hợp lệ.")
        return

    slope, intercept = hoi_quy_tuyen_tinh(x, y)

    print(f"Hệ số góc (slope): {slope}")
    print(f"Hệ số chặn (intercept): {intercept}")

    # Dự đoán với 35 công nhân
    du_doan_35 = du_doan(slope, intercept, 35)
    print(f"Dự đoán sản lượng/ngày với 35 công nhân: {du_doan_35:.2f} sản phẩm")

    # Nhập từ bàn phím
    try:
        x_nhap = float(input("Nhập số công nhân: "))
        du_doan_nhap = du_doan(slope, intercept, x_nhap)
        print(f"Dự đoán sản lượng/ngày: {du_doan_nhap:.2f} sản phẩm")
    except:
        print("Dữ liệu nhập không hợp lệ.")

if __name__ == "__main__":
    main()
