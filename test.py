import tkinter as tk
from tkinter import filedialog, messagebox, Button, Label, simpledialog, Frame
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2

# 初始化主視窗
root = tk.Tk()
root.title("圖形化修圖程式")

# 初始化圖像變數
img = None
img_label = None
original_img = None  # 初始化原始圖像變數
img_history = []  # 初始化歷史圖像列表

# 更新圖像預覽的函數
def update_preview():
    global img, img_label
    try:
        if img:
            tk_img = ImageTk.PhotoImage(img)
            if img_label is None:
                img_label = Label(root, image=tk_img)
                img_label.image = tk_img
                img_label.pack(side="left", padx=10, pady=10)
            else:
                img_label.configure(image=tk_img)
                img_label.image = tk_img
    except Exception as e:
        messagebox.showerror("錯誤", f"無法更新預覽: {e}")

# 回到上一步驟的功能
def undo():
    global img, img_history
    if img_history:
        img = img_history.pop()
        update_preview()
    else:
        messagebox.showwarning("警告", "沒有歷史可以回退。")

# 加載圖像功能
def load_image():
    global img, original_img, img_history
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            original_img = img.copy()
            img_history = []  # 清空歷史
            update_preview()
    except Exception as e:
        messagebox.showerror("錯誤", f"加載圖像失敗: {e}")


# 儲存圖像的功能
def save_image():
    global img
    try:
        if img:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                img.save(file_path)
                messagebox.showinfo("成功", "圖像儲存成功！")
    except Exception as e:
        messagebox.showerror("錯誤", f"儲存圖像失敗: {e}")

# 旋轉圖像的功能
def rotate_image():
    global img, img_history
    if img:
        # 讓使用者輸入旋轉角度
        angle = simpledialog.askfloat("輸入", "請輸入旋轉角度（度）", minvalue=-360, maxvalue=360)
        if angle is not None:  # 確保角度已輸入
            img_history.append(img.copy())  # 在修改前保存當前圖像的副本
            img = img.rotate(angle, expand=True)
            update_preview()


# 轉換成灰階的功能
def convert_to_grayscale():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img = ImageOps.grayscale(img)
        update_preview()

# 重置圖像到原始狀態的功能
def reset_image():
    global img, original_img
    if original_img is not None:  # 檢查 original_img 是否已被賦值
        img = original_img.copy()
        update_preview()

# 儲存圖像的功能
def save_image():
    if img:
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            img.save(file_path)

# 應用二值化效果的功能
def binary_image():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img = img.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
        update_preview()

# 應用 Canny 邊緣偵測的功能
def canny_edge_detection():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img.convert('L'))
        edges = cv2.Canny(img_array, 100, 200)
        img = Image.fromarray(edges)
        update_preview()

# 應用雙邊濾波的功能
def bilateral_filter():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img)
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        img = Image.fromarray(img_array)
        update_preview()

# 應用侵蝕效果的功能
def erode_image():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img)
        kernel = np.ones((5, 5), np.uint8)
        img_array = cv2.erode(img_array, kernel, iterations=1)
        img = Image.fromarray(img_array)
        update_preview()

# 應用影像銳化的功能
def sharpen_image():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        sharp_filter = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        img = img.filter(sharp_filter)
        update_preview()

# 應用中值濾波的功能
def median_filter():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img = img.filter(ImageFilter.MedianFilter(size=3))
        update_preview()

# 其他功能...

# 創建按鈕並將它們排列在頂部橫向一列
button_frame = Frame(root)
button_frame.pack(side="top", fill="x", expand=True)


# 膨脹效果
def dilate_image():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img)
        kernel = np.ones((5,5), np.uint8)
        img_array = cv2.dilate(img_array, kernel, iterations=1)
        img = Image.fromarray(img_array)
        update_preview()

# 伽瑪矯正
def gamma_correction():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        img_array = np.array(img)
        img = Image.fromarray(cv2.LUT(img_array, table))
        update_preview()

# 均值濾波
def mean_filter():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img)
        img_array = cv2.blur(img_array, (5,5))
        img = Image.fromarray(img_array)
        update_preview()

# 影像負片
def negative_image():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img = ImageOps.invert(img)
        update_preview()

# Beta 矯正
def beta_correction():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        beta = 1.5  # 假設 Beta 值
        img_array = np.array(img, dtype=np.float64)
        img_array = np.clip(beta * img_array, 0, 255)
        img = Image.fromarray(np.uint8(img_array))
        update_preview()

# 高斯濾波
def gaussian_blur():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        update_preview()

# Sobel 邊緣偵測
def sobel_edge_detection():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img.convert('L'))
        sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.hypot(sobelx, sobely)
        max_sobel = np.max(sobel)
        if max_sobel == 0:
            sobel_normalized = sobel
        else:
            sobel_normalized = sobel / max_sobel * 255
        img = Image.fromarray(np.uint8(sobel_normalized))
        update_preview()

# 椒鹽雜訊
def salt_and_pepper_noise():
    global img, img_history
    if img:
        img_history.append(img.copy())  # 在修改前保存當前圖像的副本
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            row, col, ch = img_array.shape
        else:
            row, col = img_array.shape
            ch = 1
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img_array)
        # Salt mode
        num_salt = np.ceil(amount * img_array.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
        for i in range(int(num_salt)):
            x = coords[0][i]
            y = coords[1][i]
            if ch == 1:
                out[x, y] = 255
            else:
                out[x, y, :] = [255, 255, 255]
        # Pepper mode
        num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
        for i in range(int(num_pepper)):
            x = coords[0][i]
            y = coords[1][i]
            out[x, y] = 0
        img = Image.fromarray(out)
        update_preview()

# 創建按鈕並將它們排列在頂部橫向一列
button_frame = Frame(root)
button_frame.pack(side="top", fill="x", expand=True)

Button(button_frame, text="加載圖像", command=load_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="回到上一步驟", command=undo).pack(side="left", padx=5, pady=5)
Button(button_frame, text="重製照片", command=reset_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="儲存圖像", command=save_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="旋轉圖像", command=rotate_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="灰階", command=convert_to_grayscale).pack(side="left", padx=5, pady=5)
Button(button_frame, text="二值化", command=binary_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="Canny 邊緣偵測", command=canny_edge_detection).pack(side="left", padx=5, pady=5)
Button(button_frame, text="雙邊濾波", command=bilateral_filter).pack(side="left", padx=5, pady=5)
Button(button_frame, text="侵蝕", command=erode_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="影像銳化", command=sharpen_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="中值濾波", command=median_filter).pack(side="left", padx=5, pady=5)
Button(button_frame, text="膨脹", command=dilate_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="伽瑪矯正", command=gamma_correction).pack(side="left", padx=5, pady=5)
Button(button_frame, text="均值濾波", command=mean_filter).pack(side="left", padx=5, pady=5)
Button(button_frame, text="影像負片", command=negative_image).pack(side="left", padx=5, pady=5)
Button(button_frame, text="Beta 矯正", command=beta_correction).pack(side="left", padx=5, pady=5)
Button(button_frame, text="高斯濾波", command=gaussian_blur).pack(side="left", padx=5, pady=5)
Button(button_frame, text="Sobel 邊緣偵測", command=sobel_edge_detection).pack(side="left", padx=5, pady=5)
Button(button_frame, text="椒鹽雜訊", command=salt_and_pepper_noise).pack(side="left", padx=5, pady=5)

# 運行主循環
root.mainloop()