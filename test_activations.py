import numpy as np
from src.activations import ReLU, Softmax

# ۱. تست فیلتر ReLU
print("--- تست تابع فعال‌ساز ReLU ---")
relu = ReLU()
Z_sample = np.array([[3.5], [-1.2], [0.0], [8.4], [-5.5]])
A_relu = relu.forward(Z_sample)
print("ورودی خام (Z):")
print(Z_sample)
print("\nخروجی فیلترشده (A_relu):")
print(A_relu)

# ۲. تست تابع Softmax (۳ کلاس برای ۱ تصویر نمونه)
print("\n--- تست تابع Softmax ---")
softmax = Softmax()
Z_logits = np.array([[2.0], [1.0], [0.1]])
A_probs = softmax.forward(Z_logits)
print("ورودی خام یا Logits:")
print(Z_logits)
print("\nتوزیع احتمالات خروجی (A_probs):")
print(A_probs)
print("\nمجموع احتمالات (باید دقیقاً برابر با ۱ شود):", np.sum(A_probs))