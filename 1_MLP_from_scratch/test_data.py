from src.utils import load_data

X_train, Y_train, X_test, Y_test = load_data(
    "data/fashion-mnist_train.csv", 
    "data/fashion-mnist_test.csv"
)

print("--- بررسی ابعاد ماتریس‌ها ---")
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape :", X_test.shape)
print("Y_test shape :", Y_test.shape)

print("\n--- بررسی بازه مقادیر پیکسل‌ها ---")
print("Max pixel value:", X_train.max())
print("Min pixel value:", X_train.min())