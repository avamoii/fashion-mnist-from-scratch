import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_data
from src.layers import DenseLayer
from src.activations import ReLU, Softmax
from src.network import NeuralNetwork
from src.losses import CrossEntropyLoss

# بارگذاری داده‌های آموزش و آزمون
print("Loading datasets...")
X_train, Y_train, X_test, Y_test = load_data(
    "data/fashion-mnist_train.csv", 
    "data/fashion-mnist_test.csv"
)

# ساخت معماری شبکه
layers = [
    DenseLayer(784, 256),
    ReLU(),
    DenseLayer(256, 128),
    ReLU(),
    DenseLayer(128, 64),
    ReLU(),
    DenseLayer(64, 10),
    Softmax()
]

model = NeuralNetwork(layers)
loss_fn = CrossEntropyLoss()

epochs = 30
learning_rate = 0.01
batch_size = 512
m = X_train.shape[1]

# متغیرهایی برای ذخیره روند آموزش جهت رسم نمودار
history_loss = []
history_acc = []

print("Starting training...")

for epoch in range(epochs):
    permutation = np.random.permutation(m)
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]
    
    for i in range(0, m, batch_size):
        X_batch = X_train_shuffled[:, i:i+batch_size]
        Y_batch = Y_train_shuffled[:, i:i+batch_size]
        
        predictions = model.forward(X_batch)
        loss = loss_fn.forward(predictions, Y_batch)
        
        dZ = loss_fn.backward(predictions, Y_batch)
        model.backward(dZ)
        model.update_params(learning_rate)
        
    predictions_test = model.forward(X_test)
    predictions_classes = np.argmax(predictions_test, axis=0)
    true_classes = np.argmax(Y_test, axis=0)
    
    accuracy = np.mean(predictions_classes == true_classes) * 100
    
    # ذخیره داده‌ها برای نمودار
    history_loss.append(loss)
    history_acc.append(accuracy)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss:.4f} | Test Accuracy: {accuracy:.2f}%")

# ==========================================
# بخش رسم نمودار و تست روی یک عکس واقعی
# ==========================================
print("\nTraining complete! Generating plots...")

# ساخت یک پنجره با ۳ بخش برای نمودارها
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# نمودار اول: روند کاهش خطا (Loss)
axs[0].plot(range(1, epochs+1), history_loss, marker='o', color='red')
axs[0].set_title('Training Loss over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# نمودار دوم: روند افزایش دقت (Accuracy)
axs[1].plot(range(1, epochs+1), history_acc, marker='o', color='green')
axs[1].set_title('Test Accuracy over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].grid(True)

# نمودار سوم: تست مدل روی یک عکس تصادفی از داده‌های Test
random_idx = np.random.randint(0, X_test.shape[1])
random_image_flat = X_test[:, random_idx:random_idx+1]
true_label_idx = np.argmax(Y_test[:, random_idx])

# پیش‌بینی مدل برای این یک عکس
pred_prob = model.forward(random_image_flat)
pred_label_idx = np.argmax(pred_prob)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

image_2d = random_image_flat.reshape(28, 28)

axs[2].imshow(image_2d, cmap='gray')
axs[2].set_title(f"True Label: {class_names[true_label_idx]}\nAI Predicted: {class_names[pred_label_idx]}")
axs[2].axis('off')

plt.tight_layout()
plt.show()