class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, dZ):
        # حرکت رو به عقب در لایه‌ها
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                dZ = layer.backward(dZ)

    def update_params(self, learning_rate):
        # آپدیت کردن وزن‌ها با استفاده از گرادیان‌ها
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db