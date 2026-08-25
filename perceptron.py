import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class Perceptron:
  def __init__(self, learning_rate = 0.01, n_iters = 1000): 
    self.lr = learning_rate
    self.n_iters = n_iters
    self.acc_func = self._unit_step_func
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape

    #init (initialise) weights
    self.weights = np.zeros(n_features)
    self.bias = 0

    y_ = np.array([1 if i > 0 else 0 for i in y])

    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_predicted = self.acc_func(linear_output)

        update = self.lr * (y_[idx] - y_predicted)
        self.weights += update * x_i
        self.bias += update
              
  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    y_predicted = self.acc_func(linear_output)
    return y_predicted
  
  
  def _unit_step_func(self, x):
    return np.where(x>=0, 1, 0) #where x>=0 give 1 otherwise give 0


#Now using it on a mnist database
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X,y = mnist.data, mnist.target.astype(int)

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X,y, text_size=0.2, random_state=42)
perceptrons = {}

for digit in range(10):
  print(f"Number {digit}")

y_train_binary = np.where(y_train == digit,1,0)

p=Perceptron(learning_rate=0.01, n_iters=5)
p.fit(X_train, y_train_binary)
