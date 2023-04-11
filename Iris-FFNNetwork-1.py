import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

class FNN:
    def __init__(self, n_inputs, hidden_sizes=[2], n_outputs=1):
        self.nx = n_inputs
        self.ny = n_outputs
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny]

        # 创建两个字典用来存储权重和偏置
        self.W = {}
        self.B = {}
        for i in range(self.nh+1):
            self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
            self.B[i+1] = np.zeros((1, self.sizes[i+1]))

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)
  
    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        # H0 means input X
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
            self.H[i+1] = self.sigmoid(self.A[i+1])
        self.A[self.nh+1] = np.matmul(self.H[self.nh], self.W[self.nh+1]) + self.B[self.nh+1]
        self.H[self.nh+1] = self.softmax(self.A[self.nh+1])
        return self.H[self.nh+1]
  
    def grad_sigmoid(self, x):
        return x*(1-x) 

    def cross_entropy(self, label, pred):
        y1 = np.multiply(pred, label)
        y1 = y1[y1!=0]
        y1 = -np.log(y1)
        y1 = np.mean(y1)

        return y1
    
    def grad(self, x, y):
        self.forward_pass(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.nh + 1
        self.dA[L] = (self.H[L] - y)
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)
            self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1]))
    
    def fit(self, X, Y, epochs=10, learning_rate=1, initialise=True, display_loss=False):
    
    # initialise w, b
        if initialise:
            for i in range(self.nh+1):
                self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
                self.B[i+1] = np.zeros((1, self.sizes[i+1]))
      
        if display_loss:
            loss = {}
    
        for e in tqdm(range(epochs), total=epochs, unit="epoch"):
            dW = {}
            dB = {}

            for i in range(self.nh+1):
                dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
                dB[i+1] = np.zeros((1, self.sizes[i+1]))

            for x, y in zip(X, Y):
                self.grad(x, y)
                for i in range(self.nh+1):
                    dW[i+1] += self.dW[i+1]
                    dB[i+1] += self.dB[i+1]
        
            m = X.shape[1]
            for i in range(self.nh+1):
                self.W[i+1] -= learning_rate * dW[i+1] / m
                self.B[i+1] -= learning_rate * dB[i+1] / m
      
            if display_loss:
                Y_pred = self.predict(X)
                loss[e] = self.cross_entropy(Y, Y_pred)
    
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('CE')
            plt.show()
            plt.savefig('Iris-loss.png')
      
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()


if __name__ == "__main__":
    X_train = np.loadtxt('./Exam/train/x.txt', dtype='str').astype(np.float64)
    X_test = np.loadtxt('./Exam/test/x.txt', dtype='str').astype(np.float64)
    Y_train = np.loadtxt('./Exam/train/y.txt', dtype='str').astype(np.int64)
    Y_test = np.loadtxt('./Exam/test/y.txt', dtype='str').astype(np.int64)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # one-hot encoding
    enc = OneHotEncoder()
    y_train_ohenc = enc.fit_transform(np.expand_dims(Y_train,1)).toarray()
    y_test_ohenc = enc.fit_transform(np.expand_dims(Y_test,1)).toarray()
    print(y_train_ohenc.shape, y_test_ohenc.shape)

    ffnet = FNN(2, [2, 3], 2)
    ffnet.fit(X_train, y_train_ohenc, epochs=1000, learning_rate=0.005, display_loss=True)

    Y_pred_train = ffnet.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train,1)

    Y_pred_test = ffnet.predict(X_test)
    Y_pred_test = np.argmax(Y_pred_test,1)

    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_test = accuracy_score(Y_pred_test, Y_test)

    print("Training accuracy-Iris", round(accuracy_train, 2))
    print("Testing accuracy-Iris", round(accuracy_test, 2))

    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "blue", "green"])
    plt.figure()
    plt.scatter(X_train[:,0], X_train[:,1], c=Y_pred_train, cmap=my_cmap, s=15*(np.abs(Y_pred_train-Y_train)+.2))
    plt.savefig('Iris-boundary.png')