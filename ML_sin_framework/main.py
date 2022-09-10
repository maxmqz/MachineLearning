import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class logisticRegression:
    def __init__(self,alpha=0.001,n_epoch=1000):
        self.alpha = alpha
        self.n_epoch = n_epoch
        self.coef = None
        self.bias = None # intercept

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)
        self.bias = 0

        # gradiente descendente
        for _ in range(self.n_epoch):
            prediction = np.dot(X,self.coef) + self.bias
            y_predicted = self._sigmoid(prediction)
            
            # en X.T 'T' es de transpose 
            # Calcular nuevos coeficientes
            b = (1/n_samples) * np.dot(X.T,(y_predicted-y))
            x = (1/n_samples) * np.sum(y_predicted-y)
            
            # se puede realizar esta operacion ya que son numpy arrays
            self.coef -= self.alpha * b
            self.bias -= self.alpha * x 

    def predict(self,X):
        prediction = np.dot(X,self.coef) + self.bias
        y_predicted = self._sigmoid(prediction)
        # convertir predicciones a binario usando list comprehension
        y_predicted_b = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_b

    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


def main():
    red_wine = pd.read_csv('..\MachineLearning\winequality_red.csv', header = 0)
    y = red_wine['class']
    X = red_wine.drop('class', axis=1)
    '''
    for col_name in X: 
        print(col_name)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    lr = logisticRegression(alpha=0.0001,n_epoch=1000)
    lr.fit(X_train, y_train)

    predictions = lr.predict(X_test)
    
    
    pd.set_option('display.max_columns', None)
    print(X_test.head(5))
    c = 0
    for i, j in zip(y_test, predictions):
        print(f"Expected: {i} --> Predicted: {j}")
        c += 1
        if c == 5:
            break
    
    print("Accuracy:", accuracy(y_test, predictions))

def accuracy(y_true,y_pred):
    # if true, add one, else add 0, luego divide entre la longitud
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


if __name__ == "__main__":
    main()
