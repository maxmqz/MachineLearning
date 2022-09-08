import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    red_wine = pd.read_csv('winequality_red.csv', header = 0)
    y = red_wine['class']
    X = red_wine.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


if __name__ == "__main__":
    main()