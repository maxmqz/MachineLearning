import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

def main():
    red_wine = pd.read_csv('..\MachineLearning\winequality_red.csv', header = 0)
    y = red_wine['class']
    X = red_wine.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    rfc = RandomForestClassifier(random_state=42, 
                                n_jobs=-1, 
                                max_depth=5, 
                                n_estimators=100)
                                
    rfc.fit(X_train, y_train)
    
    print("Accuracy:", rfc.score(X_test, y_test))

if __name__ == "__main__":
    main()