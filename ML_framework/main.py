from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# model
from sklearn.ensemble import RandomForestClassifier
# model tree
from sklearn import tree

def main():
    red_wine = pd.read_csv('..\MachineLearning\winequality_red.csv', header = 0)
    y = red_wine['class']
    X = red_wine.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    '''
    hiperparametros:
        - criterion: quality of split
    '''
    rfc = RandomForestClassifier(random_state=42, 
                                 n_estimators=1000, #Number of Trees
                                 criterion="entropy")
    # entropy yields better results although gini is faster
    rfc.fit(X_train, y_train)
    print("Accuracy:", rfc.score(X_test, y_test))
    
    #estimators_ return a list of DecisionTreeClassifier (trees)
    '''
    _tree = rfc.estimators_[42]
    
    print(tree.export_text(_tree))
    plt_tree = plt.figure(figsize=(150,150))
    tree.plot_tree(_tree, filled=True, feature_names=X_test.keys(), class_names="class")
    plt_tree.savefig("ML_framework/decision_tree.png")
    '''
if __name__ == "__main__":
    main()