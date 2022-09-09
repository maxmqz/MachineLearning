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
    
    rfc = RandomForestClassifier(random_state=42, 
                                 n_estimators=1000, # Number of Trees
                                 criterion="entropy", # quality de split para cada arbol
                                 max_depth=None, # max deepth of tree
                                 bootstrap=True # use different samples from dataset for each tree
                                 )
    # entropy yields better results although gini is faster
    rfc.fit(X_train, y_train)
    #plot_tree(rfc, X_test) 
    print("Accuracy:", rfc.score(X_test, y_test))

    predictions = rfc.predict(X_test)
    pd.set_option('display.max_columns', None)
    print(X_test.head(5))
    c = 0
    for i, j in zip(y_test, predictions):
        print(f"Expected: {i} --> Predicted: {j}")
        c += 1
        if c == 5:
            break
    

# code below used for visualizing a single tree(42)
def plot_tree(rfc, X_test):  
    #estimators_ return a list of DecisionTreeClassifier (trees)
    _tree = rfc.estimators_[42]
    print(tree.export_text(_tree))
    plt_tree = plt.figure(figsize=(150,150))
    tree.plot_tree(_tree, filled=True, feature_names=X_test.keys(), class_names="class")
    plt_tree.savefig("ML_framework/decision_tree.png")

if __name__ == "__main__":
    main()