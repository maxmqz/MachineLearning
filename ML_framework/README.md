# MachineLearning con el uso de un Framework

* Archivo a revisar:    main.py
* Modelo:               Bosque Aleatorio
* Libreria utilizada:   sklearn.ensemble (RandomForestClassifier)
* Dataset:              winequality_red.csv
* Métrica de desempeño: Accuracy (0.834)

* Predicciones de Prueba (primeras 5):

    * valores de entrada (hiper-parámetros): 
        * random_state=42
        * n_estimators=1000
        * criterion="entropy"
        * max_depth=None
        * bootstrap=True 

    * valores de entrada del modelo (features):
        * fixed acidity
        * volatile acidity
        * citric acid     
        * residual sugar
        * chlorides
        * free sulfur dioxide
        * total sulfur dioxide
        * density
        * pH
        * sulphates
        * alcohol
    valor a predecir: class (0, 1)
    ---
    0. Expected: 0 --> Predicted: 0
    * fixed acidity:        8.8
    * volatile acidity:     0.41
    * citric acid:          0.64     
    * residual sugar:       2.2
    * chlorides:            0.093
    * free sulfur dioxide:  9.0
    * total sulfur dioxide: 42.0
    * density:              0.9986                
    * pH:                   3.54
    * sulphates:            0.66
    * alcohol:              10.5
    ---
    1. Expected: 1 --> Predicted: 1
    * fixed acidity:        8.7
    * volatile acidity:     0.63
    * citric acid:          0.28     
    * residual sugar:       2.7
    * chlorides:            0.096
    * free sulfur dioxide:  17.0
    * total sulfur dioxide: 69.0
    * density:              0.99734                
    * pH:                   3.26
    * sulphates:            0.63
    * alcohol:              10.2
    ---
    2. Expected: 1 --> Predicted: 1
   * fixed acidity:        10.4
    * volatile acidity:     0.34
    * citric acid:          0.58     
    * residual sugar:       3.7
    * chlorides:            0.174
    * free sulfur dioxide:  6.0
    * total sulfur dioxide: 16.0
    * density:              0.99700                
    * pH:                   3.19
    * sulphates:            0.70
    * alcohol:              11.3
    ---
    3. Expected: 1 --> Predicted: 1
    * fixed acidity:        7.1
    * volatile acidity:     0.46
    * citric acid:          0.20     
    * residual sugar:       1.9
    * chlorides:            0.077
    * free sulfur dioxide:  28.0
    * total sulfur dioxide: 54.0
    * density:              0.99560                
    * pH:                   3.37
    * sulphates:            0.64
    * alcohol:              10.4
    ---
    4. Expected: 1 --> Predicted: 1
    * fixed acidity:        7.1
    * volatile acidity:     0.39
    * citric acid:          0.12     
    * residual sugar:       2.1
    * chlorides:            0.065
    * free sulfur dioxide:  14.0
    * total sulfur dioxide: 24.0
    * density:              0.99252                
    * pH:                   3.30
    * sulphates:            0.53
    * alcohol:              13.3
    ---

