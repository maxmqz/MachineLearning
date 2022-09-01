import pandas as pd # utilizado para importar dataset
from sklearn import preprocessing # utilizado para normalizar dataset
from sklearn.metrics import accuracy_score

# funcion para realizar las predicciones
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

# Estimando coeficientes utilizando gradiente descendiente
def gradiente_d(train, l_rate, n_epoch):
    # coef a generar dependiendo de las variables
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef) # Realizar predicción
            error = yhat - row[-1] # Obtener el error 
            sum_error += error**2 # Suma de error al cuadrado
            # Obtener nuevo coeficiente para la siguiente iteración
            coef[0] = coef[0] - l_rate * error
            # para cada var en la fila
            for i in range(len(row)-1):
                # a partir del segundo coeficiente obtener uno nuevo
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef


red_wine = pd.read_csv('winequality_red.csv', header = 0)
y = red_wine['class']
red_wine = red_wine.drop('class', axis=1)
print(red_wine.head())

x = red_wine.values

# escalar a entre 1 y 0
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = list(x_scaled)

l_rate = 0.01
n_epoch = 100
coef = gradiente_d(df, l_rate, n_epoch)

# Realizar prediccion y guardar en array de y_pred_line
y_pred_line = []
for row in df[:5]:
    yhat = predict(row, coef)
    y_pred_line.append(yhat)
    #print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))
