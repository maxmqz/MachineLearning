# Maximiliano Martinez Marquez
# A01251527


import pandas as pd 

columns = ["Car_Name","Year","Selling_Price",
            "Present_Price","Kms_Driven",
            "Fuel_Type","Seller_Type",
            "Transmission","Owner"] 

cars = pd.read_csv('car_data.csv', header = 0)
#cars.drop(cars.loc[cars['Present_Price']>80].index, inplace=True)
#cars.reset_index(drop=True, inplace = True)
cars.head()

x = cars['Present_Price']
y = cars['Selling_Price']

x.info() 
#y.info()

import matplotlib.pyplot as plt

plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.plot(x, y, 'bo')
plt.show()
# 92 value outlier

mean_x = sum(x) / float(len(x))
var_x = sum([(i-mean_x)**2 for i in x])
print("x stats: mean=%.3f variance=%.3f" % (mean_x, var_x))

mean_y = sum(y) / float(len(y))
var_y = sum([(i-mean_y)**2 for i in y])
print("y stats: mean=%.3f variance=%.3f" % (mean_y, var_y))

covar = 0.0
for i in range(len(x)):
	covar += (x[i] - mean_x) * (y[i] - mean_y)
print('Covariance: %.3f' % (covar))

# y = mx + b
# E = (1/n)* sum(y_i-y) 
b1 = covar / var_x
b0 = mean_y - b1 * mean_x

print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))

h = lambda x, theta: theta[0] + theta[1]*x
theta = [b1,b0]
y_new = []
for x_i in x:
    y_new.append(h(x_i,theta))

error_total = 0
for i in range(len(x)):
    error_total += (y[i] - (h(x[i], theta)))**2
error_total / float(len(x))
print("Error Total:", error_total)

plt.xlabel('Present Price')
plt.ylabel('Selling Price')
plt.plot(x, y, 'ob')
plt.plot(x, y_new, '-r')
plt.show()

# Para visualizar las dos graficas debe de cerrar
# la primera.