#simple linear regression

#import packages required for dataset
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data from csv
data = pd.read_csv('sampleData.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


#prepare the data into training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=0)


#prepare the model
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train)

y_pred = linearRegressor.predict(x_test)

#plotting for training set
# plt.scatter(x_train, y_train, color = 'red')
# plt.plot(x_train, linearRegressor.predict(x_train), color = 'blue')
# plt.title("SAT vs GPA")
# plt.xlabel("SAT")
# plt.ylabel("GPA")
# plt.show()

#plotting for test data
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
plt.title("SAT vs GPA")
plt.xlabel("SAT")
plt.ylabel("GPA")
plt.show()
















