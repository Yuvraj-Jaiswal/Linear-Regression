import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv("student-mat.csv",sep=";")
data =  data[['G1', 'G2', 'G3','failures','studytime','absences']]
x = data.drop("G3",axis=1)
y = data["G3"]

train_x , test_x , train_y , test_y = train_test_split(x,y,test_size=0.2)

model = LinearRegression()
model.fit(train_x,train_y)

accuracy = model.score(test_x,test_y)
predicted = model.predict(test_x)
print(accuracy)

##
