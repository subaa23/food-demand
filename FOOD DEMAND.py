import pandas as pd
import numpy as np
df=pd.read_csv("Food demand.csv")
print(df)
x=df[['id','week','center_id','meal_id','checkout_price','base_price','emailer_for_promotion','homepage_featured']]
y=df['num_orders']
print(x)
print(y)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("x_test",x_test.shape)
print("x_train",x_train.shape)
print("y_test",x_test.shape)
print("y_train",y_train.shape)
print(df.shape)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("y_pred",y_pred)
print("y_test",y_test)
print("MEAN SQUARED ERROR ",mean_squared_error(y_test,y_pred))
result=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(result)
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(range(len(y_test)),result['Actual'],label='Actual')
plt.scatter(range(len(y_test)),result['Predicted'],label='Predicted')
plt.legend(loc='best')
plt.show()
