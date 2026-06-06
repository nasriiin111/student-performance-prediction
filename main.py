import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv('Student_Performance.csv')
print(df.head())
print(df.info())

print(df.isnull().sum())

plt.scatter(df['study_hours'], df['overall_score'])
plt.xlabel('study_hours')
plt.ylabel('overall_score')
plt.show()

plt.scatter(df['attendance_percentage'], df['overall_score'])
plt.xlabel('attendance_percentage')
plt.ylabel('overall_score')
plt.show()

plt.scatter(df['age'], df['overall_score'])
plt.xlabel('age')
plt.ylabel('overall_score')
plt.show()

X=df[['study_hours', 'attendance_percentage', 'age']]
y=df['overall_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae=mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

