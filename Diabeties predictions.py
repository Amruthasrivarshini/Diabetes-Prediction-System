#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy matplotlib seaborn scikit-learn


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)
print(data.head())


# In[3]:


print(data.info())
print(data.describe())


# In[4]:


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[5]:


sns.countplot(x='Outcome', data=data, palette='Set2')
plt.title("Distribution of Outcome")
plt.show()


# In[6]:


for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data[column] = data[column].replace(0, np.nan)
    data[column].fillna(data[column].median(), inplace=True)


# In[7]:


X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[9]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[10]:


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[11]:


print(classification_report(y_test, y_pred))


# In[12]:


new_patient = np.array([[2, 120, 70, 30, 80, 25.6, 0.6, 32]])  # Example input
prediction = model.predict(new_patient)
if prediction[0] == 1:
    print("The patient is likely to have diabetes.")
else:
    print("The patient is unlikely to have diabetes.")


# In[13]:


import joblib
joblib.dump(model, 'diabetes_model.pkl')


# In[14]:


feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title("Feature Importance")
plt.show()


# In[ ]:




