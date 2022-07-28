#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

#Import data
url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
df_raw = pd.read_csv(url, sep = ";")

#Data cleaning | transforming
df_raw = df_raw.drop_duplicates()
df_interim = df_raw.copy()
df_interim.loc[df_interim["marital"] == "unknown", "marital"] = "married"
df_interim.loc[df_interim["job"] == "unknown", "job"] = "admin."
df_interim.loc[df_interim["education"] == "unknown", "education"] = "university.degree"
df_interim.loc[df_interim["default"] == "unknown", "default"] = "no"
df_interim.loc[df_interim["housing"] == "unknown", "housing"] = "yes"
df_interim.loc[df_interim["loan"] == "unknown", "loan"] = "no"

df_interim = df_interim.replace('unknown', np.nan) 
for var in df_interim.columns[df_interim.dtypes == 'int64']:
    df_interim[var] = df_interim[var].fillna(df_raw[var].mean())

df_interim['age_bins'] = pd.cut(x=df_interim['age'], bins=[10,20,30,40,50,60,70,80,90,100])
df_interim[['age_bins','age']].head()

df_interim['education'] = df_interim['education'].replace({'basic.9y': 'middle_school', 'basic.6y': 'middle_school', 'basic.4y': 'middle_school'})

df_interim = pd.get_dummies(df_interim, columns=['y','age_bins','job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'], drop_first=True)

df_interim = df_interim.drop(['duration','pdays'], axis=1)

#Scale data
scaler = MinMaxScaler()
df_scaler = scaler.fit(df_interim[['age','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
df_interim[['age','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = df_scaler.transform(df_interim[['age','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])

df = df_interim.copy()

#Split data
X = df[['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'age_bins_(20, 30]', 'age_bins_(30, 40]', 'age_bins_(40, 50]', 'age_bins_(50, 60]', 'age_bins_(60, 70]', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'marital_married', 'marital_single', 'education_middle_school', 'education_professional.course', 'education_university.degree', 'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone', 'month_aug','poutcome_nonexistent', 'poutcome_success']]
y = df['y_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=25)

#Model
model = LogisticRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)