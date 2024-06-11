import pandas as pd

data = pd.read_csv("CreditPrediction.csv")

data["Gender"] = data["Gender"].map({"M": 1, "F": 0})

data["Education_Level"] = data["Education_Level"].map(
    {"Unknown": 0, "Uneducated": 1, "High School": 2, "College": 3, "Graduate": 4, "Post-Graduate": 5, "Doctorate": 6})

data["Marital_Status"] = data["Marital_Status"].map({"Married": 1, "Single": 0, "Divorced": 2, "Unknown": 3})

data["Income_Category"] = data["Income_Category"].map(
    {"Less than $40K": 1, "$40K - $60K": 2, "$60K - $80K": 3, "$80K - $120K": 4, "$120K +": 5, "Unknown": 0})

data["Card_Category"] = data["Card_Category"].map({"Blue": 1, "Silver": 2, "Gold": 3, "Platinum": 4})

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[["Customer_Age",
                                         "Gender",
                                         "Dependent_count",
                                         "Education_Level",
                                         "Marital_Status",
                                         "Income_Category",
                                         "Card_Category",
                                         "Months_on_book",
                                         "Total_Relationship_Count",
                                         "Months_Inactive_12_mon",
                                         "Contacts_Count_12_mon",
                                         "Credit_Limit",
                                         "Total_Revolving_Bal",
                                         "Total_Amt_Chng_Q4_Q1",
                                         "Total_Trans_Amt",
                                         "Total_Trans_Ct",
                                         "Total_Ct_Chng_Q4_Q1",
                                         "Avg_Utilization_Ratio"]])

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data_scaled)

from sklearn.ensemble import IsolationForest

clf = IsolationForest()

clf.fit(data_imputed)

predictions = clf.predict(data_imputed)

cleaned_data = data_imputed[predictions == 1]

from sklearn.model_selection import train_test_split

X = data_imputed[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]]
Y = data_imputed[:, 11]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, Y_train)

from sklearn.metrics import mean_squared_error

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)

print("MSE : ", mse)

print("Y_test       Y_pred      |Y_test - Y_train|")
for i in range(len(Y_test)):
    print(f"{Y_test[i]}        {Y_pred[i]}      {abs(Y_test[i] - Y_pred[i])}")
