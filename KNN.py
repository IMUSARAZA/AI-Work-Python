from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_file_path = 'trainFinal.csv'
train_df = pd.read_csv(train_file_path)

test_file_path = 'testFinal.csv'  
test_df = pd.read_csv(test_file_path)

features = ['NumofDelayedPayment', 'ChangedCreditLimit', 'NumCreditInquiries', 
            'CreditMix', 'OutstandingDebt', 'CreditUtilizationRatio', 'CreditHistoryAge', 
            'PaymentofMinAmount']
target = 'CreditScore'  


X_train = train_df[features]  
y_train = train_df[target]  

X_test = test_df[features]  

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)  

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

output_df = pd.DataFrame({'PredictedCreditScore': y_pred})
output_df.to_csv('test_predictions_knn.csv', index=False)

print("Predictions saved to test_predictions_knn.csv")

