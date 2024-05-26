from sklearn.naive_bayes import GaussianNB
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

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_test_pred = gnb.predict(X_test)

output_df = pd.DataFrame({'PredictedCreditScore': y_test_pred})
output_df.to_csv('test_predictions_#NaiveB.csv', index=False)

print("Predictions saved to test_predictions_NaiveB.csv")

