import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the training data
train_df = pd.read_csv(r'C:\Users\musar\Desktop\trainFinal.csv')

# Load the testing data
test_df = pd.read_csv(r'C:\Users\musar\Desktop\testFinal.csv')

le = LabelEncoder()

def preprocess_data(df):
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    return df

train_df_processed = preprocess_data(train_df)
test_df_processed = preprocess_data(test_df)

X_train = train_df_processed.drop(columns=['CreditScore'])
y_train = train_df_processed['CreditScore']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(test_df_processed)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred = le.inverse_transform(y_pred)
predictions_df = pd.DataFrame(y_pred, columns=['CreditScore'])

predictions_df.to_csv(r'C:\Users\musar\Desktop\AI\PythonIntro\test_predictions_logisticRegression.csv', index=False)

print("Predictions saved to C:\\Users\\musar\\Desktop\\AI\\PythonIntro\\test_predictions_logisticRegression.csv")