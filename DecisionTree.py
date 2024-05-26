import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


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


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


X_test = test_df_processed
y_pred = model.predict(X_test)


print(y_pred)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_val)


report = classification_report(y_val, y_pred, target_names=['good', 'standard', 'poor'])

print(report)

y_pred = le.inverse_transform(y_pred)


predictions_df = pd.DataFrame(y_pred, columns=['CreditScore'])


predictions_df.to_csv(r'C:\Users\musar\Desktop\AI\PythonIntro\test_predictions_decisionTree.csv', index=False)

print("Predictions saved to C:\\Users\\musar\\Desktop\\AI\\PythonIntro\\test_predictions_decisionTree.csv")
