import pandas as pd
file_path = './uci_malware_detection.csv'

df = pd.read_csv(file_path)
print(df.info())
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(report)
