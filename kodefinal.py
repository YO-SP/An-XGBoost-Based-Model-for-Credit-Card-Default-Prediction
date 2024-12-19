import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Datasets
all_data_file = "./dataset.csv"  # File pertama yang menyimpan semua dataset
train_test_file = "./data_training.csv"  # File kedua untuk training dan testing (29,000 dataset)
final_test_file = "./data_testing.csv"  # File ketiga untuk menguji model dengan 1,000 dataset

# Load datasets
all_data = pd.read_csv(all_data_file)
train_test_data = pd.read_csv(train_test_file)
final_test_data = pd.read_csv(final_test_file)

# 2. Data Cleaning: Drop ID column if present
if 'ID' in all_data.columns:
    all_data = all_data.drop(columns=['ID'])
if 'ID' in train_test_data.columns:
    train_test_data = train_test_data.drop(columns=['ID'])
if 'ID' in final_test_data.columns:
    final_test_data = final_test_data.drop(columns=['ID'])

# 3. Split features (X) and target (y) for training and testing
X_train_test = train_test_data.drop(columns=['default.payment.next.month'])
y_train_test = train_test_data['default.payment.next.month']

X_final_test = final_test_data.drop(columns=['default.payment.next.month'])
y_final_test = final_test_data['default.payment.next.month']

# Split 80% for training and 20% for initial testing
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=21)

# 4. Train XGBoost model
model_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=21)
model_xgb.fit(X_train, y_train)

# 5. Evaluate model on initial test set
y_pred_test = model_xgb.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

# 6. Evaluate model on final test set
y_pred_final = model_xgb.predict(X_final_test)
accuracy_final = accuracy_score(y_final_test, y_pred_final)
conf_matrix_final = confusion_matrix(y_final_test, y_pred_final)
class_report_final = classification_report(y_final_test, y_pred_final)

# 7. Save Predictions to Excel
# Initial Test Set
results_test = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test
})
results_test.to_excel("initial_test_predictions.xlsx", index=False)

# Final Test Set
results_final = pd.DataFrame({
    'Actual': y_final_test,
    'Predicted': y_pred_final
})
results_final.to_excel("final_test_predictions.xlsx", index=False)

# 8. Print Results
print("Initial Test Set Results:")
print(f"Accuracy: {accuracy_test * 100:.2f}%")
print("\nConfusion Matrix (Initial Test Set):")
print(conf_matrix_test)
print("\nClassification Report (Initial Test Set):")
print(class_report_test)

print("\nFinal Test Set Results:")
print(f"Accuracy: {accuracy_final * 100:.2f}%")
print("\nConfusion Matrix (Final Test Set):")
print(conf_matrix_final)
print("\nClassification Report (Final Test Set):")
print(class_report_final)

# 9. Visualizations
# 9.1 Confusion Matrix Heatmap for Final Test Set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_final, annot=True, fmt='d', cmap='Blues', xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'])
plt.title('Confusion Matrix Heatmap (Final Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9.2 Feature Importance Plot
xgb.plot_importance(model_xgb, max_num_features=10, importance_type='weight', title="Feature Importance (Weight)")
plt.show()
