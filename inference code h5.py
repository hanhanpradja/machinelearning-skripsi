import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the trained model
save_model_path = 'D:\Given\ekgta\\file skripsi\modelann_nonorm2.h5'  # Sesuaikan dengan path model yang disimpan
model = tf.keras.models.load_model(save_model_path)

# Load the scaler
scaler_path = 'D:\Given\ekgta/file skripsi\scaler2.pkl'  # Sesuaikan dengan path scaler yang disimpan
scaler = joblib.load(scaler_path)

# Load and preprocess new data for inference
url_new_data = 'D:\Given\ekgta/file skripsi\Subjek2\\aesan_tidur\HasilOlahData1.xlsx'  # Path to the new dataset for inference
data_new = pd.read_excel(url_new_data)

# Split features and target
X_new = data_new.drop(columns=['Classification Result']).values
y_new = data_new['Classification Result'].values  # If you want to compare predictions with actual labels

# Normalize features using the same scaler used during training
X_new_normalized = scaler.transform(X_new)  # Use transform instead of fit_transform

# Make predictions
y_pred_new = model.predict(X_new_normalized)
y_pred_classes_new = np.argmax(y_pred_new, axis=1) + 1  # Add 1 to return to original class labels (1-4)

# Calculate accuracy
accuracy = accuracy_score(y_new, y_pred_classes_new)
print(f"Akurasi model: {accuracy:.2f}")

# Map the predicted classes to the respective labels
class_mapping = {
    1: 'Abnormal',
    2: 'Normal',
    3: 'Berpotensi Aritmia',
    4: 'Sangat Berpotensi Aritmia'
}
y_pred_labels_new = np.vectorize(class_mapping.get)(y_pred_classes_new)

# Print predicted classes
print("Prediksi kelas untuk data baru:")
print(y_pred_labels_new)

# Create a DataFrame with original data and predicted classes
data_new['Offline'] = y_pred_labels_new

# Save the DataFrame to an Excel file
output_file = 'D:\Given\ekgta\\file skripsi\\tidur.xlsx'  # Sesuaikan dengan path yang diinginkan
data_new.to_excel(output_file, index=False)
print(f"Hasil prediksi telah disimpan ke {output_file}")
