import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from scikeras.wrappers import KerasClassifier
import joblib
from keras.regularizers import l2, L1L2,l1

# Set environment variable to disable OneDNN warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the callback class
class AkurasiCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.97:
            print("\nAkurasi telah mencapai lebih dari 95%!")
            self.model.stop_training = True

# Load the data
url = 'D:\\download\\datacobacoba1.xlsx'  # Sesuaikan dengan path file Anda
data = pd.read_excel(url)

# Split features and target
X = data.drop(columns=['output']).values
y = data['output'].values

# Normalize features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Visualize data before and after normalization
# plt.figure(figsize=(16, 6))

# # Plot histograms for each feature before normalization
# plt.subplot(1, 2, 1)
# plt.title('Sebelum Normalisasi')
# for i in range(X.shape[1]):
#     sns.histplot(X[:, i], bins=20, kde=True, label=f'Fitur {i+1}', alpha=0.5)
# plt.xlabel('Nilai Fitur')
# plt.ylabel('Frekuensi')
# plt.legend()

# # Plot histograms for each feature after normalization
# plt.subplot(1, 2, 2)
# plt.title('Setelah Normalisasi')
# for i in range(X_normalized.shape[1]):
#     sns.histplot(X_normalized[:, i], bins=20, kde=True, label=f'Fitur {i+1}', alpha=0.5)
# plt.xlabel('Nilai Fitur (Dinormalisasi)')
# plt.ylabel('Frekuensi')
# plt.legend()

# plt.tight_layout()
# plt.show()

# Determine the number of classes
num_classes = len(np.unique(y))

# One-hot encode labels
y_train_encoded = tf.keras.utils.to_categorical(y - 1, num_classes)  # Adjust labels to start from 0 for one-hot encoding

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_train_encoded, test_size=0.2, random_state=123)

# Define a function to create the Keras model
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Add Dropout
        Dense(64, activation='relu'),
        Dropout(0.1),
        # Dense(50, activation='relu'),  # Additional hidden layer
        # Dropout(0.1),
        # Dense(25, activation='relu'),
        # Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create a KerasClassifier
model_s = KerasClassifier(model=create_model, epochs=300, batch_size=50, verbose=1, callbacks=[AkurasiCallback()])

# Define K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=123)

# Train and evaluate the model with cross-validation
cv_scores = cross_val_score(model_s, X_train, y_train, cv=kfold)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Accuracy: %.2f%%" % (cv_scores.mean() * 100))
print("Standard Deviation of Cross-Validation Accuracy: %.2f%%" % (cv_scores.std() * 100))

# Train the model
model_s.fit(X_train, y_train, epochs=300, batch_size=50, validation_split=0.2, verbose=1)

# Access training history
history = model_s.history_

plt.figure(figsize=(10, 4))
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Akurasi Model')
plt.ylabel('Akurasi (%)')
plt.xlabel('Epoch')
plt.legend(['Latih', 'Validasi'], loc='upper left')
plt.grid(True)
plt.show()

# Plot model loss
plt.figure(figsize=(10, 4))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Kehilangan Model')
plt.ylabel('Kehilangan')
plt.xlabel('Epoch')
plt.legend(['Latih', 'Validasi'], loc='upper left')
plt.grid(True)
plt.show()

# Evaluate the model
print('Evaluasi model pada data pengujian:')
print(model_s.score(X_test, y_test))

# Predict classes for test set
y_pred = model_s.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print("Laporan Klasifikasi:")
print(classification_report(np.argmax(y_test, axis=1), y_pred_classes))

# Print confusion matrix
print("Matriks Kebingungan:")
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes))

# Calculate precision, recall, and F1-score for ANN
presisi_ann = precision_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted', zero_division=1)
recall_ann = recall_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted', zero_division=1)
f1_ann = f1_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')

# Convert precision, recall, and F1-score to percentage
persen_presisi_ann = presisi_ann * 100
persen_recall_ann = recall_ann * 100
persen_f1_ann = f1_ann * 100

# Display precision, recall, and F1-score for ANN
print("Presisi ANN: {:.2f}%".format(persen_presisi_ann))
print("Recall ANN: {:.2f}%".format(persen_recall_ann))
print("F1-score ANN: {:.2f}%".format(persen_f1_ann))

# Compute confusion matrix for ANN
matriks_kebingungan_ann = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

# Plot confusion matrix for ANN
kelas_nama = np.unique(np.argmax(y_test, axis=1))
fig, ax = plt.subplots()
sns.heatmap(matriks_kebingungan_ann, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, xticklabels=kelas_nama, yticklabels=kelas_nama)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Matriks Kebingungan ANN')
plt.show()

# Simpan model setelah pelatihan
model = model_s.model_  # Akses model yang sudah dilatih dari KerasClassifier
save_model_path = 'D:/Given/ekgta/modelann_nonorm2.h5'  # Sesuaikan dengan path yang diinginkan
model.save(save_model_path)

# Save the scaler
scaler_path = 'D:/Given/ekgta/scaler2.pkl'  # Sesuaikan dengan path yang diinginkan
joblib.dump(scaler, scaler_path)
