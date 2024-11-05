import numpy as np
import pandas as pd
import os, shutil


# Classifier
classification_rules = {
    'NORM': ['NORM'],
    'ABN': [
        'PACE', 'SR', 'RAO', 'LAO',
        'LVH', 'RVH', 'SEHYP'
    ],
    'ARR_POT': [
        'LAFB', '1AVB', '2AVB', '3AVB', 'WPW', 'LPFB',
        'NDT', 'NST_', 'DIG', 'LNGQT', 'EL', 'ABQRS', 
        'IVCD','CLBBB', 'ILBBB', 'IRBBB', 'CRBBB'
    ],
    'ARR_VERY_POT': [
        'IMI', 'ASMI', 'ILMI', 'ALMI', 'LMI', 'IPLMI', 
        'IPMI', 'PMI', 'AFLT', 'AMI', 'PSVT', 'STACH',
        'ISC_', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 
        'ISCIN', 'ISCLA', 'INJAL', 'INJAS', 'INJLA', 
        'INJIL', 'INJIN', 'PVC', 'AFIB'
    ]
}

# Definisikan direktori
source_directory = "all_data"  # Ganti dengan path folder sumber
destination_directory = "dataFix"  # Ganti dengan path folder tujuan

# Buat folder tujuan untuk setiap kategori
os.makedirs(destination_directory, exist_ok=True)
for category in classification_rules:
    os.makedirs(os.path.join(destination_directory, category), exist_ok=True)

# Fungsi untuk menentukan kategori dari folder nama berdasarkan aturan klasifikasi
def classify_folder(folder_name):
    for category, names in classification_rules.items():
        if folder_name in names:
            return category
    return None

# Pindahkan atau salin file berdasarkan klasifikasi
for root, dirs, files in os.walk(source_directory):
    for folder in dirs:
        category = classify_folder(folder)
        if category:
            source_path = os.path.join(root, folder)
            dest_path = os.path.join(destination_directory, category)
            for file_name in os.listdir(source_path):
                file_path = os.path.join(source_path, file_name)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, dest_path)  # Menyalin file ke folder tujuan yang sesuai

print("Pengelompokan file selesai.")