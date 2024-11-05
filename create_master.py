import os
import pandas as pd

# Folder yang berisi file CSV
folder_path = 'datasetfix'

# Aturan klasifikasi
classification_rules = {
    'NORM': 0,
    'ABN': 1,
    'ARR_POT': 2,
    'ARR_VERY_POT': 3
}

classification_mapping = {
    'NORM': ['NORM'],
    'ABN': [
        'PACE', 'SR', 'RAO', 'LAO', 'LAE',
        'LVH', 'RVH', 'SEHYP'
    ],
    'ARR_POT': [
        'LAFB', '1AVB', '2AVB', '3AVB', 'WPW', 'LPFB',
        'NDT', 'NST_', 'DIG', 'LNGQT', 'EL', 'ABQRS', 
        'IVCD', 'CLBBB', 'ILBBB', 'IRBBB', 'CRBBB'
    ],
    'ARR_VERY_POT': [
        'IMI', 'ASMI', 'ILMI', 'ALMI', 'LMI', 'IPLMI', 
        'IPMI', 'PMI', 'AFLT', 'AMI', 'PSVT', 'STACH',
        'ISC_', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 
        'ISCIN', 'ISCLA', 'INJAL', 'INJAS', 'INJLA', 
        'INJIL', 'INJIN', 'PVC', 'AFIB'
    ]
}

# Fungsi untuk menentukan klasifikasi berdasarkan nama file
def get_classification(filename):
    for key, labels in classification_mapping.items():
        if any(label in filename for label in labels):
            return classification_rules[key]
    return None

# List untuk menyimpan data frame
data_frames = []

# Loop untuk membaca dan menambahkan kolom pada setiap file CSV
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # Membaca file CSV dengan header diabaikan
        df = pd.read_csv(file_path, header=None, skiprows=2)

        # Tentukan klasifikasi berdasarkan nama file
        classification = get_classification(filename)
        
        # Tambahkan kolom klasifikasi ke dataframe
        if classification is not None:
            df['output'] = classification  # Ganti nama kolom klasifikasi menjadi 'output'
            data_frames.append(df)
        else:
            print(f"File '{filename}' tidak sesuai klasifikasi.")

# Gabungkan semua data frame dan ubah nama kolom
if data_frames:
    master_df = pd.concat(data_frames, ignore_index=True)
    
    # Ubah nama kolom sesuai dengan yang diinginkan
    master_df.columns = ['rr', 'rr_std', 'pr', 'pr_std', 
                         'qs', 'qs_std', 'qt', 'qt_std', 
                         'st', 'st_std', 'r/s', 'heartrate', 'output']
    
    # Simpan sebagai file CSV
    master_df.to_csv('master_file.csv', index=False)
    print("File master berhasil dibuat sebagai 'master_file.csv'")
else:
    print("Tidak ada file yang cocok untuk klasifikasi.")
