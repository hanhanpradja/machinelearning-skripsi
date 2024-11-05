import pandas as pd

# Muat data dari file CSV dan lewati baris pertama (header)
data = pd.read_csv('master_data.csv', skiprows=1, header=None)

# Tetapkan nama kolom, sesuaikan jika kolom Anda memiliki nama tertentu
data.columns = ['rr', 'rr_std', 'pr', 'pr_std', 'qs', 'qs_std', 'qt', 'qt_std', 'st', 'st_std', 'r/s', 'heartrate', 'output']

# Tentukan jumlah data yang diinginkan untuk setiap label
target_count = 241

# Buat list kosong untuk menyimpan data setelah filtering
balanced_data = []

# Loop untuk setiap label dari 0 hingga 3
for label in range(4):
    # Filter data sesuai dengan label
    label_data = data[data['output'] == label]
    
    # Jika jumlah data untuk label lebih besar dari target, lakukan undersampling
    if len(label_data) > target_count:
        label_data = label_data.sample(n=target_count, random_state=42)
    # Jika jumlah data untuk label lebih kecil atau sama dengan target, gunakan semua data yang tersedia
    
    # Tambahkan hasil filtering ke dalam list
    balanced_data.append(label_data)

# Gabungkan semua data yang sudah seimbang
final_data = pd.concat(balanced_data, ignore_index=True)

# Simpan hasil ke file CSV baru, tambahkan header
final_data.to_csv('balanced_data.csv', index=False, header=True)

print("Data berhasil disaring dan disimpan dalam 'balanced_data.csv'")
