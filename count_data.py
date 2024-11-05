import pandas as pd

# Membaca file CSV
file_path = 'dataset_balance_2.csv'  # Ganti dengan path file CSV Anda
df = pd.read_csv(file_path)

# Menghitung jumlah data untuk setiap nilai di kolom 'output'
output_counts = df['output'].value_counts().sort_index()

# Menampilkan hasil
for value in range(4):  # Untuk nilai 0, 1, 2, dan 3
    count = output_counts.get(value, 0)  # Mengambil jumlah, default 0 jika tidak ada
    print(f"Jumlah data dengan output {value}: {count}")
