import pandas as pd

name = 'ISCAN'
# Memuat kedua file Excel dengan memastikan baris pertama sebagai header
file1 = pd.read_excel(f'hasil pqrst lama/kelas3/{name}.xlsx', header=0)
file2 = pd.read_excel(f'hasil pqrst lama/kelas3/{name}2.xlsx', skiprows=1)

file2.columns = file1.columns
# Menggabungkan kedua file Excel
combined = pd.concat([file1, file2], ignore_index=True)

# Menyimpan hasil gabungan ke file baru
combined.to_csv(f'datasetfix/{name}.csv', index=False)
print("Files successfully merged into 'combined_file.xlsx'")

