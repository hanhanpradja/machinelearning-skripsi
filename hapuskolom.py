import pandas as pd

# Membaca file CSV
df = pd.read_csv('dataset_balance_1_bfr.csv', header=0)

# Menghapus kolom yang diinginkan
df.drop(columns=['rr_std', 'pr_std', 'qt_std', 'qs_std', 'st_std'], inplace=True)

# Menyimpan hasil ke file baru tanpa menyertakan kolom indeks
df.to_csv('dataset_balance_1.csv', index=False)

