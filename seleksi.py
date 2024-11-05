import os
import pandas as pd

# Tentukan folder yang berisi file Excel
folder_path = 'datasetfix'

# Loop untuk membaca setiap file di dalam folder
for filename in os.listdir(folder_path):
    # Cek apakah file berformat .xlsx
    if filename.endswith('.xlsx'):
        excel_path = os.path.join(folder_path, filename)
        
        # Baca file Excel
        df = pd.read_excel(excel_path)
        
        # Tentukan nama file output CSV
        csv_filename = filename.replace('.xlsx', '.csv')
        csv_path = os.path.join(folder_path, csv_filename)
        
        # Simpan data sebagai CSV
        df.to_csv(csv_path, index=False)
        
        # Hapus file Excel setelah konversi
        os.remove(excel_path)
        print(f"Converted and deleted: {filename}")

print("All Excel files have been converted to CSV and deleted.")
