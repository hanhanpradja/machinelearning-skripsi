import os
import shutil

def merge_folders(src_folder, dest_folder):
    """Gabungkan isi folder src ke folder dest tanpa overwrite."""
    for root, dirs, files in os.walk(src_folder):
        relative_path = os.path.relpath(root, src_folder)
        target_path = os.path.join(dest_folder, relative_path)
        
        # Buat folder target jika tidak ada
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        # Pindahkan setiap file
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(target_path, file)
            
            # Jika file sudah ada, rename atau tambahkan penanda agar tidak overwrite
            if os.path.exists(dest_file):
                # Tambahkan _copy atau nama unik untuk file yang sama agar tidak replace
                dest_file = os.path.join(target_path, f"{os.path.splitext(file)[0]}_copy{os.path.splitext(file)[1]}")
            
            shutil.copy2(src_file, dest_file)  # Copy file ke lokasi tujuan

# Paths ke dua folder yang ingin digabungkan
folder1 = '1-5492'  # Ganti dengan path ke folder pertama
folder2 = '5493-21000'  # Ganti dengan path ke folder kedua
merged_folder = 'all_data'  # Lokasi untuk hasil penggabungan

# Buat folder gabungan jika belum ada
if not os.path.exists(merged_folder):
    os.makedirs(merged_folder)

# Gabungkan isi dari kedua folder
merge_folders(folder1, merged_folder)  # Gabungkan isi folder 1 ke folder tujuan
merge_folders(folder2, merged_folder)  # Gabungkan isi folder 2 ke folder yang sama
