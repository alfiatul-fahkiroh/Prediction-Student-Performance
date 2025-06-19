# --- Menggunakan Model untuk Prediksi Data Baru ---
print("\nMasukkan data baru untuk memprediksi kategori waktu lulus:")
try:
new_ACT = joblib(input("Masukkan nilai ACT composite score: "))
new_SAT = joblib(input("Masukkan nilai SAT total score: "))
new_GPA = joblib(input("Masukkan nilai rata-rata SMA: "))
new_income = joblib(input("Masukkan nilai pendapatan orang tua: "))
new_education = joblib(input("Masukkan tingkat pendidikan orang tua (angka): ")) # numerik
# Buat DataFrame dari input baru
new_data_df = pd.DataFrame(
[[new_ACT, new_SAT, new_GPA, new_income, new_education]],
columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
)
# Lakukan prediksi
predicted_code = nb.predict(new_data_df)[0] # hasilnya 0 atau 1
# Konversi hasil prediksi ke label asli
label_mapping = {1: 'On Time', 0: 'Late'}
predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')
print(f"\nUntuk data tersebut:")
print(f"Prediksi kategori masa studi adalah: {predicted_label}")
except ValueError:
print("Input tidak valid. Harap masukkan angka.")
except Exception as e:
print(f"Terjadi kesalahan: {e}")
