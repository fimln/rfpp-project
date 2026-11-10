# Coffee Roast Classification

Implementasi tugas project **Multimodal Classification of Coffee Roast Level Using Bean Color Images and Roasting Crack Sound**, tugas Rekayasa Fitur dan Pengenalan Pola, Magister Ilmu Komputer, Universitas Gadjah Mada.

Anggota kelompok:

1. Alif Ilman Nafian (25/572957/PPA/07203)
2. Kevin Pratama (25/574311/PPA/07241)
3. Maulana Yusuf Habibi (25/575185/PPA/07275)
4. Alfi Maulana Akbar (25/574622/PPA/07253)

## Menjalankan

Buka [notebook di Google Colab](https://colab.research.google.com/github/fimln/rfpp-project/blob/main/notebooks/Coffee_Roast_Colab.ipynb), lalu pilih **Runtime > Run all**. Notebook mengambil kode dan dependensi dari repository.

## Metode dan data

![Rancangan metode dengan gambar dan audio dari batch roasting yang sama](assets/method-diagram.png)

| Komponen | Implementasi |
| --- | --- |
| Gambar | 108 fitur RGB/HSV dari seluruh frame: histogram, mean, dan standar deviasi |
| Audio | 15 fitur: durasi, amplitudo, energi, spektrum, serta kepadatan dan interval puncak lokal |
| Model | SVM RBF dan Random Forest 200 pohon dengan bobot kelas seimbang |
| Seleksi | Akurasi validasi tertinggi, scaler dilatih pada train |
| Evaluasi | Accuracy, precision, recall, F1, dan confusion matrix pada test |

Duplikat dihapus berdasarkan SHA-256, pembagian data memakai seed 42.

| Dataset | File asli | Unik | Train / Val / Test |
| --- | --- | --- | --- |
| Gambar | 1.600 | 1.194 | 614 / 180 / 400 |
| Audio | 4.304 | 3.557 | 2.334 / 592 / 631 |

Kelas gambar: Green, Light, Medium, Dark. Kelas audio: Background, First Crack, Second Crack. Split test gambar dan split audio mengikuti penerbit, dengan prioritas deduplikasi test > val > train. Delapan trial audio beririsan antara train dan test, sehingga evaluasi masih mencakup trial yang dikenal model. Fitur puncak audio dihitung dalam setiap cuplikan pendek.

## Penggabungan gambar dan audio

Kedua dataset berasal dari observasi terpisah. Proyek menyediakan dua cara penggabungan:

- **Fungsi `coffee.fusion.fuse`:** menggabungkan probabilitas berdasarkan akurasi validasi. First Crack dipetakan ke Light dan Second Crack ke Dark. Background mengikuti distribusi gambar, atau kelas Green/Medium yang diberikan melalui pengamatan eksternal.
- **Simulasi Light–Dark di notebook:** memasangkan gambar Light dengan First Crack dan Dark dengan Second Crack berdasarkan label, menggunakan seed 42–46. Probabilitas dinormalisasi ke dua kelas. Pasangan seimbang dibuat di dalam split masing-masing, dengan setiap file dipakai sekali per split/seed. File dapat dipakai kembali pada seed lain.

Bobot simulasi dipilih berdasarkan macro F1 validasi pada grid 0–1 dengan langkah 0,05, hasil seri memakai bobot terdekat ke 0,5. Gambar, audio, dan fusion dibandingkan pada pasangan test identik. Semua seed dilaporkan melalui rata-rata, standar deviasi, dan confusion matrix seed pertama.

Skor simulasi menggambarkan pasangan yang kesesuaian kelasnya ditentukan sejak awal. Standar deviasi menunjukkan variasi susunan pasangan pada file test yang sama. Evaluasi pada roasting nyata memerlukan gambar, audio, dan label dari batch yang sama, sesuai rancangan pada diagram.

Untuk menambahkan Green dan Medium, diperlukan rekaman lengkap atau metadata waktu per batch serta acuan label roast. Background dapat muncul pada berbagai tahap, penentuan kelasnya membutuhkan konteks pengamatan.

## Dasar literatur

- **Anokye-Bempah dkk. (2025):** tujuh profil roasting dan tiga asal Arabika menunjukkan hubungan konsisten antara warna dan tahap crack. Rata-rata CIELAB L* adalah 29,74 ± 1,9 saat first crack dan 19,96 ± 1,0 saat second crack. Pengukuran menggunakan kopi giling dan spektrofotometer, penerapan pada foto biji utuh memerlukan kalibrasi warna dan pencahayaan.
- **Wilson (2014):** first crack berlangsung sekitar detik 400–600 dan second crack sekitar 620–730 pada biji dan mesin percobaannya. Waktu tersebut menjadi contoh urutan proses, pemetaan ke dataset lain memerlukan waktu roasting masing-masing.

Kedua studi mendukung pemakaian tahap crack sebagai pendekatan untuk pairing Light–Dark. Kesesuaiannya dengan label dataset tetap menjadi asumsi simulasi.

## Berkas hasil

| Berkas | Isi |
| --- | --- |
| `outputs/metrics.json`, `outputs/*_manifest.json` | Metrik, audit, dan pembagian data |
| `outputs/image.joblib`, `outputs/audio.joblib` | Model, kelas, dan akurasi validasi |
| `outputs/*_test_predictions.npz` | Label dan probabilitas test |
| `outputs/synthetic_*` | Pasangan, metrik, grafik, dan kesimpulan simulasi |
| [reports/verification.md](reports/verification.md) | Hasil verifikasi lokal dan cakupan pengujian |

## Sumber dan atribusi

- [Coffee Bean Dataset Resized, gpiosenka](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224), CC BY-SA 4.0, atribusi Ontoum dkk. (2022), [Coffee Roast Intelligence](https://doi.org/10.48550/arXiv.2206.01841).
- [Coffee Roasting Acoustic Dataset, Shi, Li, He](https://github.com/Theseus-427/coffee-roasting-acoustic-dataset), CC BY 4.0.
- Nair dkk. (2025), [CBD: Coffee Beans Dataset](https://doi.org/10.1016/j.dib.2025.111434), referensi pendekatan fitur pada proposal.
- Wilson (2014), [Coffee roasting acoustics](https://doi.org/10.1121/1.4874355).
- Anokye-Bempah dkk. (2025), [A universal color curve for roasted arabica coffee](https://doi.org/10.1038/s41598-025-06601-w), [koreksi label sumbu Gambar 4](https://doi.org/10.1038/s41598-025-23127-3).
