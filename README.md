# Coffee Roast Classification

Implementasi **Multimodal Classification of Coffee Roast Level Using Bean Color Images and Roasting Crack Sound**, untuk tugas mata kuliah Rekayasa Fitur dan Pengenalan Pola, Magister Ilmu Komputer, Universitas Gadjah Mada.

Anggota Kelompok:
1. Alif Ilman Nafian (25/572957/PPA/07203)
2. Kevin Pratama (25/574311/PPA/07241)
3. Maulana Yusuf Habibi (25/575185/PPA/07275)
4. Alfi Maulana Akbar (25/574622/PPA/07253)

## Google Colab

Buka [notebooks/Coffee_Roast_Colab.ipynb](notebooks/Coffee_Roast_Colab.ipynb) di Google Colab, lalu **Runtime > Run all**. Notebook menyertakan seluruh kode sehingga tidak memerlukan URL repository, upload source, atau mount Drive. Sel berurutan menulis source, memasang dependensi/Git LFS, mengunduh dan mengekstrak dataset, menjalankan tes, melatih model, menampilkan metrik, dan mengunduh hasil ZIP.

## Metode

| Komponen | Implementasi |
| --- | --- |
| Fitur gambar | RGB + HSV: histogram 16 bin per kanal, mean, dan standar deviasi (108 fitur), dari seluruh frame tanpa segmentasi bean |
| Fitur audio | Durasi, peak, RMS, ZCR, centroid, bandwidth, rolloff, lima band energi, kepadatan dan interval puncak lokal (15 fitur) |
| Klasifikasi | SVM RBF dan Random Forest 200 pohon, bobot kelas seimbang, tanpa pencarian hyperparameter |
| Seleksi | Dilatih hanya pada train, pilih akurasi validasi tertinggi (seri: SVM). `StandardScaler` hanya fit pada train |
| Evaluasi | Held-out test: accuracy, precision/recall/F1 per kelas, macro/weighted F1, confusion matrix |

**Split data** (duplikat dibuang berdasarkan hash SHA-256, seed 42)

| Data | File asli | Unik | Train / Val / Test |
| --- | --- | --- | --- |
| Gambar | 1.600 | 1.194 | 614 / 180 / 400 |
| Audio | 4.304 | 3.557 | 2.334 / 592 / 631 |

- Gambar: prioritas test, validasi stratified 15% dari total unik. Test penerbit dipertahankan, sehingga train dan validasi mengisi sisanya.
- Audio: split dari paper yang ada dipertahankan, prioritas deduplikasi test > val > train. Delapan trial beririsan antara train dan test.
- Audio berupa cuplikan event pendek, sehingga kepadatan dan interval puncak hanya menggambarkan puncak lokal dalam satu cuplikan. Jumlah file audio aktual 4.304, sedangkan dokumentasi upstream menyebut 4.301.

## Fusion dan batasan ilmiah

`coffee.fusion.fuse` menggabungkan probabilitas gambar dan audio dengan bobot proporsional akurasi validasi tiap modalitas. Dataset gambar (4 kelas) dan audio (3 kelas, tanpa Medium) tidak berpasangan, sehingga bobot ini berperan sebagai aturan heuristik.

- First Crack dipetakan ke Light, Second Crack ke Dark.
- Background: jika tahap diketahui, masukkan `Green` (sebelum first crack) atau `Medium` (di antara dua crack). Tanpa konteks, mengikuti distribusi gambar.
- Evaluasi fusion memerlukan gambar, audio, dan label dari batch roasting yang sama.

## Berkas hasil

- `outputs/metrics.json`: metrik dan audit.
- `outputs/*_manifest.json`: split (path dan label per sampel).
- `outputs/image.joblib`, `outputs/audio.joblib`: model, kelas, dan akurasi validasi.
- `outputs/*_test_predictions.npz`: label dan probabilitas test.
- `reports/verification.md`: pengujian yang benar-benar dijalankan.

Urutan kelas confusion matrix: Green, Light, Medium, Dark (gambar); Background, First Crack, Second Crack (audio).

Tes mencakup fitur, audio hening, fusion, validasi input, ZIP traversal, dan integrasi data nyata (dilewati jika folder data tidak ada).

## Sumber dan atribusi

- [Coffee Bean Dataset Resized, gpiosenka](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224), CC BY-SA 4.0, cite Ontoum et al. (2022), [Coffee Roast Intelligence](https://doi.org/10.48550/arXiv.2206.01841).
- [Coffee Roasting Acoustic Dataset, Shi, Li, He](https://github.com/Theseus-427/coffee-roasting-acoustic-dataset), CC BY 4.0. Dataset diunduh tanpa perubahan konten.
- Nair et al. (2025), [CBD: Coffee Beans Dataset](https://doi.org/10.1016/j.dib.2025.111434), dasar pendekatan klasifikasi fitur pada proposal (hanya sebagai referensi).
- Wilson (2014), [Coffee roasting acoustics](https://doi.org/10.1121/1.4874355), landasan karakteristik first/second crack.
