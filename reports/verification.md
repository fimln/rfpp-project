# Verifikasi lokal, 10 November 2026

Environment: Windows, Python 3.12.13, dependensi baseline pada requirements.txt. Baseline scikit-learn berjalan di CPU karena tidak memakai backend CUDA. RTX 5060 tersedia tetapi PyTorch tidak terpasang; jalur opsional MobileNet belum dieksekusi lokal. Jangan menyebut jalur CNN sudah teruji.

## Hasil yang dijalankan

- `python -m pytest -q`: **7 passed**, termasuk integrasi dataset nyata.
- `python -m coffee.train`: selesai pada seluruh konten unik gambar dan audio; model serta prediksi tersimpan di outputs.
- Gambar: SVM dipilih berdasarkan validasi; test accuracy **0.9825** pada 400 file.
- Audio: hasil test **0.9096671949286846** pada 631 file unik. Detail model terpilih, validasi, F1, dan confusion matrix ada pada [metrics.json](metrics.json).
- `python -m coffee.predict` berhasil menghasilkan distribusi 4 kelas dengan input gambar Medium dan audio Background lokal. Pasangan ini hanya smoke test fungsi, bukan pasangan observasi nyata.
- `git lfs fsck` pada data/audio-lfs: **OK**. Commit sumber audio: `ae963ef3a2e18c93a36b6b2d088c959f82efd765`.
- `uv pip check`: semua paket kompatibel.
- Compile source, validasi nbformat, parsing Python sel biasa, dan pencocokan source tertanam notebook dengan source proyek berhasil.
- `git check-ignore` memastikan paper, konversi Markdown, dua folder dataset asli, data unduhan, outputs dan venv diabaikan.

## Koreksi dataset dan batas verifikasi

Gambar awal 1.600 file memiliki 406 salinan duplikat, termasuk 400 salinan test pada train. Angka 98,5% sebelum deduplikasi dibatalkan. Hasil final memakai 1.194 konten unik, tanpa duplikat byte lintas split.

Audio awal lokal seluruhnya pointer Git LFS. Objek WAV asli berhasil diambil melalui clone dengan Git LFS. Sebanyak 4.304 file menjadi 3.557 konten unik. Hasil 93,98% sebelum deduplikasi dibatalkan. Hasil final setelah deduplikasi masih memiliki irisan delapan trial train/test sesuai split penerbit, sehingga generalisasi trial baru belum diukur.

Unduhan ZIP Kaggle berhasil mengalirkan data tetapi lambat dan dihentikan sebelum lengkap untuk menghindari mengunduh ulang data yang sudah tersedia. Jadi unduhan dan ekstraksi **penuh ZIP Kaggle belum diverifikasi**. Ekstraksi ZIP valid dan penolakan path traversal diuji dengan fixture. Tidak ada klaim bahwa notebook sudah dijalankan di layanan Google Colab; yang diverifikasi adalah struktur, sintaks, kesamaan source, serta pipeline baseline lokal. Jalur CNN belum diuji. Tidak ada akurasi fusion yang diukur karena dataset tidak berpasangan.

Sklearn 1.9 memberi FutureWarning untuk SVC(probability=True); masih berfungsi pada versi yang dipin, akan memerlukan migrasi kalibrasi bila menaikkan sklearn ke 1.11.
