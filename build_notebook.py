import nbformat as n
nb=n.v4.new_notebook()
cells=[n.v4.new_markdown_cell('# Coffee Roast Classification\nImplementasi **Multimodal Classification of Coffee Roast Level Using Bean Color Images and Roasting Crack Sound**, untuk tugas mata kuliah Rekayasa Fitur dan Pengenalan Pola, Magister Ilmu Komputer, Universitas Gadjah Mada.\n\nAnggota Kelompok:\n1. Alif Ilman Nafian (25/572957/PPA/07203)\n2. Kevin Pratama (25/574311/PPA/07241)\n3. Maulana Yusuf Habibi (25/575185/PPA/07275)\n4. Alfi Maulana Akbar (25/574622/PPA/07253)\n\nJalankan berurutan dengan **Run all**. Baseline CPU; opsional MobileNet memakai GPU jika tersedia. Dataset gambar dan audio tidak berpasangan, sehingga tidak ada klaim akurasi fusion. Background bukan bukti Green; Medium memerlukan konteks waktu eksternal.\n\nSumber: [Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224), [audio](https://github.com/Theseus-427/coffee-roasting-acoustic-dataset). Internet diperlukan. Unduhan audio memakai Git LFS agar bukan pointer WAV. Jika Kaggle membatasi akses, sel download gagal dengan pesan, bukan melanjutkan dengan data kosong.'),n.v4.new_code_cell('''import sys
from pathlib import Path
import os
import subprocess
assert sys.version_info >= (3,12), 'Memerlukan Python 3.12 atau lebih baru'
REPOSITORY = 'https://github.com/fimln/rfpp-project.git'
REVISION = 'main'  # Ganti dengan commit SHA untuk mengulang versi yang sama.
ROOT = Path('/content/rfpp-project')
RUN_COMPLETED = False
'''),n.v4.new_code_cell('''if not (ROOT / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', REPOSITORY, str(ROOT)], check=True)
subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth', '1', 'origin', REVISION], check=True)
subprocess.run(['git', '-C', str(ROOT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
os.chdir(ROOT)
SOURCE_COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Source commit:', SOURCE_COMMIT)
print('Requirements:', ROOT / 'requirements.txt')
'''),n.v4.new_code_cell('''%pip install -q -r requirements.txt
!apt-get -qq update
!apt-get -qq install -y git-lfs
import subprocess
subprocess.run(['git','lfs','install'],check=True)
'''),n.v4.new_code_cell('''import subprocess, sys
subprocess.run([sys.executable, '-m', 'coffee.download', '--destination', 'data'], check=True)
'''),n.v4.new_code_cell('''# Invalidate previous results before starting this run.
RUN_COMPLETED = False
metrics_path = Path('outputs/metrics.json')
metrics_path.unlink(missing_ok=True)
subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
# False: baseline yang diuji lokal; True: tambah perbandingan CNN.
RUN_CNN = False
if RUN_CNN:
    subprocess.run([sys.executable,'-m','pip','install','torch','torchvision'],check=True)
command = [sys.executable,'-m','coffee.train','--images','data/gpiosenka_coffee-bean-dataset-resized-224-x-224','--audio','data/audio-lfs/datasets']
if RUN_CNN:
    command.append('--cnn')
subprocess.run(command,check=True)
Path('outputs/source_commit.txt').write_text(SOURCE_COMMIT + '\\n', encoding='utf-8')
RUN_COMPLETED = True
'''),n.v4.new_code_cell('''import json
if not globals().get('RUN_COMPLETED', False):
    raise RuntimeError('Jalankan sel training hingga selesai pada sesi ini sebelum menampilkan hasil.')
metrics = json.loads(Path('outputs/metrics.json').read_text())
for modality, result in metrics.items():
    if isinstance(result, dict):
        print(modality, 'selected:',result['selected'], 'test accuracy:',result['test_accuracy'])
        print('Audit:',result['audit'])
print(metrics['fusion'])
'''),n.v4.new_markdown_cell('''## Inferensi fusion opsional
Gunakan gambar dan WAV dari observasi roasting yang sama. Pasangan acak lintas dataset hanya menguji fungsi, bukan validasi multimodal. `--background-stage Medium` hanya bila Anda mengetahui observasi berada di antara first dan second crack. Tanpa konteks, massa probabilitas Background mengikuti gambar (audio abstain).'''),n.v4.new_code_cell('''# Contoh setelah upload input milik Anda:
# subprocess.run([sys.executable,'-m','coffee.predict','--image','bean.png','--audio','event.wav'],check=True)
import shutil
from google.colab import files
shutil.make_archive('/content/coffee-results','zip','outputs')
files.download('/content/coffee-results.zip')
''')]
visual_cells = [n.v4.new_markdown_cell('''## Visualisasi hasil run ini
Semua angka berikut dibaca dari `outputs/metrics.json` yang baru dihasilkan training pada sesi ini. Tidak ada hasil referensi atau angka lokal yang ditanam. Macro F1 memberi bobot sama pada setiap kelas. Gambar dan audio merupakan tugas berbeda, sehingga selisih skornya tidak membuktikan keunggulan satu modalitas.
'''), n.v4.new_code_cell('''import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

if not globals().get('RUN_COMPLETED', False):
    raise RuntimeError('Training run ini belum berhasil selesai.')
metrics = json.loads(Path('outputs/metrics.json').read_text())
results = {name: result for name, result in metrics.items() if isinstance(result, dict)}
titles = {'image': 'Gambar', 'audio': 'Audio', 'image_cnn': 'Gambar CNN'}
table = ['| Modalitas | Model terpilih | Jumlah test | Akurasi test | Macro F1 test |',
         '| --- | --- | ---: | ---: | ---: |']
for name, result in results.items():
    table.append(f"| {titles.get(name, name)} | {result['selected'].upper()} | {result['audit']['counts']['test']} | {result['test_accuracy']:.2%} | {result['test_report']['macro avg']['f1-score']:.2%} |")
display(Markdown('\\n'.join(table)))

fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
for ax, (name, result) in zip(axes[0], results.items()):
    scores = [result['test_accuracy'], result['test_report']['macro avg']['f1-score']]
    bars = ax.bar(['Akurasi', 'Macro F1'], scores, color=['#2563eb', '#0d9488'])
    ax.set_ylim(0, 1.12)
    ax.set_title(titles.get(name, name) + ' — ' + result['selected'].upper())
    ax.set_ylabel('Skor test (0–1)')
    ax.bar_label(bars, labels=[f'{score:.2%}' for score in scores], padding=4)
fig.suptitle('Evaluasi tiap modalitas pada test masing-masing')
fig.tight_layout()
fig.savefig('outputs/test_scores.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
'''), n.v4.new_code_cell('''# Warna menunjukkan persentase per kelas aktual; teks menunjukkan jumlah file.
if not globals().get('RUN_COMPLETED', False):
    raise RuntimeError('Training run ini belum berhasil selesai.')
fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
for ax, (name, result) in zip(axes[0], results.items()):
    labels = [key for key, value in result['test_report'].items()
              if isinstance(value, dict) and key not in ('macro avg', 'weighted avg', 'micro avg')]
    matrix = np.asarray(result['confusion_matrix'])
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, f'{matrix[row, col]}\\n({normalized[row, col]:.1%})',
                    ha='center', va='center', color='white' if normalized[row, col] > .5 else 'black')
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title(titles.get(name, name))
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label='Proporsi kelas aktual')
fig.suptitle('Confusion matrix test: diagonal = prediksi benar')
fig.tight_layout()
fig.savefig('outputs/test_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
'''), n.v4.new_code_cell('''if not globals().get('RUN_COMPLETED', False):
    raise RuntimeError('Training run ini belum berhasil selesai.')
lines = ['### Kesimpulan run ini', '']
for name, result in results.items():
    matrix = np.asarray(result['confusion_matrix'])
    labels = [key for key, value in result['test_report'].items()
              if isinstance(value, dict) and key not in ('macro avg', 'weighted avg', 'micro avg')]
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    lines.append(f"- **{titles.get(name, name)}**: {correct}/{total} prediksi benar, akurasi **{result['test_accuracy']:.2%}**, macro F1 **{result['test_report']['macro avg']['f1-score']:.2%}**. Model {result['selected'].upper()} dipilih dari validasi.")
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    if mistakes.max() > 0:
        row, col = np.unravel_index(mistakes.argmax(), mistakes.shape)
        lines.append(f"  Kesalahan terbanyak (salah satu jika seri): **{labels[row]} → {labels[col]}**, {mistakes[row, col]} file.")
    overlap = result['audit']['train_test_trial_overlap']
    if overlap:
        lines.append(f"  Ada **{len(overlap)} trial** yang beririsan antara train dan test; generalisasi ke trial baru belum dibuktikan.")
    duplicates = result['audit']['cross_split_identical_files']
    if duplicates:
        lines.append(f"  **Peringatan:** ditemukan {duplicates} salinan identik lintas split; hasil berisiko bocor.")
lines.extend(['', '**Fusion belum dievaluasi** karena tidak tersedia test berpasangan. Skor di atas adalah evaluasi masing-masing modalitas.'])
conclusion = '\\n'.join(lines)
display(Markdown(conclusion))
Path('outputs/conclusion.md').write_text(conclusion, encoding='utf-8')
''')]
cells[7:7] = visual_cells
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
n.validate(nb)
n.write(nb,'notebooks/Coffee_Roast_Colab.ipynb')
print('Notebook validated')
