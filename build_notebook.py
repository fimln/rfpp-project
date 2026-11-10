import json
from pathlib import Path
import nbformat as n
nb=n.v4.new_notebook()
files={str(p).replace('\\','/'):p.read_text(encoding='utf-8') for folder in ['coffee','tests'] for p in Path(folder).glob('*.py')}
files['requirements.txt']=Path('requirements.txt').read_text()
cells=[n.v4.new_markdown_cell('''# Coffee Roast Classification
Notebook mandiri, tidak perlu repository GitHub. Jalankan berurutan dengan **Run all**. Baseline CPU; opsional MobileNet memakai GPU jika tersedia. Dataset gambar dan audio tidak berpasangan, sehingga tidak ada klaim akurasi fusion. Background bukan bukti Green; Medium memerlukan konteks waktu eksternal.

Sumber: [Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224), [audio](https://github.com/Theseus-427/coffee-roasting-acoustic-dataset). Internet diperlukan. Unduhan audio memakai Git LFS agar bukan pointer WAV. Jika Kaggle membatasi akses, sel download gagal dengan pesan, bukan melanjutkan dengan data kosong.'''),n.v4.new_code_cell('''import sys
from pathlib import Path
import os
assert sys.version_info >= (3,12), 'Memerlukan Python 3.12 atau lebih baru'
ROOT = Path('/content/coffee-roast')
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)
'''),n.v4.new_code_cell('files = '+repr(files)+'''\nfor name, content in files.items():
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
'''),n.v4.new_code_cell('''%pip install -q -r requirements.txt
!apt-get -qq update
!apt-get -qq install -y git-lfs
import subprocess
subprocess.run(['git','lfs','install'],check=True)
'''),n.v4.new_code_cell('''import subprocess, sys
subprocess.run([sys.executable, '-m', 'coffee.download', '--destination', 'data'], check=True)
'''),n.v4.new_code_cell('''subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
# False: baseline yang diuji lokal; True: tambah perbandingan CNN.
RUN_CNN = False
if RUN_CNN:
    subprocess.run([sys.executable,'-m','pip','install','torch','torchvision'],check=True)
command = [sys.executable,'-m','coffee.train','--images','data/gpiosenka_coffee-bean-dataset-resized-224-x-224','--audio','data/audio-lfs/datasets']
if RUN_CNN:
    command.append('--cnn')
subprocess.run(command,check=True)
'''),n.v4.new_code_cell('''import json
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
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
n.validate(nb)
n.write(nb,'notebooks/Coffee_Roast_Colab.ipynb')
print('Notebook validated')
