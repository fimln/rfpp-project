"""Download public ZIPs, validate archive paths, extract without shell tools."""
import argparse
import hashlib
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    'image': ('https://www.kaggle.com/api/v1/datasets/download/gpiosenka/coffee-bean-dataset-resized-224-x-224?datasetVersionNumber=1', 'gpiosenka_coffee-bean-dataset-resized-224-x-224'),
    'audio': ('https://codeload.github.com/Theseus-427/coffee-roasting-acoustic-dataset/zip/refs/heads/main', '.')}


def safe_extract(archive, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (target / member.filename).resolve().is_relative_to(target):
                raise ValueError('Unsafe archive path')
        z.extractall(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, default=Path('data'))
    p.add_argument('--skip-images', action='store_true', help='Use already available local images; only fetch real audio objects')
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, (url, folder) in SOURCES.items():
        if name == 'image' and args.skip_images:
            continue
        if name == 'audio':
            target = args.destination / 'audio-lfs'
            subprocess.run(['git', 'lfs', 'version'], check=True)
            if not (target / '.git').exists():
                subprocess.run(['git', 'clone', 'https://github.com/Theseus-427/coffee-roasting-acoustic-dataset.git', str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'lfs', 'pull'], check=True)
            commit = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            receipts[name] = {'repository': 'https://github.com/Theseus-427/coffee-roasting-acoustic-dataset', 'commit': commit}
            continue
        archive = args.destination / f'{name}.zip'
        if not archive.exists() or not zipfile.is_zipfile(archive):
            print(f'Downloading {name}', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'coffee-roast-research/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as f:
                while chunk := response.read(1024*1024):
                    f.write(chunk)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(f'{name}: server did not return ZIP; check dataset access at {url}')
        safe_extract(archive, args.destination / folder)
        receipts[name] = {'url': url, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (args.destination / 'download_receipts.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
