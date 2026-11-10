import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from coffee.download import safe_extract
from coffee.features import audio_features, image_features
from coffee.fusion import fuse
from coffee.train import audit, deduplicate, discover, image_rows


def test_image_features(tmp_path):
    path = tmp_path / 'red.png'
    Image.new('RGB', (224, 224), (255, 0, 0)).save(path)
    result = image_features(path)
    assert result.shape == (108,)
    assert np.isfinite(result).all()
    assert result[16] == 1


def test_audio_features(tmp_path):
    path = tmp_path / 'silence.wav'
    wavfile.write(path, 44100, np.zeros(4410, np.int16))
    result = audio_features(path)
    assert result.shape == (15,)
    assert np.isfinite(result).all()
    assert result[0] == .1
    assert result[1] == 0


def test_fusion():
    image = np.array([.1, .2, .6, .1])
    np.testing.assert_allclose(fuse(image, [1, 0, 0], .8, .9), image)
    result = fuse(image, [1, 0, 0], .8, .9, background_stage='Medium')
    assert result.argmax() == 2
    assert result.sum() == pytest.approx(1)
    assert fuse(image, [0, 0, 1], .8, .9).argmax() == 3
    with pytest.raises(ValueError):
        fuse(image, [1, 1, 1], .8, .9)


def test_zip_traversal(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr('../escape.txt', 'bad')
    data.seek(0)
    with pytest.raises(ValueError):
        safe_extract(data, tmp_path)


def test_lfs_pointer_rejected(tmp_path):
    path = tmp_path / 'pointer.wav'
    path.write_text('version https://git-lfs.github.com/spec/v1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Git LFS pointer'):
        audio_features(path)


def test_zip_extract(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr('nested/sample.txt', 'coffee')
    data.seek(0)
    safe_extract(data, tmp_path)
    assert (tmp_path / 'nested/sample.txt').read_text() == 'coffee'


def test_local_datasets():
    root = Path('gpiosenka_coffee-bean-dataset-resized-224-x-224')
    if not root.exists():
        root = Path('data/gpiosenka_coffee-bean-dataset-resized-224-x-224')
    audio = Path('data/audio-lfs/datasets')
    if not root.exists() or not audio.exists():
        pytest.skip('Local datasets absent; feature and fusion tests still run')
    rows = image_rows(root)
    assert len(rows) == 1194  # 1600 source files minus 406 duplicate copies.
    assert rows == image_rows(root)
    assert len({r['path'] for r in rows}) == len(rows)
    assert audit(rows)['cross_split_identical_files'] == 0
    for split in ['train', 'val', 'test']:
        assert {r['label'] for r in rows if r['split'] == split} == {0, 1, 2, 3}
    sounds = discover(audio, ['0', '1', '2'], {'.wav'})
    assert len(sounds) == 4304  # Actual upstream files; README states 4301.
    assert audit(deduplicate(sounds))['cross_split_identical_files'] == 0
    assert np.isfinite(audio_features(sounds[0]['path'])).all()
    assert np.isfinite(image_features(rows[0]['path'])).all()
