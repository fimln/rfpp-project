import argparse
import json
from pathlib import Path

import joblib

from .features import audio_features, image_features
from .fusion import ROASTS, fuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--models', type=Path, default=Path('outputs'))
    p.add_argument('--background-stage', choices=['Green', 'Medium'])
    args = p.parse_args()
    image = joblib.load(args.models / 'image.joblib')
    audio = joblib.load(args.models / 'audio.joblib')
    pi = image['model'].predict_proba([image_features(args.image)])[0]
    pa = audio['model'].predict_proba([audio_features(args.audio)])[0]
    result = fuse(pi, pa, image['validation_accuracy'], audio['validation_accuracy'], background_stage=args.background_stage)
    print(json.dumps({'image': dict(zip(ROASTS, pi)), 'audio': dict(zip(audio['classes'], pa)),
                      'experimental_fusion': dict(zip(ROASTS, result)), 'prediction': ROASTS[result.argmax()]}, indent=2))


if __name__ == '__main__':
    main()
