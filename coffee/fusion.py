import numpy as np

ROASTS = ['Green', 'Light', 'Medium', 'Dark']


def fuse(image, audio, image_accuracy, audio_accuracy, *, background_stage=None):
    """Experimental heuristic; background needs externally observed roast context.

    background_stage: Green before first crack, Medium between cracks.
    Without context, audio abstains for Background and image evidence is retained.
    """
    image, audio = np.asarray(image, float), np.asarray(audio, float)
    for p, size in [(image, 4), (audio, 3)]:
        if p.shape != (size,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError('Expected normalized probability vector')
    if background_stage not in (None, 'Green', 'Medium'):
        raise ValueError('background_stage must be Green, Medium, or None')
    if not all(np.isfinite(a) and 0 <= a <= 1 for a in (image_accuracy, audio_accuracy)) or image_accuracy+audio_accuracy == 0:
        raise ValueError('Invalid validation accuracies')
    mapped = np.array([0., audio[1], 0., audio[2]])
    if background_stage:
        mapped[ROASTS.index(background_stage)] = audio[0]
    else:
        mapped += audio[0] * image
    return (image_accuracy * image + audio_accuracy * mapped) / (image_accuracy + audio_accuracy)
