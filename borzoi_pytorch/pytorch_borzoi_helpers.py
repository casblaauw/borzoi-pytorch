import numpy as np
import torch

def predict_tracks(models, sequence_one_hot, head_i, slices):
    if sequence_one_hot.ndim == 2:
        sequence_one_hot = sequence_one_hot[None, ...]
    with torch.no_grad():
        predicted_tracks = [m(sequence_one_hot)[head_i].numpy(force=True)[:, slices, :] for m in models]
    return np.concatenate(predicted_tracks, axis = 0)
