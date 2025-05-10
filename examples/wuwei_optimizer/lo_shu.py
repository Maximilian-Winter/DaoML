# Define the Lo Shu magic square
import torch

lo_shu = torch.tensor([
    [4, 9, 2],
    [3, 5, 7],
    [8, 1, 6]
]).float()

# Normalize to range [0.5, 1.0] to avoid zeroing out connections
lo_shu_normalized = 0.5 + 0.5 * (lo_shu - lo_shu.min()) / (lo_shu.max() - lo_shu.min())