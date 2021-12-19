
import torch
import torch.nn as nn

class NCorrectPredictions(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, wngy, semg):
        out_channels = logits.shape[1]
        logits = logits.permute(0, 2, 3, 1).reshape(-1, out_channels)
        wngy = wngy.permute(0, 2, 3, 1).reshape(-1, 1)
        semg = semg.permute(0, 2, 3, 1).reshape(-1).type(torch.bool)

        wngy_pred = logits.argmax(dim=1).reshape(-1, 1)
        return (wngy == wngy_pred)[semg, :].sum()




