import torch
from torch import nn

class TokenDrop(nn.Module):
    """
    For a batch of token indices, randomly replace non-special tokens.

    Args:
        prob (float): probability of dropping a token
        blank_token (int): index for the blank token (thường = pad_token_id)
        eos_token (int): index for the eos token (thường = sep_token_id)
    """
    def __init__(self, prob=0.1, blank_token=0, eos_token=102):
        super().__init__()
        self.prob = prob
        self.blank_token = blank_token
        self.eos_token = eos_token

    def __call__(self, sample):
        # sample: Tensor [batch_size, seq_len] chứa token IDs
        mask = torch.bernoulli(self.prob * torch.ones_like(sample, dtype=torch.float)).long()

        # không drop nếu là eos
        can_drop = (~(sample == self.eos_token)).long()
        mask = mask * can_drop

        # không drop token đầu tiên (thường là [CLS])
        mask[:, 0] = 0

        replace_with = self.blank_token * torch.ones_like(sample)

        sample_out = (1 - mask) * sample + mask * replace_with
        return sample_out