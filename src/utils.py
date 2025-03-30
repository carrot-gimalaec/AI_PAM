import torch
from torch import Tensor


def last_logit_pool(logits: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last logit.

    Args:
        logits (torch.Tensor): The output logits of the model.
        attention_mask (torch.Tensor): Attention mask.

    Returns:
        torch.Tensor: The tensor after pooling.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return logits[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack(
            [logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0
        )
