import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn import util
from torch.nn import SmoothL1Loss


def calc_entropy_loss(select_probs, passage_mask):
    legal_select_probs = select_probs[(select_probs > 0) & (select_probs < 1)]
    entropies = -(legal_select_probs * torch.log(legal_select_probs) + (1 - legal_select_probs) * torch.log(
        1 - legal_select_probs))

    seq_length = passage_mask.sum().float()
    mean_entropy = entropies.sum() / seq_length

    return mean_entropy


def calc_entropy_loss_unmasked(select_probs_logits, mask):
    select_probs = util.masked_softmax(select_probs_logits, mask)
    epsilon = 1e-7
    select_probs[(select_probs < epsilon)] = epsilon
    select_probs[(select_probs > (1 - epsilon))] = 1 - epsilon

    legal_select_probs = select_probs
    entropies = -(legal_select_probs * torch.log(legal_select_probs) + (1 - legal_select_probs) * torch.log(
        1 - legal_select_probs))

    seq_length = select_probs.shape[-1]
    mean_entropy_per_sentence = entropies.sum(dim=2) / seq_length
    total_entropy = mean_entropy_per_sentence.sum()

    return total_entropy


class EntropyLoss(nn.Module):
    """
    Summed over each component in input, averaged over all entries in batch
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, mask):
        """ x assumed to be logits"""
        # Softmax will convert logits to probabilities, this version is numerically stable
        b = util.masked_softmax(x, mask) * util.masked_log_softmax(x, mask)
        non_batch_dimensions = tuple(range(-len(b.shape) + 1, 0))
        b = -1.0 * b.sum(dim=non_batch_dimensions)
        return b.mean()


_huber_loss = SmoothL1Loss()
_entropy_loss = EntropyLoss()


def count_loss(answer_as_counts, count_number, count_probability_logits, count_probability_mask,
               entropy_loss_weight=1.0):
    # Count answers are padded with label -1,
    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
    # Shape: (batch_size, # of count answers)
    gold_count_mask = (answer_as_counts != -1).long()
    # Shape: (batch_size,)
    gold_counts_masked = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)
    count_number_masked = util.replace_masked_values(count_number, gold_count_mask, 0)

    aggregated_count_loss = _huber_loss(count_number_masked, gold_counts_masked.float())
    entropy_loss = _entropy_loss(count_probability_logits, count_probability_mask)

    return (aggregated_count_loss) + (entropy_loss * entropy_loss_weight)
