import torch.nn as nn
import torch
from transformers import BertModel


class BertTagger(nn.Module):
    def __init__(
        self,
        encoder: BertModel,
        n_intent: int,
        n_slot: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.intent_head = nn.Linear(encoder.config.hidden_size, n_intent)
        self.slot_head = nn.Linear(encoder.config.hidden_size, n_slot)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_attn_mask: torch.Tensor,
    ):
        """_summary_

        Args:
            context_ids (torch.Tensor): _description_
            context_attn_mask (torch.Tensor): _description_

        Returns (keys):
            intent_logits: tensor of shape (B, n_intent)
            slot_logits: tensor of shape (B, T, n_slot)

        """
        outputs = self.encoder(
            input_ids=context_ids, attention_mask=context_attn_mask
        )
        # shape: (B, h)
        context_h = outputs[1]
        # shape: (B, n_intent)
        intent_logits = self.intent_head(context_h)

        # shape: (B, T, h)
        last_h = outputs[0]
        # shape: (B, T, n_slot)
        slot_logits = self.slot_head(last_h)

        return {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }
