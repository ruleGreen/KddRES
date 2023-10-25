import torch.nn as nn
import torch
from transformers import BertModel


class HierBert(nn.Module):
    def __init__(
        self,
        context_encoder: BertModel,
        desc_encoder: BertModel,
        n_intent: int,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.desc_encoder = desc_encoder
        self.intent_head = nn.Linear(
            context_encoder.config.hidden_size, n_intent
        )

    def forward(
        self,
        context_ids: torch.Tensor,
        context_attn_mask: torch.Tensor,
        main_slot_desc_ids: torch.Tensor,
        main_slot_desc_attn_mask: torch.Tensor,
        second_slot_desc_ids: torch.Tensor = None,
        second_slot_desc_attn_mask: torch.Tensor = None,
    ):
        """_summary_

        Args:
            context_ids (torch.Tensor): _description_
            context_attn_mask (torch.Tensor): _description_
            main_slot_desc_ids (torch.Tensor): _description_
            main_slot_desc_attn_mask (torch.Tensor): _description_
            second_slot_desc_ids (torch.Tensor, optional): _description_.
                Defaults to None.
            second_slot_desc_attn_mask (torch.Tensor, optional): _description_.
                Defaults to None.

        Returns (keys):
            intent_logits: tensor of shape (B, n_intent)
            main_slot_logits: tensor of shape (B, T, n_main_slot)
            second_slot_logits: tensor of shape (B, T, n_second_slot).
                It's None if `second_slot_desc_ids` is not provided.
        """
        outputs = self.context_encoder(
            input_ids=context_ids, attention_mask=context_attn_mask
        )
        # shape: (B, h)
        context_h = outputs[1]
        # shape: (B, n_intent)
        intent_logits = self.intent_head(context_h)

        B, T, h = outputs[0].shape
        # shape: (B*T, h)
        last_h = outputs[0].reshape(-1, h)
        outputs = self.desc_encoder(
            input_ids=main_slot_desc_ids,
            attention_mask=main_slot_desc_attn_mask,
        )
        # shape: (n_main_desc, h)
        main_desc_h = outputs[1]
        # shape: (B*T, n_main_slot)
        main_slot_logits = last_h @ main_desc_h.permute(1, 0)
        # shape: (B, T, n_main_slot)
        main_slot_logits = main_slot_logits.reshape(B, T, -1)

        second_slot_logits = None
        if second_slot_desc_ids is not None:
            outputs = self.desc_encoder(
                input_ids=second_slot_desc_ids,
                attention_mask=second_slot_desc_attn_mask,
            )
            # shape: (n_second_desc, h)
            second_desc_h = outputs[1]
            # shape: (B*T, n_second_slot)
            second_slot_logits = last_h @ second_desc_h.permute(1, 0)
            # shape: (B, T, n_second_slot)
            second_slot_logits = second_slot_logits.reshape(B, T, -1)
        return {
            "intent_logits": intent_logits,
            "main_slot_logits": main_slot_logits,
            "second_slot_logits": second_slot_logits,
        }
