from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import List
import json
import torch

"""
  {
    "raw_context": "歡迎光臨耀記美食!",
    "tokenized_context": [
      "歡",
      "迎",
      "光",
      "臨",
      "耀",
      "記",
      "美",
      "食",
      "!"
    ],
    "slot_labels": [
      "O",
      "O",
      "O",
      "O",
      "B-Inform+餐館+名稱",
      "I-Inform+餐館+名稱",
      "I-Inform+餐館+名稱",
      "I-Inform+餐館+名稱",
      "O"
    ],
    "intent_labels": [
      "General+餐館+greet+none"
    ],
    "dialog_act": [
      [
        "General",
        "餐館",
        "greet",
        "none"
      ],
      [
        "Inform",
        "餐館",
        "名稱",
        "耀記美食"
      ]
    ]
  },
"""


class KddRESDataset(Dataset):
    def __init__(
        self, mode: str, tokenizer: PreTrainedTokenizer, n_samples: int = -1
    ):
        file_path = f"data/kddres/kddres_nlu_{mode}.json"
        self.data = json.load(open(file_path, "r"))
        if n_samples > 0:
            self.data[:n_samples]
        self.slots = open("data/kddres/slots.txt", "r").read().splitlines()
        self.intents = open("data/kddres/intents.txt", "r").read().splitlines()
        self.desc = json.load(open("data/kddres/slots_desc.json", "r"))
        self.tokenizer = tokenizer

        main_slots = []
        second_slots = []
        self.main2second = {}
        self.slot2desc = {"O": self.desc["O"]}

        for slot in self.slots[1:]:
            b, i, d, sv = slot.split("+")
            desc = self._get_desc(b, i, d, sv)
            self.slot2desc[slot] = desc
            idsv = "+".join([i, d, sv])
            if (
                idsv
                not in [
                    "Inform+餐館+停車-收費",
                    "Inform+餐館+外賣-配送時間",
                    "Info-confirm+餐館+桌位-桌位類型",
                ]
                and "-" in sv
            ):
                second_slots.append(slot)
                main, second = sv.split("-")
                desc = self._get_desc(b, i, d, main)
                main = "+".join([b, i, d, main])
                self.slot2desc[main] = desc
                main_slots.append(main)
                self.main2second.setdefault(main, []).append(slot)
            else:
                main_slots.append(slot)

        main_slots = ["O"] + sorted(list(set(main_slots)))
        self.main_slots = main_slots
        self.second_slots = second_slots
        self.slots = main_slots + second_slots
        self.n_main_slot = len(main_slots)
        # second_slot_id starts at 0
        self.second2main = {
            second_slots.index(v): main_slots.index(k)
            for k, vs in self.main2second.items()
            for v in vs
        }
        self.main2second = {
            main_slots.index(k): [second_slots.index(v) for v in vs]
            for k, vs in self.main2second.items()
        }
        main_slot_desc = [self.slot2desc[slot] for slot in main_slots]
        second_slot_desc = [self.slot2desc[slot] for slot in second_slots]
        cls_token_id = self.tokenizer.cls_token_id
        main_slot_desc_ids = [
            torch.tensor(
                [cls_token_id]
                + tokenizer.encode(desc, add_special_tokens=False),
                dtype=torch.long,
            )
            for desc in main_slot_desc
        ]
        second_slot_desc_ids = [
            torch.tensor(
                [cls_token_id]
                + tokenizer.encode(desc, add_special_tokens=False),
                dtype=torch.long,
            )
            for desc in second_slot_desc
        ]
        self.main_slot_desc_ids = pad_sequence(
            main_slot_desc_ids,
            padding_value=tokenizer.pad_token_id,
            batch_first=True,
        )
        self.main_slot_desc_attn_mask = (
            self.main_slot_desc_ids != tokenizer.pad_token_id
        ).long()

        self.second_slot_desc_ids = pad_sequence(
            second_slot_desc_ids,
            padding_value=tokenizer.pad_token_id,
            batch_first=True,
        )
        self.second_slot_desc_attn_mask = (
            self.second_slot_desc_ids != tokenizer.pad_token_id
        ).long()
        self.n_intent = len(self.intents)

    def __getitem__(self, index):
        # keys: raw_context, tokenized_context, slot_labels, intent_labels
        cls_token_id = self.tokenizer.cls_token_id
        intents_str: List[str] = self.data[index]["intent_labels"]
        if intents_str == []:
            intents = torch.tensor([0], dtype=torch.long)
        else:
            intents = torch.tensor(
                [self.intents.index(i) for i in intents_str], dtype=torch.long
            )
        intents_idx = torch.zeros(len(self.intents), dtype=torch.long)
        intents_idx[intents] = 1

        context: List[str] = self.data[index]["tokenized_context"]
        slots = ["O"] + self.data[index]["slot_labels"]
        slot_ids = [self.slots.index(s) for s in slots]
        main_slot_ids = [
            slot_id
            if slot_id < self.n_main_slot
            else self.second2main[slot_id - self.n_main_slot]
            for slot_id in slot_ids
        ]
        second_slot_ids = [
            -100 if slot_id < self.n_main_slot else slot_id - self.n_main_slot
            for slot_id in slot_ids
        ]
        context_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(
            context
        )
        return (
            self.data[index]["raw_context"],
            self.data[index]["tokenized_context"],
            self.data[index]["dialog_act"],
            slots,
            torch.tensor(context_ids, dtype=torch.long),
            torch.tensor(slot_ids, dtype=torch.long),
            torch.tensor(main_slot_ids, dtype=torch.long),
            torch.tensor(second_slot_ids, dtype=torch.long),
            intents_str,
            intents_idx,
        )

    def collate_fn(self, batch):
        """

        raw_context: List of str.
        dialog_act: List of dialog acts. Each dialog acts is a list of 4 str as
            usual.
        context_ids: Tensor of shape (B, T).
        slot_ids: Tensor of shape (B, T).
        intent_idx: Tensor of shape (B, n_intents).

        """
        (
            raw_context,
            tokenized_context,
            dialog_act,
            slots,
            context_ids,
            slot_ids,
            main_slot_ids,
            second_slot_ids,
            intents_str,
            intent_idx,
        ) = zip(*batch)
        # shape: (B, T)
        context_ids = pad_sequence(
            sequences=context_ids,
            padding_value=self.tokenizer.pad_token_id,
            batch_first=True,
        )
        context_attn_mask = (context_ids != self.tokenizer.pad_token_id).long()
        # shape: (B, T)
        slot_ids = pad_sequence(
            sequences=slot_ids, padding_value=-100, batch_first=True
        )
        main_slot_ids = pad_sequence(
            sequences=main_slot_ids, padding_value=-100, batch_first=True
        )
        second_slot_ids = pad_sequence(
            sequences=second_slot_ids, padding_value=-100, batch_first=True
        )
        # shape(B, n_intent)
        intent_idx = torch.stack(intent_idx, dim=0)
        return {
            "raw_context": raw_context,
            "tokenized_context": tokenized_context,
            "raw_dialog_act": dialog_act,
            "raw_slots": slots,
            "context_ids": context_ids,
            "context_attn_mask": context_attn_mask,
            "slot_ids": slot_ids,
            "main_slot_ids": main_slot_ids,
            "second_slot_ids": second_slot_ids,
            "main_slot_desc_ids": self.main_slot_desc_ids,
            "main_slot_desc_attn_mask": self.main_slot_desc_attn_mask,
            "second_slot_desc_ids": self.second_slot_desc_ids,
            "second_slot_desc_attn_mask": self.second_slot_desc_attn_mask,
            "intent_str": intents_str,
            "intent_idx": intent_idx,
            "main2second": self.main2second,
        }

    def __len__(self):
        return len(self.data)

    def _get_desc(self, b, i, d, s):
        desc = self.desc[s]
        if i == "Inform":
            desc = "通知" + desc
        elif i == "Order-confirm":
            desc = "確定訂位" + desc
        elif i == "Info-confirm":
            desc = "查詢確認" + desc
        else:
            raise ValueError(f"Unidentified intent {i}")

        if b == "B":
            desc = "字開端，" + desc
        elif b == "I":
            desc = "字中間，" + desc
        else:
            raise ValueError(f"Unidentified begin tag {b}")
        return desc


if __name__ == "__main__":
    from transformers import BertTokenizer

    t = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    d = KddRESDataset(mode="test", tokenizer=t)
    # d[4]
    d[6]
