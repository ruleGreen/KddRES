import sys
import os

parent_folder = os.path.abspath(__file__)
for _ in range(2):
    parent_folder = os.path.dirname(parent_folder)
sys.path.append(parent_folder)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import importlib
import torchmetrics
import wandb
import json
import shutil
from pprint import pprint
from utils.metric import SpanF1
from typing import List, Dict, Iterable, Any
from utils.util import getlogger, set_seed
from models.bert import BertTagger
from tqdm import tqdm
from data.dataset import KddRESDataset
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertModel,
    PreTrainedTokenizer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer(config: Dict[str, any]):
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_model_name"])
    return tokenizer


def get_model(config: Dict[str, any]):
    encoder = BertModel.from_pretrained(config["pretrained_model_name"])
    model = BertTagger(
        encoder=encoder,
        n_slot=config["n_slot"],
        n_intent=config["n_intent"],
    )
    return model


def get_dataset(
    config: Dict[str, any], mode: str, tokenizer: PreTrainedTokenizer
):
    return KddRESDataset(mode=mode, tokenizer=tokenizer)


def get_optim(
    config: Dict[str, Any], model: nn.Module, num_training_steps: int
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        eps=config["adam_epsilon"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_step_ratio"] * num_training_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, help="The random seed", default=2048
    )
    parser.add_argument(
        "--save_prefix", help="The prefix of the save folder", type=str
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Directory path for checkpoints.",
        default="saved/",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project_name", type=str, help="Name for wandb")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_val", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="Location of the checkpoint for evaluation.",
    )

    # dataset
    parser.add_argument(
        "--desc_path",
        type=str,
        help="Path of the slot descriptions",
        default="data/slots_desc.json",
    )

    # optim and scheduler
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--warmup_step_ratio",
        default=0,
        type=float,
        help="Portion of steps to do warmup. Default is 10% following Bert.",
    )
    parser.add_argument(
        "--lr_decay",
        action="store_true",
        help="If true, it will linear decay the lr to 0 at the end.",
    )

    # model
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        help="The pretrained model or path.",
    )

    # train
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--epoch_per_val",
        type=int,
        help="How many epochs to train before one validation.",
        default=1,
    )
    parser.add_argument(
        "--patience",
        type=int,
        help=(
            "Number of epochs with no improvement after which training will be"
            " stopped"
        ),
        default=1e17,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of training epochs",
        default=5,
    )

    parser.add_argument("--validate_before_train", action="store_true")

    args = parser.parse_args()
    return args


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        slots: List[str],
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler = None,
    ):
        """
        Optimizer can be None if constructed only for testing.
        """
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.slots = slots
        self.slotid2str = {i: slot for i, slot in enumerate(slots)}
        self.n_slot = len(slots)

        self.slot_loss_fn = nn.CrossEntropyLoss()
        self.intent_loss_fn = nn.BCEWithLogitsLoss()
        self.intent_f1 = torchmetrics.classification.MultilabelF1Score(
            num_labels=config["n_intent"], average="micro"
        ).to(DEVICE)
        self.slot_f1 = SpanF1()
        self.main_slot_f1 = SpanF1()
        self.second_slot_f1 = SpanF1()

    def _reset_metrics(self):
        self.intent_f1.reset()
        self.slot_f1.reset()
        self.main_slot_f1.reset()
        self.second_slot_f1.reset()

    def training_step(
        self, batch_dict: Dict[str, torch.Tensor], batch_i: int
    ) -> Dict[str, Any]:
        """Define a update step of the model.

        Args:
            batch_dict: Input for the model.
            batch_idx: Current batch idx of this epoch.

        Returns:
            Dict with book keeping variables:

        """
        self.optimizer.zero_grad()
        outputs = self.model(
            context_ids=batch_dict["context_ids"].to(DEVICE),
            context_attn_mask=batch_dict["context_attn_mask"].to(DEVICE),
        )
        # intent_logits: shape (B, n_intent)
        # intent_idx: shape (B, n_intent)
        intent_loss = self.intent_loss_fn(
            outputs["intent_logits"],
            batch_dict["intent_idx"].float().to(DEVICE),
        )
        slot_loss = self.slot_loss_fn(
            outputs["slot_logits"].reshape(-1, config["n_slot"]),
            batch_dict["slot_ids"].reshape(-1).to(DEVICE),
        )
        loss = slot_loss + intent_loss
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        intent_f1 = self.intent_f1(
            outputs["intent_logits"], batch_dict["intent_idx"].to(DEVICE)
        )

        # shape: (B, T)
        # slot_pred = torch.zeros_like(batch_dict["context_ids"]).to(
        #     DEVICE
        # ) + self.slots.index("O")
        # attn_mask = batch_dict["context_attn_mask"].bool().to(DEVICE)
        # slot_pred[attn_mask] = outputs["slot_logits"][attn_mask].max(dim=1)[1]
        # slot_pred_str = [
        #     [self.slotid2str[slot_id.item()] for slot_id in pred]
        #     for pred in slot_pred
        # ]
        # slot_f1 = self.slot_f1(slot_pred_str, batch_dict["raw_slots"])
        return {
            "loss": loss.item(),
            "intent_f1": intent_f1.item(),
            # "slot_f1": slot_f1,
            "batch_i": batch_i,
        }

    def validating_step(
        self, batch_dict: Dict[str, torch.Tensor], batch_i: int
    ) -> Dict[str, Any]:
        outputs = self.model(
            context_ids=batch_dict["context_ids"].to(DEVICE),
            context_attn_mask=batch_dict["context_attn_mask"].to(DEVICE),
        )
        intent_preds = outputs["intent_logits"] > 0.5
        intent_f1 = self.intent_f1(
            outputs["intent_logits"], batch_dict["intent_idx"].to(DEVICE)
        )

        attn_mask = batch_dict["context_attn_mask"].bool().to(DEVICE)
        pred_ids = outputs["slot_logits"][attn_mask].max(dim=1)[1]

        slot_pred = torch.zeros_like(batch_dict["context_ids"]).to(
            DEVICE
        ) + self.slots.index("O")
        slot_pred[attn_mask] = pred_ids
        slot_pred_str = [
            [self.slotid2str[slot_id.item()] for slot_id in pred]
            for pred in slot_pred
        ]
        slot_f1 = self.slot_f1(slot_pred_str, batch_dict["raw_slots"])

        main_slot_pred = torch.zeros_like(batch_dict["context_ids"]).to(
            DEVICE
        ) + self.slots.index("O")
        main_pred_ids = pred_ids.clone()
        main_pred_ids[pred_ids >= config["n_main_slot"]] = self.slots.index(
            "O"
        )
        main_slot_pred[attn_mask] = main_pred_ids
        main_slot_pred_str = [
            [self.slotid2str[slot_id.item()] for slot_id in pred]
            for pred in main_slot_pred
        ]
        main_slot_label_str = [
            [
                self.slotid2str[slot_id.item()]
                if slot_id < config["n_main_slot"] and slot_id != -100
                else "O"
                for slot_id in slot_ids
            ]
            for slot_ids in batch_dict["slot_ids"]
        ]
        main_slot_f1 = self.main_slot_f1(
            main_slot_pred_str, main_slot_label_str
        )

        second_slot_pred = torch.zeros_like(batch_dict["context_ids"]).to(
            DEVICE
        ) + self.slots.index("O")
        second_pred_ids = pred_ids.clone()
        second_pred_ids[pred_ids < config["n_main_slot"]] = self.slots.index(
            "O"
        )
        second_slot_pred[attn_mask] = second_pred_ids
        second_slot_pred_str = [
            [self.slotid2str[slot_id.item()] for slot_id in pred]
            for pred in second_slot_pred
        ]
        second_slot_label_str = [
            [
                "O"
                if slot_id < config["n_main_slot"] or slot_id == -100
                else self.slotid2str[slot_id.item()]
                for slot_id in slot_ids
            ]
            for slot_ids in batch_dict["slot_ids"]
        ]
        second_slot_f1 = self.second_slot_f1(
            second_slot_pred_str, second_slot_label_str
        )
        return {
            "intent_f1": intent_f1.item(),
            "slot_f1": slot_f1,
            "intent_pred": [
                [config["intents"][i.item()] for i in pred.nonzero()]
                for pred in intent_preds
            ],
            "intent_label": batch_dict["intent_str"],
            "slot_pred_str": slot_pred_str,
            "slot_label_str": batch_dict["raw_slots"],
            "main_slot_f1": main_slot_f1,
            "main_slot_pred_str": main_slot_pred_str,
            "main_slot_label_str": main_slot_label_str,
            "second_slot_f1": second_slot_f1,
            "second_slot_pred_str": second_slot_pred_str,
            "second_slot_label_str": second_slot_label_str,
            "raw_context": batch_dict["raw_context"],
            "tokenized_context": batch_dict["tokenized_context"],
        }

    def testing_step(
        self, batch_dict: Dict[str, torch.Tensor], batch_i: int
    ) -> Dict[str, Any]:
        return self.validating_step(batch_dict, batch_i)

    def test(self, test_dl: Iterable):
        bar = tqdm(test_dl, desc="Testing")
        self.model.eval()
        test_outputs = []
        self._reset_metrics()
        for i, batch_dict in enumerate(bar):
            with torch.no_grad():
                output = self.testing_step(batch_dict, i)
            test_outputs.append(output)
            intent_f1 = self.intent_f1.compute().item()
            slot_f1 = self.slot_f1.compute()
            main_slot_f1 = self.main_slot_f1.compute()
            second_slot_f1 = self.second_slot_f1.compute()
            bar.set_postfix(
                {
                    "test/intent_f1": f"{intent_f1:.3f}",
                    "test/slot_f1": f"{slot_f1:.3f}",
                    "test/main_slot_f1": f"{main_slot_f1:.3f}",
                    "test/second_slot_f1": f"{second_slot_f1:.3f}",
                }
            )
        if config["use_wandb"]:
            wandb.run.summary["test_intent_f1"] = intent_f1
            wandb.run.summary["test_slot_f1"] = slot_f1
            wandb.run.summary["test_main_slot_f1"] = main_slot_f1
            wandb.run.summary["test_second_slot_f1"] = second_slot_f1
        return test_outputs

    def validate(self, val_dl: Iterable):
        bar = tqdm(val_dl, desc="Validating")
        self.model.eval()
        val_outputs = []
        for i, batch_dict in enumerate(bar):
            with torch.no_grad():
                output = self.validating_step(batch_dict, i)
            val_outputs.append(output)
            intent_f1 = self.intent_f1.compute().item()
            slot_f1 = self.slot_f1.compute()
            main_slot_f1 = self.main_slot_f1.compute()
            second_slot_f1 = self.second_slot_f1.compute()
            bar.set_postfix(
                {
                    "val/intent_f1": f"{intent_f1:.3f}",
                    "val/slot_f1": f"{slot_f1:.3f}",
                    "val/main_slot_f1": f"{main_slot_f1:.3f}",
                    "val/second_slot_f1": f"{second_slot_f1:.3f}",
                }
            )
        return val_outputs

    def fit(self, train_dl: Iterable, val_dl: Iterable = None):
        best_score = 0
        best_slot_f1 = 0
        best_intent_f1 = 0
        best_path = None
        wait = 0
        global_step = 0
        if config["validate_before_train"] and val_dl is not None:
            self.validate(val_dl)
            set_seed(config["seed"])
        for epoch_i in range(1, config["n_epochs"] + 1):
            bar = tqdm(
                train_dl, desc=f"Train epoch {epoch_i}/{config['n_epochs']}"
            )
            self.model.train()
            train_outputs = []
            self._reset_metrics()
            for i, batch_dict in enumerate(bar):
                output = self.training_step(batch_dict, i)
                global_step += 1
                train_outputs.append(output)
                intent_f1 = self.intent_f1.compute().item()
                # slot_f1 = self.slot_f1.compute()
                avg_loss = np.mean([o["loss"] for o in train_outputs])
                bar.set_postfix(
                    {
                        "train/loss": f"{avg_loss:.3f}",
                        "train/intent_f1": f"{intent_f1:.3f}",
                        # "train/slot_f1": f"{slot_f1:.3f}",
                    }
                )
                if config["use_wandb"]:
                    wandb.log(
                        {
                            "train/loss": output["loss"],
                            "train/intent_f1": output["intent_f1"],
                            # "train/slot_f1": output["slot_f1"],
                            "trainer/global_step": global_step,
                            "trainer/learning_rate": self.optimizer.param_groups[
                                0
                            ][
                                "lr"
                            ],
                        }
                    )
            # train epoch statistics
            if config["use_wandb"]:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/intent_f1": intent_f1,
                        # "train/slot_f1": slot_f1,
                        "epoch": epoch_i,
                        "trainer/learning_rate": self.optimizer.param_groups[
                            0
                        ]["lr"],
                    }
                )

            if val_dl is not None and epoch_i % config["epoch_per_val"] == 0:
                self._reset_metrics()
                val_outputs = self.validate(val_dl)
                slot_f1 = self.slot_f1.compute()
                main_slot_f1 = self.main_slot_f1.compute()
                second_slot_f1 = self.second_slot_f1.compute()
                intent_f1 = self.intent_f1.compute().item()
                combine_score = (slot_f1 + intent_f1) / 2
                if config["use_wandb"]:
                    wandb.log(
                        {
                            "val/intent_f1": intent_f1,
                            "val/slot_f1": slot_f1,
                            "val/main_slot_f1": main_slot_f1,
                            "val/second_slot_f1": second_slot_f1,
                            "epoch": epoch_i,
                            "trainer/global_step": global_step,
                        }
                    )

                if combine_score > best_score:
                    logger.info(
                        "combine scored improved by"
                        f" {(combine_score - best_score):.3f}"
                    )
                    best_score = combine_score
                    best_slot_f1 = slot_f1
                    best_intent_f1 = intent_f1
                    wait = 0
                    # save model
                    if config["save_prefix"] is not None:
                        if best_path is not None:
                            if os.path.exists(best_path):
                                shutil.rmtree(best_path)
                                logger.info(f"Removed old save {best_path}.")
                        best_path = os.path.join(
                            config["save_path"],
                            (
                                f"{config['save_prefix']}"
                                f"-epoch={epoch_i:02d}"
                                f"-slot_f1={slot_f1:.3f}"
                                f"-intent_f1={intent_f1:.3f}"
                            ),
                        )
                        os.makedirs(best_path, exist_ok=True)
                        torch.save(
                            {"model": self.model.state_dict()},
                            os.path.join(best_path, "ckpt.pth"),
                        )
                        json.dump(
                            config,
                            open(
                                os.path.join(best_path, "config.json"),
                                "w",
                            ),
                        )
                        logger.info(
                            f"Saved best with best score: {best_score:.2f}"
                        )
                    if config["use_wandb"]:
                        wandb.run.summary["best_path"] = best_path
                        wandb.run.summary["best_combine_score"] = best_score
                        wandb.run.summary["best_slot_f1"] = best_slot_f1
                        wandb.run.summary["best_intent_f1"] = best_intent_f1
                        wandb.run.summary["best_main_slot_f1"] = main_slot_f1
                        wandb.run.summary[
                            "best_second_slot_f1"
                        ] = second_slot_f1

                elif wait >= config["patience"]:
                    logger.info("Out of patience. Early stop now.")
                    # maybe save sth
                    break
                else:
                    wait += 1
                    logger.debug(
                        f"Not improved on epoch {epoch_i}, wait={wait}",
                    )
        # End training
        config["checkpoint"] = best_path

    def overfit_test(self, train_dl):
        batch_dict = next(iter(train_dl))
        i = 0
        x = []
        y = []
        while True:
            i += 1
            output = self.training_step(batch_dict, i)
            x.append(i)
            y.append(output["loss"])
            print(
                f"{i}: loss: {output['loss']:.3f}, "
                f"intent_f1: {output['intent_f1']:.3f}, "
                f"slot_f1: {output['slot_f1']:.3f}"
            )
            print(self.validating_step(batch_dict, i))
            if i % 50 == 0:
                pass
                # plt.plot(x, y)
                # plt.xlabel("train iter")
                # plt.ylabel("loss")
                # plt.pause(0.05)


def train():
    tokenizer = get_tokenizer(config)
    # create dataset and dataloader object
    train_ds = get_dataset(config=config, mode="train", tokenizer=tokenizer)
    val_ds = get_dataset(config=config, mode="val", tokenizer=tokenizer)
    slots = [
        slot
        for i, slot in enumerate(train_ds.slots)
        if i not in train_ds.main2second
    ]
    train_ds.slots = slots
    val_ds.slots = slots
    config["n_main_slot"] = len(slots) - len(train_ds.second_slots)
    config["intents"] = train_ds.intents
    config["n_intent"] = train_ds.n_intent
    config["n_slot"] = len(slots)
    model = get_model(config)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=train_ds.collate_fn,
        shuffle=True,
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        collate_fn=val_ds.collate_fn,
        shuffle=False,
    )
    optimizer, scheduler = get_optim(
        config=config,
        model=model,
        num_training_steps=len(train_dl) * config["n_epochs"],
    )
    if not config["lr_decay"]:
        scheduler = None

    trainer = Trainer(
        model=model,
        slots=slots,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if config["use_wandb"]:
        if not config["project_name"]:
            raise ValueError("Must specify `project_name` if using wandb.")
        wandb.init(
            project=config["project_name"],
            name=config["save_prefix"],
            config=config,
        )
    trainer.fit(train_dl=train_dl, val_dl=val_dl)


def validate():
    global config
    torch_saved = torch.load(os.path.join(config["checkpoint"], "ckpt.pth"))
    checkpoint = config["checkpoint"]
    config = json.load(open(os.path.join(checkpoint, "config.json"), "r"))
    config["checkpoint"] = checkpoint
    tokenizer = get_tokenizer(config)
    model = get_model(config)
    model.load_state_dict(torch_saved["model"])

    val_ds = get_dataset(config=config, mode="val", tokenizer=tokenizer)
    slots = [
        slot
        for i, slot in enumerate(val_ds.slots)
        if i not in val_ds.main2second
    ]
    val_ds.slots = slots
    config["intents"] = val_ds.intents
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=1,
        num_workers=4,
        collate_fn=val_ds.collate_fn,
        shuffle=False,
    )
    trainer = Trainer(model=model, slots=slots)
    outputs = trainer.validate(val_dl=val_dl)
    print_outputs = [
        f"intent_f1: {o['intent_f1']:.3f}\n"
        f"slot_f1: {o['slot_f1']:.3f}\n"
        f"raw_context: {o['tokenized_context'][0]}\n"
        f"intent_pred: {o['intent_pred'][0]}\n"
        f"intent_label: {o['intent_label'][0]}\n"
        f"slot_pred_str: {o['slot_pred_str'][0]}\n"
        f"slot_label_str: {o['slot_label_str'][0]}\n"
        f"main_slot_pred_str: {o['main_slot_pred_str'][0]}\n"
        f"main_slot_label_str: {o['main_slot_label_str'][0]}\n"
        f"second_main_slot_pred_str: {o['second_slot_pred_str'][0]}\n"
        f"second_main_slot_label_str: {o['second_slot_label_str'][0]}\n"
        for o in outputs
    ]
    output_path = os.path.join(checkpoint, "validate_outputs.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(print_outputs))
    logger.info(f"Wrote validate outputs to {output_path}")


def test():
    global config
    checkpoint = config["checkpoint"]
    torch_saved = torch.load(os.path.join(config["checkpoint"], "ckpt.pth"))
    config = json.load(
        open(os.path.join(config["checkpoint"], "config.json"), "r")
    )
    config["checkpoint"] = checkpoint
    tokenizer = get_tokenizer(config)
    model = get_model(config)
    model.load_state_dict(torch_saved["model"])

    test_ds = get_dataset(config=config, mode="test", tokenizer=tokenizer)
    slots = [
        slot
        for i, slot in enumerate(test_ds.slots)
        if i not in test_ds.main2second
    ]
    test_ds.slots = slots
    config["intents"] = test_ds.intents
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=10000,
        num_workers=4,
        collate_fn=test_ds.collate_fn,
        shuffle=False,
    )
    trainer = Trainer(model=model, slots=slots)
    outputs = trainer.test(test_dl=test_dl)
    print_outputs = [
        f"intent_f1: {o['intent_f1']:.3f}\n"
        f"slot_f1: {o['slot_f1']:.3f}\n"
        f"raw_context: {o['tokenized_context'][0]}\n"
        f"intent_pred: {o['intent_pred'][0]}\n"
        f"intent_label: {o['intent_label'][0]}\n"
        f"slot_pred_str: {o['slot_pred_str'][0]}\n"
        f"slot_label_str: {o['slot_label_str'][0]}\n"
        f"main_slot_pred_str: {o['main_slot_pred_str'][0]}\n"
        f"main_slot_label_str: {o['main_slot_label_str'][0]}\n"
        f"second_main_slot_pred_str: {o['second_slot_pred_str'][0]}\n"
        f"second_main_slot_label_str: {o['second_slot_label_str'][0]}\n"
        for o in outputs
    ]
    output_path = os.path.join(checkpoint, "test_outputs.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(print_outputs))
    logger.info(f"Wrote test outputs to {output_path}")


if __name__ == "__main__":
    config = vars(get_args())
    pprint(config)
    set_seed(config["seed"])
    if importlib.util.find_spec("wandb") is None:
        config["use_wandb"] = False
    logger = getlogger(
        logger_level="debug",
        console_level="info",
    )
    if config["do_train"]:
        train()
    if config["do_val"]:
        validate()
    if config["do_test"]:
        test()
