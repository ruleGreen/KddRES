import os
import random
import torch
import numpy as np
import sys
import logging
import logging.config
import datetime
from typing import List, Tuple
from torch import nn


def hyperparam_string(options):
    """Hyerparam string."""
    task_path = "model_%s" % (options.task)
    dataset_path = "data_%s" % (options.dataset)

    exp_name = ""
    if options.__contains__("bidirectional"):
        exp_name += "bidir_%s__" % (options.bidirectional)
    if options.__contains__("emb_size"):
        exp_name += "emb_dim_%s__" % (options.emb_size)
    if options.__contains__("hidden_size"):
        exp_name += "hid_dim_%s_x_%s__" % (
            options.hidden_size,
            options.num_layers,
        )
    exp_name += "bs_%s__" % (options.batchSize)
    exp_name += "dropout_%s__" % (options.dropout)
    if options.__contains__("optim"):
        exp_name += "optimizer_%s__" % (options.optim)
    exp_name += "lr_%s__" % (options.lr)
    if options.__contains__("max_norm"):
        exp_name += "mn_%s__" % (options.max_norm)
    exp_name += "me_%s" % (options.max_epoch)

    return os.path.join(task_path, dataset_path, exp_name)


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_random_state():
    return (
        torch.random.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )


def set_random_state(states):
    torch.random.set_rng_state(states[0])
    np.random.set_state(states[1])
    random.setstate(states[2])


def format_size(size: int):
    suffix = "B"
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(size) < 1024.0:
            return "%3.1f %s%s" % (size, unit, suffix)
        size /= 1024.0
    return "%.1f %s%s" % (size, "Yi", suffix)


def print_tensor_size(nlargest: int = 10):
    sizes = []
    for name, value in globals().items():
        if isinstance(value, torch.Tensor):
            sizes.append((name, "tensor", sys.getsizeof(value.storage())))
        elif isinstance(value, nn.Module):
            size = sum(
                [
                    sys.getsizeof(param.storage())
                    for param in list(value.parameters())
                ]
            )
            sizes.append((name, "module", size))
    for name, type, size in sorted(
        sizes,
        key=lambda x: x[1],
        reverse=True,
    )[:nlargest]:
        print(
            "{:>30}{:>10}: {:>8}".format(
                name, "(" + type + ")", format_size(size)
            )
        )


# def lineplot(
#     data: pd.DataFrame,
#     x: str,
#     y: str,
#     hue: str = None,
#     hue_order: List[str] = None,
#     title: str = None,
#     xlabel: str = None,
#     ylabel: str = None,
#     legend_title: str = None,
#     out_filename: str = None,
#     show: bool = True,
# ):
#     ax = sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)
#     if title is not None:
#         ax.set_title(title)
#     if xlabel is not None:
#         ax.set_xlabel(xlabel)
#     if ylabel is not None:
#         ax.set_ylabel(ylabel)
#     if legend_title is not None:
#         ax.get_legend().set_title(legend_title)
#     if out_filename:
#         plt.savefig(out_filename)
#     if show:
#         plt.show()
#     return ax


def getlogger(
    name: str = None,
    logger_level: str = None,
    console_level: str = None,
    file_level: str = None,
    log_path: str = None,
):
    """Configure the logger and return it.

    Args:
        name (str, optional): Name of the logger, usually __name__.
            Defaults to None. None refers to root logger, usually useful
            when setting the default config at the top level script.
        logger_level (str, optional): level of logger. Defaults to None.
            None is treated as `debug`.
        console_level (str, optional): level of console. Defaults to None.
            None is treated as `debug`.
        file_level (str, optional): level of file. Defaults to None.
            None is treated `debug`.
        log_path (str, optional): The path of the log.

    Note that console_level should only be used when configuring the
    root logger.
    """

    logger = logging.getLogger(name)
    if name:
        logger.setLevel(level_map[logger_level or "debug"])
    else:  # root logger default lv should be high to avoid external lib log
        logger.setLevel(level_map[logger_level or "warning"])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up the logfile handler
    if file_level and log_path:
        logTime = datetime.datetime.now()
        fn1, fn2 = os.path.splitext(log_path)
        log_filename = f"{fn1}-{logTime.strftime('%Y%m%d-%H%M%S')}{fn2}"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        fh = logging.FileHandler(log_filename)
        fh.setLevel(level_map[file_level or "debug"])
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # set up the console/stream handler
    if name and console_level:
        raise ValueError(
            "`console_level` should only be set when configuring root logger."
        )
    if console_level:
        sh = logging.StreamHandler()
        sh.setLevel(level_map[console_level or "debug"])
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def get_gradients(model) -> torch.tensor:
    """

    Returns:
        torch.tensor: The flatten gradients.

    """
    return torch.cat(
        [p.grad.reshape(-1) for p in model.parameters() if p is not None],
        dim=0,
    )


def get_params(model) -> torch.tensor:
    """

    Returns:
        torch.tensor: The flatten parameters.

    """
    return torch.cat(
        [p.reshape(-1) for p in model.parameters() if p is not None],
        dim=0,
    )


def seq2tuples(labels: List[str]) -> List[Tuple[str, int, int]]:
    """
    Args:
        labels: List of tags.

    Returns:
        List of Tuple of (start_idx, end_idx, tag)

    """
    chunks = []
    start_idx, end_idx = 0, 0
    for idx in range(1, len(labels) - 1):
        chunkStart, chunkEnd = False, False
        if labels[idx - 1] not in (
            "O",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<STOP>",
            "<START>",
        ):
            prevTag, prevType = labels[idx - 1][:1], labels[idx - 1][2:]
        else:
            prevTag, prevType = "O", "O"
        if labels[idx] not in (
            "O",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<STOP>",
            "<START>",
        ):
            Tag, Type = labels[idx][:1], labels[idx][2:]
        else:
            Tag, Type = "O", "O"
        if labels[idx + 1] not in (
            "O",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<STOP>",
            "<START>",
        ):
            nextTag, nextType = labels[idx + 1][:1], labels[idx + 1][2:]
        else:
            nextTag, nextType = "O", "O"

        if (
            Tag == "B"
            or Tag == "S"
            or (prevTag, Tag)
            in {
                ("O", "I"),
                ("O", "E"),
                ("E", "I"),
                ("E", "E"),
                ("S", "I"),
                ("S", "E"),
            }
        ):
            chunkStart = True
        if Tag != "O" and prevType != Type:
            chunkStart = True

        if (
            Tag == "E"
            or Tag == "S"
            or (Tag, nextTag)
            in {
                ("B", "B"),
                ("B", "O"),
                ("B", "S"),
                ("I", "B"),
                ("I", "O"),
                ("I", "S"),
            }
        ):
            chunkEnd = True
        if Tag != "O" and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx, end_idx, Type))
            start_idx, end_idx = 0, 0
    return chunks
