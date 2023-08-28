#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Init a logger with options from env variables.

- set log level by ``LOG_LEVEL``, default: ``INFO``;
- output log message to file by ``LOG_FILE``, default: output to stdout.

TODO:
    support setting log level and log file from config file.
"""
import logging
import os

_LOG_FMT = "[%(asctime)s][%(levelname).1s][%(process)d-%(name)s-%(filename)s:%(lineno)s]- %(message)s"
_DATE_FMT = "%Y-%m-%d,%H:%M:%S"

_logging_level = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    # Distributed Level, print log in main proc only by default, set this level to print all messages.
    "DP": logging.INFO,
    "DEBUG": logging.DEBUG,
    None: logging.INFO,
}

_level = os.environ.get("LOG_LEVEL", "INFO").upper()


class ShortNameFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        raw = record.name  # save and restore for other formatters if desired
        parts = raw.split(".")
        record.name = ".".join(p[:3] for p in parts) if len(parts) > 1 else raw  # keep first char for module name.
        result = super().format(record)
        record.name = raw
        return result


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None, style="default"):
        super().__init__(logger, extra or {})
        self._style = style
        self._enable = self._enable()

    @classmethod
    def _enable(cls):
        # Note: to make this Logger more standalone, perform basic check without extra deps, e.g. tf/torch et al.
        worker = os.getenv("WORKER")
        rank = os.getenv("RANK")
        # not in DP/DDP mode or proc_id = "0"
        is_main = (not worker and not rank) or (worker == "0" or rank == "0")
        is_jeeves_job = os.getenv("JEEVES_JOB_ID")
        return _level in ["DEBUG", "DP"] or is_jeeves_job or is_main

    def _format(self, *msgs, color: str = None):
        if self._style == "legacy":
            if len(msgs) == 1:
                msg_str = msgs[0]
            else:
                msg_str = msgs[0] % msgs[1:]
        else:
            msg_str = ", ".join([str(msg) for msg in msgs])

        if color:
            pass
        return msg_str

    def log(self, level, msg, *args, **kwargs):
        color = kwargs.pop("color", None)
        if self.isEnabledFor(level) and self._enable:
            msg, kwargs = self.process(msg, kwargs)
            msg_str = self._format(msg, *args, color=color)
            # noinspection PyProtectedMember
            self.logger._log(level, msg_str, (), **kwargs)


def init_logger(name="ai", filename=os.environ.get("LOG_FILE", ""), fmt=_LOG_FMT, level=_level, style="legacy"):
    """init logger

    Args:
        name(str): optional, default: ai.
        filename(str): optional, default: "". Output log to file if specified, by default is set by env `LOG_FILE`.
        fmt(str): optional, default: _LOG_FMT
        level(str): optional, default: INFO
        style(str): optional, choice from ["print", "legacy"]
            - legacy: take first argument as a formatter, the remaining positional arguments as message values.
                this is consistent with the constraint of `logging` pkg
            - print: all positional arguments are message values which will be concatenated with ", "

    Returns:
        a logger instance

    Examples:
    >>> log = init_logger("log2stdout", level="INFO")
    >>> log.error("info")
    """
    logger = logging.getLogger(name)
    logger.setLevel(_logging_level[level])
    if fmt:
        # formatter = logging.Formatter(fmt, datefmt=_DATE_FMT)
        formatter = ShortNameFormatter(fmt, datefmt=_DATE_FMT)
    else:
        formatter = None

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logging.basicConfig(format=fmt, level=_logging_level[_level], handlers=[handler])

    if filename:
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return StyleAdapter(logger, style=style)