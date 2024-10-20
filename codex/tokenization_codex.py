# coding=utf-8
"""Tokenization classes for Codex."""

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


MAX_MODEL_INPUT_SIZES = {"khairi/codex-tokenizer": 2048}


class CodexTokenizer(PreTrainedTokenizer):
    """
    Construct a Codex tokenizer. Based on byte-level Byte-Pair-Encoding.
    
    ```python
    >>> from codex.tokenization_codex import CodexTokenizer

    >>> tokenizer = CodexTokenizer.from_pretrained("khairi/Codex-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    ```

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

