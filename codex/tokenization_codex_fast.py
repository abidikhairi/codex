# coding=utf-8
"""Tokenization classes for Codex."""

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging

from codex.tokenization_codex import CodexTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

MAX_MODEL_INPUT_SIZES = {"khairi/codex-tokenizer": 2048}


class CodexTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Codex tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.


    ```python
    >>> from codex.tokenization_codex_fast import CodexTokenizerFast

    >>> tokenizer = CodexTokenizerFast.from_pretrained("khairi/Codex-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    ```
    
    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Not applicable to this tokenizer.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = CodexTokenizer

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
