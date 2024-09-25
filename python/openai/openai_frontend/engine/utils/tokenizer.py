from typing import Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def get_cached_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer with cached properties.

    This will patch the tokenizer object in place.

    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown. This
    function caches these properties for faster access."""

    tokenizer_all_special_ids = set(tokenizer.all_special_ids)
    tokenizer_all_special_tokens_extended = tokenizer.all_special_tokens_extended
    tokenizer_all_special_tokens = set(tokenizer.all_special_tokens)
    tokenizer_len = len(tokenizer)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self):
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self):
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self):
            return tokenizer_all_special_tokens_extended

        def __len__(self):
            return tokenizer_len

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    tokenizer.__class__ = CachedTokenizer
    return tokenizer


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface/modelscope."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
    except ValueError as e:
        raise e
    except AttributeError as e:
        raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        print(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return get_cached_tokenizer(tokenizer)
