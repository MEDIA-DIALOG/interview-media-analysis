import argparse
import os
from typing import Sequence, Union

import torch
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizerFast, GPT2Tokenizer

SEG_SEP = '[SEP]'


class TransformerTokenizer(object):
    def __init__(self,
                 cache_path: str = None,
                 special_tokens: list = [SEG_SEP],
                 **kwargs):
        # Initialize tokenizer
        super().__init__()
        self.cache_path = cache_path

        # Special kwargs
        self.pretrained_model = kwargs.get('pretrained_model',
                                           'bert-base-uncased')
        if 'bert' in self.pretrained_model:
            self.model_class = BertTokenizerFast
        elif 'gpt' in self.pretrained_model:
            self.model_class = GPT2Tokenizer
        else:
            raise ValueError('Unsupported tokenizer {}'.format(
                self.pretrained_model))
        self.encode_kwargs = {
            'add_prefix_space': kwargs.get('add_prefix_space', False)
        }

        # Add additional tokens to tokenizer
        self.special_tokens = special_tokens or []

        # Load from cache
        self.tokenizer = self._make_tokenizer()

        # Tokens
        self._bos_token_id = self.tokenizer.bos_token_id  # BOS
        self._eos_token_id = self.tokenizer.eos_token_id  # EOS
        self._pad_token_id = self.tokenizer.pad_token_id  # PAD
        self._cls_token_id = self.tokenizer.cls_token_id  # CLS
        self._sep_token_id = self.tokenizer.sep_token_id  # SEP

    def __len__(self):
        if not hasattr(self, 'tokenizer'):
            self._make_tokenizer()
        return len(self.tokenizer)

    def __getstate__(self):
        """ For pickle serialization """
        state = self.__dict__.copy()
        try:
            del state['tokenizer']
        except:
            pass
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.tokenizer = self._make_tokenizer()

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def bos_token_id(self):
        return self._bos_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    @property
    def cls_token_id(self):
        return self._cls_token_id

    @property
    def sep_token_id(self):
        return self._sep_token_id

    def encode(self,
               sentence: str,
               bos_token: bool = True,
               eos_token: bool = True,
               max_len: int = None,
               **kwargs) -> Sequence[int]:
        """
        Encodes a sentence to token IDs

        Args:
            sentence (str): Sentence in text form
            bos_token (bool, optional): Whether to add a BOS token. Defaults to True.
            eos_token (bool, optional): Whether to add an EOS token. Defaults to True.
            max_len (int, optional): Truncate the encoding to a maximum length. Defaults to None.

        Returns:
            Sequence[int]: List of token IDs (integers)
        """
        if not hasattr(self, 'tokenizer'):
            self._make_tokenizer()

        sentence_tokens = self._encode(sentence, **kwargs)
        if max_len:
            sentence_tokens = sentence_tokens[:(max_len - bos_token -
                                                eos_token)]
        if bos_token:
            sentence_tokens = [self.bos_token_id] + sentence_tokens
        if eos_token:
            sentence_tokens += [self.eos_token_id]

        return sentence_tokens

    def decode(self, tokens: Union[list, torch.LongTensor], **kwargs) -> str:
        """
        Decodes a sequence of IDs into a string

        Args:
            tokens (Union[list, torch.LongTensor]): Tokens sequence

        Returns:
            str: Output text
        """
        if len(tokens) == 0:
            return ''

        if not hasattr(self, 'tokenizer'):
            self._make_tokenizer()

        if isinstance(tokens, torch.Tensor):
            return self.decode_list(tokens.cpu().tolist(), **kwargs)

        return self.decode_list(tokens, **kwargs)

    def _make_tokenizer(self):
        # Tokenizer base
        self.tokenizer = self.model_class.from_pretrained(
            self.pretrained_model,
            cache_dir=os.path.join(self.cache_path, self.pretrained_model) \
                if self.cache_path else None
        )

        # Special tokens
        if 'bert' not in self.pretrained_model:
            added_tokens_dict = {
                'additional_special_tokens': self.special_tokens
            }
            if self.tokenizer.pad_token_id is None:
                added_tokens_dict['pad_token'] = '~PAD~'
            self.tokenizer.add_special_tokens(added_tokens_dict)

        return self.tokenizer

    def __getitem__(self, index):
        if not hasattr(self, 'tokenizer'):
            self._make_tokenizer()
        if isinstance(index, str):
            return self.tokenizer.encoder.get(
                index, self.tokenizer.added_tokens_encoder.get(index, None))
        if isinstance(index, int):
            if index < len(self):
                return self.decode([index])

    def _encode(self, sentence: str, **kwargs) -> Sequence[int]:
        """
        Encodes a sentence to token IDs

        Args:
            sentence (str): Sentence in text form

        Returns:
            Sequence[int]: List of token IDs (integers)
        """
        return self.tokenizer.encode(sentence, **kwargs, **self.encode_kwargs)

    def decode_list(self, tokens: list, **kwargs) -> str:
        """
        Decode a list of IDs

        Args:
            tokens (list): List of token IDs

        Returns:
            str: Output strings
        """
        output_str = self.tokenizer.decode(tokens, **kwargs)
        return output_str
