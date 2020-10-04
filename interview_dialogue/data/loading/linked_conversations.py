import os
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import chain, groupby

import numpy as np
import torch
from interview_dialogue.data import (END_GPT2, PAD_GPT2, SPEAKER_GPT2,
                                     START_GPT2)
from interview_dialogue.data.tokenizers import TransformerTokenizer
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["[SEP]", "[host]", "[guest]", "[grounding]", "[target]"]

DISCOURSE_PATTERN_ENCODER = {
    'HQ': 0,
    'HA': 1,
    'GQ': 2,
    'GA': 3,
}
DISCOURSE_PATTERN_DECODER = {
    v: k
    for k, v in DISCOURSE_PATTERN_ENCODER.items()
}


class LinkedConversationDataset(Dataset):
    """
    Dataset supporting sentence-level splitting in a conversation

    Use case: Apply dialogue history
    """
    def __init__(
        self,
        conversations,
        linked_documents,
        tokenizer,
        host_only: int = False,
        question_only: int = False,
        min_history: int = 3,
        min_following: int = 2,
        history_turns: int = 5,
        n_documents: int = 5,
        debug_mode: bool = False,
        cache_path: str = None,
        assignments: dict = None,
        **tokenizer_kwargs,
    ):
        super().__init__()

        self.start = datetime.now()

        # Params
        self.min_history = min_history
        self.min_following = min_following
        self.history_turns = history_turns
        self.n_documents = n_documents
        self.debug_mode = debug_mode
        self.host_only = host_only
        self.question_only = question_only
        self.document_assignments = assignments or dict()

        # Load converations file
        self.conversations = conversations
        if isinstance(conversations, str):
            with open(conversations, 'rb') as rf:
                self.conversations = pickle.load(rf)

        # Load document headlines & paragraphs
        self.documents = linked_documents
        if isinstance(linked_documents, str):
            with open(linked_documents, 'rb') as rf:
                self.documents = pickle.load(rf)

        # Tokenizer
        if tokenizer.lower() == 'gpt2':
            self.tokenizer = TransformerTokenizer(
                cache_path=cache_path,
                pretrained_model='gpt2',
                special_tokens=SPECIAL_TOKENS,
                **tokenizer_kwargs)
        else:
            raise ValueError(
                '"tokenizer" argument {} must be one of the following:\n{}'.
                format(tokenizer, ['gpt2']))

        # Create index: Conversation ID, Turns #
        self.index = []
        for c_id, c_dict in enumerate(self.conversations):
            turn_patterns = [
                '{}{}'.format(
                    'H' if spk == 0 else 'G',
                    'Q' if '?' in utt else 'A',
                ) for spk, utt in c_dict['turns']
            ]
            for i, (spk, utt) in enumerate(c_dict['turns']):
                # Minimum # context/history turns
                if i <= self.min_history:
                    continue

                # Minimum # following turns
                if len(c_dict['turns']) - i <= self.min_following:
                    continue

                # Host-only
                if self.host_only and spk != 0:
                    continue

                # Question-only
                if self.question_only and '?' not in utt:
                    continue

                # Compressed pattern - make sure there are 3 turns remaining
                compr_discourse_pattern = [
                    i[0] for i in groupby(turn_patterns[i + 1:])
                ][:3]
                if len(compr_discourse_pattern) < 3:
                    continue

                # Add index entry
                self.index.append((c_id, i))

        # Shuffle
        np.random.shuffle(self.index)
        self.log('Created index with {:,} elements'.format(len(self.index)))

        self.log(
            'Created {} {} with {} min history, {} min following, {} history turns'
            .format('host-only' if self.host_only else 'all-speaker',
                    self.__class__.__name__, self.min_history,
                    self.min_following, self.history_turns))
        if self.debug_mode:
            self.log('DEBUG MODE')

    def log(self, m):
        print('{} [{}] {}'.format(datetime.now() - self.start,
                                  self.__class__.__name__, m))

    def debug(self, m):
        if self.debug_mode:
            self.log(m)

    def __len__(self):
        return len(self.index)

    def _sample(self, n=5):
        indices = np.random.choice(len(self), size=n, replace=False)
        self.debug('Indices: {}'.format(indices))
        return [self[i] for i in indices]

    def __getitem__(self, index):
        # get item with no external assignment
        return self.get_item_w_assignments(index, assignments=None)

    def get_item_w_assignments(self,
                               index,
                               conv_id=None,
                               turn_ix=None,
                               assignments=None):
        # Format:
        # <speaker/pad> <sent_1> <space> <sent_2> <space> <eos> ...
        # <gold speaker/pad> <gold sent_1> <space> ... <eos>

        # Conversation
        if conv_id is None or turn_ix is None:
            conv_id, turn_ix, = self.index[index]
        conv = self.conversations[conv_id]

        # Get turns
        target_turn = conv['turns'][turn_ix]
        target_speaker, target_utterance = target_turn
        target_utterance = self.tokenizer.encode(target_utterance,
                                                 bos_token=True,
                                                 eos_token=True)

        # Get compressed pattern
        following_discourse = [
            '{}{}'.format('H' if spk == 0 else 'G', 'Q' if '?' in utt else 'A')
            for spk, utt in conv['turns'][turn_ix + 1:]
        ]
        compressed_discourse_pattern = [0] + [
            DISCOURSE_PATTERN_ENCODER[i[0]]
            for i in groupby(following_discourse)
        ][:3]

        # Get historical turns
        history = conv['turns'][(
            max(turn_ix -
                self.history_turns, 0) if self.history_turns else 0):turn_ix]
        history_speakers, history_utterances = zip(*history)

        # Encode history utterances
        history_utterances = [
            self.tokenizer.encode(h, bos_token=False, eos_token=False)
            for h in history_utterances
        ]

        # Ordered list of documents
        document_ids = []
        if self.n_documents:
            if assignments is not None:
                candidates = assignments
            elif self.document_assignments:
                candidates = self.document_assignments.get(conv_id)
            else:
                candidates = conv['documents']
            if candidates:
                _, document_ids = zip(*candidates[:self.n_documents])
            else:
                document_ids = []
        document_headlines = [
            self.tokenizer.encode(self.documents[i]['headline'],
                                  bos_token=False,
                                  eos_token=False) for i in document_ids
        ]

        if self.debug_mode:
            self.log('===========================')

            # Target
            self.log('Target utterance ({}):\n\t{}'.format(
                'Host' if target_speaker == 0 else 'Guest',
                self.tokenizer.decode(target_utterance)))

            # History
            self.log('History:\n\t{}'.format('\n\t'.join([
                '{}:\t{}'.format('Host' if sp == 0 else 'Guest',
                                 self.tokenizer.decode(utt))
                for sp, utt in zip(history_speakers, history_utterances)
            ])))

            # Source documents
            self.log('Source document headlines:\n\t{}'.format('\n\t'.join(
                [self.tokenizer.decode(d) for d in document_headlines])))

            # Following pattern
            self.log('Following discourse pattern: "{}"'.format('-'.join([
                DISCOURSE_PATTERN_DECODER[i]
                for i in compressed_discourse_pattern
            ])))

            self.log('===========================\n\n')

        # Returns:
        #   Target speaker ID (int)
        #   Target utterance tokens (list of ints)
        #   Historical speaker IDs (list of ints)
        #   Historical utterance tokens (list of lists)
        #   Source document headlines (list of lists)
        #   Discourse pattern IDs (list of ints)
        return target_speaker, target_utterance, \
            history_speakers, history_utterances, \
            document_headlines, compressed_discourse_pattern


class LinkedConversationDecoderCollator(object):
    def __init__(self,
                 bos_token: int,
                 eos_token: int,
                 pad_token: int,
                 sep_token: int,
                 host_seg_id: int,
                 guest_seg_id: int,
                 grounding_seg_id: int,
                 target_seg_id: int,
                 max_len: int = 1024,
                 **kwargs):
        super().__init__()

        # Limitations
        self.max_len = max_len

        # Special tokens
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token
        self.sep = sep_token

        # Grounding segments
        self.host = host_seg_id
        self.guest = guest_seg_id
        self.grounding = grounding_seg_id
        self.target = target_seg_id

    def __call__(self, batch):
        """
        Specification

        # PART              # SEGMENT               # Target Mask   # Padding Mask
        [BOS]               Grounding               False           True
        Source 1            Grounding               False           True
        Source 2            Grounding               False           True
        [SEP]               Grounding               False           True
        History 1           History speaker (H/G)   False           True
        History 2           History speaker (H/G)   False           True
        [SEP]               History speaker (H/G)   False           True
        Target utterance    Target                  True            True
        [EOS]               Target                  True            True
        PADDING             Target                  False           False

        + Discourse pattern prediction
        Pattern tokens B x 4
        """
        batch_size = len(batch)

        # Separate blocks
        target_speakers, target_utterances, \
            history_speakers, history_utterances, \
            document_headlines, compressed_discourse_pattern = zip(*batch)

        # Initialize
        tokens = []
        segment_ids = []
        target_masks = []
        padding_masks = []

        for i in range(batch_size):
            # BOS
            obs_tok = [self.bos]
            obs_seg = [self.grounding]

            # [SEP] [target] [EOS]
            # Strip the BOS and EOS existing tokens from targets
            target_tok = [self.sep] + target_utterances[i][1:-1] + [self.eos]
            # Do not calculate loss for the [SEP]
            target_loss_mask = [False] + [True] * (len(target_tok) - 1)
            # Remaining length - minus the source/history [SEP] and the [BOS] token
            remaining_len = self.max_len - 2 - len(target_tok)

            # Sources - add until it fulfills the remaining length
            for source in document_headlines[i]:
                if len(source) > remaining_len:
                    break

                # Add source & associated segments
                obs_tok.extend(source)
                obs_seg.extend([self.grounding] * len(source))
                remaining_len -= len(source)

            # Pack history tokens
            history_reversed = []
            history_seg_reversed = []
            # Start from most recent
            for u, spk in zip(history_utterances[i][::-1],
                              history_speakers[i][::-1]):
                seg_id = self.host if spk == 0 else self.guest
                if len(u) > remaining_len:
                    # Truncate
                    u_trunc = u[:remaining_len]
                    u_trunc_seg = [seg_id] * len(u_trunc)

                    # History
                    history_reversed.append(u_trunc)
                    history_seg_reversed.append(u_trunc_seg)

                    # Track
                    remaining_len -= len(u_trunc)
                    break

                history_reversed.append(u)
                history_seg_reversed.append([seg_id] * len(u))
                remaining_len -= len(u)

            # Put back in the correct order
            history_tokens = list(chain.from_iterable(history_reversed[::-1]))
            history_segments = list(
                chain.from_iterable(history_seg_reversed[::-1]))
            if history_tokens:
                obs_tok.extend([self.sep] + history_tokens)
                obs_seg.extend([self.grounding] + history_segments)

            # Compose
            obs_tgt = [False] * len(obs_tok) + target_loss_mask
            obs_tok += target_tok
            # [SEP] token should have the same segment embedding as the last history utt
            obs_seg += [obs_seg[-1]] + [self.target] * (len(target_tok) - 1)

            assert len(obs_tok) <= self.max_len
            assert len(obs_seg) <= self.max_len
            assert len(obs_tgt) <= self.max_len

            # Append to batch
            tokens.append(obs_tok)
            segment_ids.append(obs_seg)
            target_masks.append(obs_tgt)

        # Padding
        max_T = max(len(t) for t in tokens)

        # Input padding
        padded_tokens = torch.LongTensor(
            [t + [self.pad] * (max_T - len(t)) for t in tokens])
        padded_segment_ids = torch.LongTensor(
            [t + [0] * (max_T - len(t)) for t in segment_ids])
        padded_target_masks = torch.BoolTensor(
            [t + [False] * (max_T - len(t)) for t in target_masks])
        padding_masks = torch.BoolTensor(
            [[True] * len(t) + [False] * (max_T - len(t)) for t in tokens])

        # Sequence
        discourse_patterns = torch.LongTensor(compressed_discourse_pattern)
        """
        1. Padded tokens
        2. Padded segment IDs
        3. Padded target (loss) masks
        4. Padding masks
        5. Discourse structure (3 following compressed turn patterns)
        """
        return padded_tokens, padded_segment_ids, padded_target_masks, padding_masks, \
            discourse_patterns


class LinkedConversationCollator(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.max_len = kwargs.get('max_len', 512) or 512
        self.split_source_hist = kwargs.get('encoder_split', False) or False
        self.bos_token = kwargs['bos_token']
        self.eos_token = kwargs['eos_token']

    def __call__(self, batch):
        """
        Encoder input spec:

        Source 1        Segment ID for source (0)
        Source 2        Segment ID for source (0)
        History 1       Segment ID for turn speaker (1/2)
        History 2       Segment ID for turn speaker (1/2)

        Decoder input spec:

        Target utterance    Segment ID for target speaker (1/2)
        """
        batch_size = len(batch)

        # Separate blocks
        target_speakers, target_utterances, \
            history_speakers, history_utterances, \
            document_headlines, compressed_discourse_pattern = zip(*batch)

        # Initialize
        input_tokens = []
        input_segments = []
        target_tokens = []
        target_segments = []

        # Iterate through batch
        for i in range(batch_size):
            # Targets
            target_tokens.append(target_utterances[i])
            target_segments.append([target_speakers[i] + 1] *
                                   len(target_utterances[i]))

            # Encoder inputs
            input_tok = [self.bos_token]
            input_seg = [0]
            remaining_len = self.max_len - 2 - self.split_source_hist

            # Sources
            for source in document_headlines[i]:
                if len(source) > remaining_len:
                    break

                # Add source & associated segments
                input_tok.extend(source)
                input_seg.extend([0] * len(source))
                remaining_len -= len(source)

            # If splitting sources and history & both segments exist
            n_sources = len(document_headlines)
            n_history = len(history_utterances[i])
            if self.split_source_hist and n_sources > 0 and n_history > 0:
                input_tok.append(self.eos_token)
                input_seg.append(0)

            # Find how many remaining turns of history (starting w/ most recent) can fit in model
            n_hist_allowed = 0
            hist_tokens_total = 0
            for u in history_utterances[i][::-1]:
                hist_tokens_total += len(u)
                if hist_tokens_total > remaining_len:
                    break
                n_hist_allowed += 1

            # Historical utterances
            for utt, sp in zip(history_utterances[i][-n_hist_allowed:],
                               history_speakers[i][-n_hist_allowed:]):
                # Add utterance & associated segments
                input_tok.extend(utt)
                input_seg.extend([sp + 1] * len(utt))

            # Append EOS
            input_tok.append(self.eos_token)
            input_seg.append(0)

            # Append row of inputs
            input_tokens.append(input_tok)
            input_segments.append(input_seg)

        max_T_input = max(len(i) for i in input_tokens)
        max_T_target = max(len(t) for t in target_tokens)

        # Input padding
        padded_inputs = torch.LongTensor(
            [t + [PAD_GPT2] * (max_T_input - len(t)) for t in input_tokens])
        padded_input_segments = torch.LongTensor(
            [t + [0] * (max_T_input - len(t)) for t in input_segments])
        input_padding_masks = torch.BoolTensor([[True] * len(t) + [False] *
                                                (max_T_input - len(t))
                                                for t in input_tokens])

        # Target padding
        padded_targets = torch.LongTensor(
            [t + [PAD_GPT2] * (max_T_target - len(t)) for t in target_tokens])
        padded_target_segments = torch.LongTensor(
            [t + [0] * (max_T_target - len(t)) for t in target_segments])
        target_padding_masks = torch.BoolTensor([[True] * len(t) + [False] *
                                                 (max_T_target - len(t))
                                                 for t in target_tokens])

        return padded_inputs, padded_input_segments, input_padding_masks, \
            padded_targets, padded_target_segments, target_padding_masks
