import os
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import torch
from interview_dialogue.data import (END_GPT2, PAD_GPT2, SPEAKER_GPT2,
                                     START_GPT2)
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class SpeakerUtteranceDataset(Dataset):
    """
    Dataset supporting sentence-level splitting in a conversation

    Use case: Apply dialogue history
    """
    def __init__(
        self,
        gpt_utterances_map,
        gpt_sentence_dicts,
        host_only=False,
        debug_mode=False,
        min_history=3,
        min_following=2,
    ):
        """
        Arguments:
            gpt_utterances_map (dict): Map of:
                (episode, episode_order) : 
                    'sentences':    list of component sentence indices
                    'speaker_id':   Speaker ID (order) in the conversation
            gpt_sentence_dicts (dict): Map of sentence index :
                'episode':          Episode ID
                'episode_order':    Order in the conversation
                'utterance_order':  Order in the utterance
                'speaker_id':       Speaker ID (order) in the conversation
                'host_id':          Speaker ID (absolute) of the host
                'host':             Whether the speaker is the host
                'token_ids':        Utterance tokens]
            host_only (bool): Host-only responses
            debug_mode (bool): Whether to print debug logs (default False).
            min_history (int): Minimum # historical utterances
            min_following (int): Guarantee 
        """
        super().__init__()

        self.start = datetime.now()

        # Store maps
        self.gpt_u_map = gpt_utterances_map
        self.gpt_s_map = gpt_sentence_dicts
        self.debug_mode = debug_mode

        # We need to handle discontinuities here - because noise is removed and episode_order
        # is extracted from the transcript directly, there can be gaps in the record.
        # Valid target utterances have `min_history` utterances ahead and `min_following`
        # utterances following them, regardless of nominal "episode_order"

        # Number of utterances per conversation
        episode_utterances = defaultdict(list)
        for (e, e_o), u in self.gpt_u_map.items():
            if host_only:
                if u.get('host'):
                    episode_utterances[e].append(e_o)
            else:
                episode_utterances[e].append(e_o)

        # Only have utterances whose episode order
        min_history = min_history or 0
        min_following = min_following or 0
        episode_utterances = {
            e: sorted(e_u)
            for e, e_u in episode_utterances.items()
        }
        valid_utterances = list(chain.from_iterable([
            [(e, o) for o in e_u if \
                (e_u.index(o) >= min_history and (len(e_u) - e_u.index(o) - 1) >= min_following)
            ] for e, e_u in episode_utterances.items()
        ]))
        self.log(
            self.start,
            'Restricted to {:,}/{:,} utterances{} w/ >= {} turns of history and >= {} turns following (e.g.:)\n{}'
            .format(len(valid_utterances), len(self.gpt_u_map),
                    ' (host only)' if host_only else '', min_history,
                    min_following, valid_utterances[:3]))

        # Create a mapping from data loading index to utterance key
        # Since __getitem__ takes an arbitrary index within [0, len(self))
        # Restrict to utterances more than 3 steps into the conversation
        self.gpt_ix_to_id = dict(
            zip(
                range(len(valid_utterances)),
                valid_utterances,
            ))
        self.log(
            self.start, 'Loaded {} with {:,} utterances{}'.format(
                self.__class__.__name__,
                len(self),
                ' in DEBUG mode' if self.debug_mode else '',
            ))

    def log(self, start, m):
        print('{} [{}] {}'.format(datetime.now() - start,
                                  self.__class__.__name__, m))

    def debug(self, start, m):
        if self.debug_mode:
            self.log(self.start, m)

    def __len__(self):
        return len(self.gpt_ix_to_id)

    def __getitem__(self, index):
        # Format:
        # <speaker/pad> <sent_1> <space> <sent_2> <space> <eos> ...
        # <gold speaker/pad> <gold sent_1> <space> ... <eos>

        # Get gold label
        gold = self.gpt_u_map[self.gpt_ix_to_id[index]]
        host_id = gold['host_id']
        is_host = gold['host']

        # Reconstruct full utterance
        episode_id = gold['episode']
        episode_order = gold['episode_order']
        target_tokens = list(
            chain.from_iterable(
                [self.gpt_s_map[s]['token_ids'] for s in gold['sentences']]))

        # Gold target is up to 512 tokens, including:
        # Speaker ID (1), Target (?), and [END] (1)
        max_target_len = 512 - 1 - 1
        target_tokens = target_tokens[:max_target_len] + [END_GPT2]

        # Get dialogue history, starting from the most recent
        # We prepend a trash token, since that's the padding for our host/guest speaker indices (0)
        # This trash token will be truncated before we pass the finalized embedded batch to
        # the transformer layers.
        history_max_len = 512 - 1  # trash token
        dialogue_history = []

        # We will populate host and guest indices from the end of the history.
        # Since we are constructing the history starting from the most recent.
        # Then, we will do HIST_LEN - INDEX to create the actual index
        host_ix = []
        guest_ix = []
        stop_here = False  # Flag to terminate crawling
        for u in range(episode_order - 1, -1, -1):
            # History utterance
            add_speaker = True  # Flag to prepend speaker tokens to an utterance
            history_utterance = self.gpt_u_map.get((episode_id, u), {})
            if not history_utterance:
                continue
            if not history_utterance.get('sentences', []):
                continue

            # Get whether
            u_host = history_utterance['host']
            u_host_id = history_utterance['host_id']
            if u_host:
                host_id = u_host_id

            # <pad/speaker> <utterance tokens> <eos>
            history_utterance_ended = [END_GPT2]

            # Sentences
            for ii, s_ix in enumerate(history_utterance['sentences'][::-1]):
                try:
                    candidate = self.gpt_s_map[s_ix]['token_ids']
                except:
                    self.log(
                        self.start,
                        'Missing sentence index {} from {:,} sentences'.format(
                            s_ix, len(self.gpt_s_map)))
                    raise

                if not candidate:
                    self.debug(self.start,
                               'Sentence {:,} has no text!'.format(s_ix))
                    continue

                # Dialogue History
                # Make sure this won't overrun
                # Max DH len <= Current DH len + 1 (speaker/pad) + candidate + current utterance
                potential_hist_length = len(dialogue_history) + \
                    1 + \
                    len(candidate) + \
                    len(history_utterance_ended)
                if potential_hist_length > history_max_len:
                    stop_here = True
                    if ii == 0:
                        add_speaker = False
                    break

                history_utterance_ended = candidate + history_utterance_ended

            # For GPT history, if we stop with this utterance/sentence,
            # We leave a spot to insert the speaker ID
            if add_speaker:
                dialogue_history = [
                    PAD_GPT2
                ] + history_utterance_ended + dialogue_history
                if u_host:
                    host_ix.append(len(dialogue_history))
                else:
                    guest_ix.append(len(dialogue_history))

            # Histories may only be 512 tokens long
            if stop_here:
                break

        # Trash token prepended
        dialogue_history = [PAD_GPT2] + dialogue_history
        dh_len = len(dialogue_history)

        # Host/guest indices are calculated from the dialogue history length
        host_ix = [dh_len - h for h in host_ix]
        guest_ix = [dh_len - g for g in guest_ix]

        # Add the current user as host or guest
        full_tokens = dialogue_history + [PAD_GPT2] + target_tokens
        if is_host:
            host_ix += [dh_len]
        else:
            guest_ix += [dh_len]

        if self.debug_mode:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            print('\n\n===========================')

            print('UTTERANCE: {}'.format(gold))
            print('PADDING TOKEN: {}'.format(PAD_GPT2))

            # Decode target
            decoded_target = tokenizer.decode(target_tokens)
            self.log(self.start, 'TARGET:\n{}'.format(decoded_target))

            # Decode history if GPT2
            decoded_history = tokenizer.decode(dialogue_history)
            self.log(self.start, '\nHISTORY:\n{}'.format(decoded_history))

            # Host ix
            self.log(self.start, '\nHOST IX:\n{}'.format(host_ix))
            self.log(
                self.start,
                '\nHOST TK:\n{}'.format([full_tokens[i] for i in host_ix]))

            # Guest ix
            self.log(self.start, '\nGUEST IX:\n{}'.format(guest_ix))
            self.log(
                self.start,
                '\nGUEST TK:\n{}'.format([full_tokens[i] for i in guest_ix]))

            print('===========================\n\n')

        # Utterance mask: 0 for dialogue history + gold speaker, 1 for utterance
        utterance_mask = [0] * (dh_len + 1) + [1] * len(target_tokens)

        # Returns:
        #   Tokens (<= 1024 element list)
        #   Utterance mask (list of same size as `Tokens`, 0/1 binary)
        #   Host ID (int)
        #   Host indices
        #   Guest indices
        return full_tokens, utterance_mask, host_id, host_ix, guest_ix


def collate_gpt_speaker_utterances(batch, device='cpu'):
    """
    Takes in a batch of history and utterance tokens, as well as speaker, and returns
    a padded tensor batch & padding tensor mask

    Arguments:
        batch (list): List of list of tokens, 1 per sample utterance
        device (str): Device onto which to create the tensor

    Returns:
        torch.LongTensor: Padded batch tokens, of shape B x T
        torch.BoolTensor: Padding batch mask, of shape B x T
    """
    full_tokens, utterance_masks, host_ids, host_ixs, guest_ixs = zip(*batch)
    max_len = max(len(t) for t in full_tokens)

    # Pad and mask
    padded_tokens = torch.LongTensor(
        [t + [PAD_GPT2] * (max_len - len(t)) for t in full_tokens]).to(device)
    padding_masks = torch.BoolTensor([[1] * len(t) + [0] * (max_len - len(t))
                                      for t in full_tokens]).to(device)
    utterance_masks = torch.BoolTensor(
        [m + [0] * (max_len - len(m)) for m in utterance_masks]).to(device)

    # Padding for host and guest indices
    max_hix_len = max(len(i) for i in host_ixs)
    p_host_ixs = torch.LongTensor(
        [h + [0] * (max_hix_len - len(h)) for h in host_ixs]).to(device)
    max_gix_len = max(len(i) for i in guest_ixs)
    p_guest_ixs = torch.LongTensor(
        [g + [0] * (max_gix_len - len(g)) for g in guest_ixs]).to(device)

    # Host IDs to tensor of shape B x 1
    host_ids = torch.LongTensor([[hi] for hi in host_ids]).to(device)

    return padded_tokens, padding_masks, utterance_masks, \
        host_ids, p_host_ixs, p_guest_ixs
