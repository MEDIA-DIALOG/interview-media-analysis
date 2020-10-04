import gc

from transformers import GPT2Tokenizer, RobertaTokenizer

# ROBERTA
DH_ROBERTA = [38462, 35]  # 'DH:'
U_ROBERTA = [791, 35]  # 'U:'
START_ROBERTA = 0  # <s>
END_ROBERTA = 2  # </s>
PAD_ROBERTA = 1  # <pad>
SPEAKER_ROBERTA = [
    [104, 288, 35],  # 'S0:'
    [104, 134, 35],  # 'S1:'
]

# GPT2
DH_GPT = [41473, 25]  # 'DH:'
U_GPT = [52, 25]  # 'U:'
START_GPT2 = 50256  # <|endoftext|> GPT2 didn't have a start token...
END_GPT2 = 50256  # <|endoftext|> again
PAD_GPT2 = 49129  # '..................' <- WTF?
SPEAKER_GPT2 = [
    [50, 15, 25],  # 'S0: '
    [50, 16, 25],  # 'S1: '
]
GPT2_OG_TOK_LEN = 50257

# BERT
START_BERT = 101  #'[CLS]'
END_BERT = 102  # '[SEP]'
PAD_BERT = 0  # '[PAD]'
