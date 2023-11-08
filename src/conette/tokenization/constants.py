#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Begin of sentence token
BOS_TOKEN = "<bos>"
# End of sentence token
EOS_TOKEN = "<eos>"
# Pad token
PAD_TOKEN = "<pad>"
# Unknown token
UNK_TOKEN = "<unk>"

# Special tokens list. Order matters because it will define the index of the specials tokens in trainable tokenizers.
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
