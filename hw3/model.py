from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

## set the input length.
X_LEN = 128


class GPT2Model(nn.Module):
    def __init__(self, tokenizer):
        super(GPT2Model, self).__init__()

        config = {
            "n_positions": X_LEN,
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.vocab["EOS_None"],
            "bos_token_id": tokenizer.vocab["BOS_None"],
            "pad_token_id": tokenizer.vocab["PAD_None"],
            "mask_token_id": tokenizer.vocab["MASK_None"],
            "oov_token_id": tokenizer.vocab["OOV_None"],
        }

        model_config = GPT2Config(**config)
        self.model = GPT2LMHeadModel(model_config)

    def forward(self, x):
        output = self.model(x).logits
        return output
