import torch
import transformers
from typing import List
from transformers import CLIPTokenizerFast, CLIPTextModel, CLIPTextConfig
from einops import rearrange

transformers.logging.set_verbosity_error()


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# config
REAL_MAX_LENGTH = 77
MAX_LENGTH = REAL_MAX_LENGTH * 3
BOS = 49406
EOS = 49407
DEFAULT_CLIP_NAME = "openai/clip-vit-large-patch14"
CLIP_CONFIGS = {}


# singleton globals
def get_tokenizer(name):
    tokenizer = CLIPTokenizerFast.from_pretrained(name, model_max_length=MAX_LENGTH)
    return tokenizer


def get_model(name):
    model = CLIPTextModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name):
    global CLIP_CONFIGS

    if name not in CLIP_CONFIGS:
        CLIP_CONFIGS[name] = dict()
    if "model" not in CLIP_CONFIGS[name]:
        CLIP_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in CLIP_CONFIGS[name]:
        CLIP_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return CLIP_CONFIGS[name]["model"], CLIP_CONFIGS[name]["tokenizer"]


def get_encoded_dim(name):
    if name not in CLIP_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = CLIPTextConfig.from_pretrained(name)
        CLIP_CONFIGS[name] = dict(config=config)
    elif "config" in CLIP_CONFIGS[name]:
        config = CLIP_CONFIGS[name]["config"]
    elif "model" in CLIP_CONFIGS[name]:
        config = CLIP_CONFIGS[name]["model"].config
    else:
        assert False

    return config.hidden_size


# encoding text
def clip_tokenize(texts: List[str], name=DEFAULT_CLIP_NAME):
    clip, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        clip.cuda()

    device = next(clip.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask


def clip_encode_tokenized_text(
    token_ids,
    attn_mask=None,
    pad_id=EOS,
    name=DEFAULT_CLIP_NAME,
    return_hidden_layer_num=None,
    do_final_ln=False,
):
    assert exists(attn_mask) or exists(pad_id)
    clip, _ = get_model_and_tokenizer(name)

    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())

    clip.eval()

    with torch.no_grad():
        output = clip(
            input_ids=token_ids,
            attention_mask=attn_mask,
            return_dict=True,
            output_hidden_states=exists(return_hidden_layer_num),
        )
        if exists(return_hidden_layer_num):
            encoded_text = output.hidden_states[return_hidden_layer_num]
            if do_final_ln:
                clip.text_model.final_layer_norm(encoded_text)
            encoded_text = encoded_text.detach()
        else:
            encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    encoded_text = encoded_text.masked_fill(
        ~rearrange(attn_mask, "... -> ... 1"), 0.0
    )  # just force all embeddings that is padding to be equal to 0.
    return encoded_text


def clip_encode_text(
    texts: List[str],
    name=DEFAULT_CLIP_NAME,
    return_attn_mask=False,
    return_hidden_layer_num=None,
    do_final_ln=False,
):
    token_ids, attn_mask = clip_tokenize(texts, name=name)
    encoded_text = clip_encode_tokenized_text(
        token_ids,
        attn_mask=attn_mask,
        name=name,
        return_hidden_layer_num=return_hidden_layer_num,
        do_final_ln=do_final_ln,
    )

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask

    return encoded_text


def clip_encode_text_extended(
    texts: List[str],
    name=DEFAULT_CLIP_NAME,
    return_attn_mask=False,
    return_hidden_layer_num=None,
    do_final_ln=False,
):
    token_ids, _ = clip_tokenize(texts, name=name)
    device = token_ids.device
    token_ids_split = token_ids.split(REAL_MAX_LENGTH, dim=1)

    out = []
    for token_ids in token_ids_split:
        attn_mask = torch.any(token_ids != EOS, dim=-1).long()
        tokenized = clip_encode_tokenized_text(
            token_ids,
            attn_mask=attn_mask,
            name=name,
            return_hidden_layer_num=return_hidden_layer_num,
            do_final_ln=do_final_ln,
        )
        out.append(tokenized)
    encoded_text = torch.cat(out, dim=1).to(device)

    if return_attn_mask:
        return encoded_text, torch.any(encoded_text != EOS, dim=-1)
    return encoded_text
