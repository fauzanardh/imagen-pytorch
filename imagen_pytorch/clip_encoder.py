import math
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
MAX_LENGTH = 77
DEFAULT_CLIP_NAME = "openai/clip-vit-large-patch14"
CLIP_CONFIGS = {}


# singleton globals
def get_tokenizer(name):
    tokenizer = CLIPTokenizerFast.from_pretrained(name)
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
        truncation=False,
        add_special_tokens=False,
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask


def clip_encode_tokenized_text(
    token_ids,
    attn_mask=None,
    # pad_id=EOS,
    name=DEFAULT_CLIP_NAME,
    return_hidden_layer_num=None,
    do_final_ln=False,
):
    clip, tokenizer = get_model_and_tokenizer(name)

    assert exists(attn_mask) or exists(tokenizer.eos_token_id)
    attn_mask = default(attn_mask, lambda: (token_ids != tokenizer.eos_token_id).long())

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
    _, tokenizer = get_model_and_tokenizer(name)

    # tokens
    comma_token_id = [
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token == ",</w>"
    ][0]

    token_ids, _ = clip_tokenize(texts, name=name)
    device = token_ids.device

    remade_tokens_batch: List[List[int]] = []
    for i in range(token_ids.size(0)):
        remade_tokens: List[int] = []
        last_comma = -1
        for token_id in token_ids[i]:
            if token_id == comma_token_id:
                last_comma = len(remade_tokens)
            elif (
                max(len(remade_tokens), 1) % (MAX_LENGTH - 2) == 0 and last_comma != -1
            ):
                last_comma += 1
                relocate_tokens = remade_tokens[last_comma:]
                remade_tokens = remade_tokens[:last_comma]
                length = len(remade_tokens)
                remaining = (
                    int(math.ceil(length / (MAX_LENGTH - 2))) * (MAX_LENGTH - 2)
                    - length
                )
                remade_tokens += [tokenizer.eos_token_id] * remaining + relocate_tokens

            remade_tokens.append(token_id)

        token_count = len(remade_tokens)
        prompt_target_count = math.ceil(max(token_count, 1) / (MAX_LENGTH - 2)) * (
            MAX_LENGTH - 2
        )
        remade_tokens += [tokenizer.eos_token_id] * (prompt_target_count - token_count)
        remade_tokens_batch.append(remade_tokens)

    out = None
    i = 0
    while max(map(len, remade_tokens_batch)) != 0:
        remaining_tokens_batch = [x[(MAX_LENGTH - 2) :] for x in remade_tokens_batch]

        tokens_ids = []
        for j in range(len(remade_tokens_batch)):
            if len(remade_tokens_batch[j]) > 0:
                tokens_ids.append(remade_tokens_batch[j][: MAX_LENGTH - 2])
            else:
                tokens_ids.append([tokenizer.eos_token_id] * (MAX_LENGTH - 2))
        tokens_tensor = torch.tensor(tokens_ids, device=device)
        attn_mask = (tokens_tensor != tokenizer.eos_token_id).long()

        encoded_text = clip_encode_tokenized_text(
            tokens_tensor,
            attn_mask=attn_mask,
            name=name,
            return_hidden_layer_num=return_hidden_layer_num,
            do_final_ln=do_final_ln,
        )

        if out is None:
            out = encoded_text
        else:
            out = torch.cat([out, encoded_text], dim=1)

        remade_tokens_batch = remaining_tokens_batch
        i += 1

    if return_attn_mask:
        return out, torch.any(out != 0.0, dim=-1)  # type: ignore
    return out
