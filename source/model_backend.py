"""Explicit model loading and tokenization contract, independent of STAR."""
import json
from pathlib import Path, PurePosixPath

def add_model_arguments(parser):
    parser.add_argument('--local_llm_path', '--model-path', dest='model_path', required=True,
                        help='Local pretrained model directory; shared by runner and resident')
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='bfloat16')


def load_model_bundle(model_path, dtype='bfloat16', *, tokenizer_loader=None):
    """Resolve identity, tokenizer and stop IDs from one model directory.

    Native padding is preserved. A tokenizer without padding uses EOS, as in
    the author pipeline. generation_config EOS overrides config EOS, following
    library configuration precedence. Neither policy changes model text.
    """
    directory = Path(model_path).expanduser().resolve(strict=True)
    metadata = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    architectures = metadata['architectures']
    if (not isinstance(architectures, list) or len(architectures) != 1
            or not isinstance(architectures[0], str) or not architectures[0]
            or not isinstance(metadata['model_type'], str) or not metadata['model_type']):
        raise ValueError('model directory must declare one architecture and model_type')
    if dtype not in ('float16', 'bfloat16'):
        raise ValueError('unsupported precision')
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_loader(str(directory), local_files_only=True, trust_remote_code=False)
    config = dict(model_path=str(directory), model_type=metadata['model_type'],
                  model_class=architectures[0], torch_dtype=dtype,
                  padding_policy='native' if tokenizer.pad_token_id is not None else 'eos')
    configure_tokenizer(tokenizer, config)
    generation_path = directory / 'generation_config.json'
    generation = json.loads(generation_path.read_text(encoding='utf-8')) if generation_path.exists() else {}
    eos = generation['eos_token_id'] if 'eos_token_id' in generation else metadata['eos_token_id']
    eos_ids = [eos] if type(eos) is int else eos
    if (not isinstance(eos_ids, list) or not eos_ids
            or any(type(item) is not int or item < 0 for item in eos_ids)
            or len(set(eos_ids)) != len(eos_ids) or tokenizer.eos_token_id not in eos_ids):
        raise ValueError('model EOS configuration does not match its tokenizer')
    return config, tokenizer, tuple(eos_ids)


def check_model_health(health, model):
    # model_backend is the canonical identity in both resident schema versions.
    # Some old residents duplicate only a subset of its fields at top level.
    backend = health['model_backend']
    for key, expected in model.items():
        if backend[key] != expected or (key in health and health[key] != expected):
            raise ValueError('resident model identity mismatch: ' + key)


def load_model_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {'model_path', 'model_type', 'model_class', 'padding_policy', 'torch_dtype'}:
        raise ValueError('model config must declare path, architecture, padding policy and precision')
    if not config['model_type'] or not config['model_class']:
        raise ValueError('model architecture must be explicit')
    if config['torch_dtype'] not in ('float16', 'bfloat16'):
        raise ValueError('explicit supported model precision required')
    if config['padding_policy'] not in ('native', 'eos') or not PurePosixPath(config['model_path']).is_absolute():
        raise ValueError('explicit padding policy and absolute model path required')
    return config


def verify_architecture(actual, expected):
    if actual.model_type != expected['model_type'] or actual.architectures != [expected['model_class']]:
        raise ValueError('model files differ from declared architecture')


def configure_tokenizer(tokenizer, config):
    # EOS padding is a declared transport policy when native padding is absent.
    if config['padding_policy'] == 'eos':
        if type(tokenizer.eos_token_id) is not int:
            raise ValueError('EOS padding requires a scalar tokenizer EOS')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if type(tokenizer.pad_token_id) is not int:
        raise ValueError('declared padding policy did not provide a padding token')
    tokenizer.padding_side = 'left'


def encode_prompt(tokenizer, prompt):
    # The supported tokenizers prepend their native prefix to an empty input:
    # none for Qwen, BOS for Llama, and gMASK/sop for native GLM.
    options = dict(truncation=False, return_attention_mask=False)
    prefix = tokenizer('', add_special_tokens=True, **options)['input_ids']
    raw = tokenizer(prompt, add_special_tokens=False, **options)
    if prefix and raw['input_ids'][:len(prefix)] == prefix:
        return raw
    return tokenizer(prompt, add_special_tokens=True, **options)


def format_model_prompt(tokenizer, prompt, prompt_format):
    """Model-owned chat framing; never changes task instructions or actions."""
    if prompt_format == 'plain':
        return prompt
    if prompt_format != 'native_chat':
        raise ValueError('unsupported model prompt format')
    if not tokenizer.chat_template:
        raise ValueError('native_chat requires the selected tokenizer chat template')
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                         tokenize=False, add_generation_prompt=True)
