"""Byte-level delimited parameters: exact framing, UTF-8, and excluded values.

This is a declared generation language, not output repair. Prefix and suffix
use canonical tokenization; parameter bytes remain model-selected.
"""
import codecs
from functools import lru_cache


def validate_spec(spec):
    if type(spec) is not dict or set(spec) != {'prefix', 'suffix', 'forbidden'}:
        raise ValueError('parameter language requires prefix, suffix, forbidden')
    if (type(spec['prefix']) is not str or not spec['prefix']
            or type(spec['suffix']) is not str or len(spec['suffix']) != 1
            or spec['suffix'] in '\r\n' or type(spec['forbidden']) is not list
            or any(type(x) is not str for x in spec['forbidden'])
            or len(set(spec['forbidden'])) != len(spec['forbidden'])):
        raise ValueError('invalid delimited parameter language')
    return spec


class ParameterVocabulary:
    def __init__(self, token_bytes):
        self.token_bytes = token_bytes

    @lru_cache(maxsize=128)
    def body_tokens(self, residual, excluded):
        result = []
        for token, value in self.token_bytes.items():
            if not value or any(b in value for b in excluded):
                continue
            decoder = codecs.getincrementaldecoder('utf-8')('strict')
            try:
                decoder.decode(residual + value, final=False)
            except UnicodeDecodeError:
                continue
            result.append(token)
        if not result:
            raise ValueError('parameter language has no valid continuation')
        return tuple(result)


class ParameterLanguage:
    def __init__(self, spec, vocabulary, encode, eos):
        self.spec = validate_spec(spec)
        self.vocab = vocabulary
        self.prefix = tuple(encode(spec['prefix']))
        self.suffix = tuple(encode(spec['suffix']))
        self.eos = eos
        if not self.prefix or len(self.suffix) != 1:
            raise ValueError('parameter framing requires a single-token suffix')
        for text, tokens in ((spec['prefix'], self.prefix), (spec['suffix'], self.suffix)):
            if b''.join(vocabulary.token_bytes[t] for t in tokens) != text.encode('utf-8'):
                raise ValueError('parameter frame tokenization is not lossless')
        self.excluded = tuple(set(b'\r\n' + spec['suffix'].encode() + spec['prefix'][-1:].encode()))

    def allowed_next(self, tokens):
        tokens = tuple(tokens)
        if len(tokens) < len(self.prefix):
            if tokens != self.prefix[:len(tokens)]:
                raise ValueError('parameter prefix escaped its language')
            return (self.prefix[len(tokens)],)
        if tokens[:len(self.prefix)] != self.prefix:
            raise ValueError('invalid parameter prefix')
        body = tokens[len(self.prefix):]
        closed = bool(body and body[-1] == self.suffix[0])
        data = b''.join(self.vocab.token_bytes[t] for t in (body[:-1] if closed else body))
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        text = decoder.decode(data, final=closed)
        residual, _ = decoder.getstate()
        can_close = not residual and bool(text.strip()) and (
            self.spec['prefix'] + text + self.spec['suffix'] not in self.spec['forbidden'])
        if closed:
            if not can_close:
                raise ValueError('forbidden or empty parameter completed')
            return (self.eos,)
        allowed = self.vocab.body_tokens(residual, self.excluded)
        return (*allowed, self.suffix[0]) if can_close else allowed


def parameter_vocabulary(tokenizer):
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    import json
    if json.loads(tokenizer.backend_tokenizer.to_str())['decoder']['type'] != 'ByteLevel':
        raise ValueError('parameter constraints require a verified byte-level tokenizer')
    decoder = {value: key for key, value in bytes_to_unicode().items()}
    special = set(tokenizer.all_special_ids)
    return ParameterVocabulary({token: bytes(decoder[c] for c in text)
        for text, token in tokenizer.get_vocab().items() if token not in special})
