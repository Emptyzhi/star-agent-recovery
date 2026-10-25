"""Yellow's syntax-only language; arithmetic validity remains the environment's job."""

YELLOW_LANGUAGE = {'kind': 'game24_yellow_arithmetic', 'version': 3,
                   'action_kind': 'operation_or_answer'}
YELLOW_FINAL_LANGUAGE = dict(YELLOW_LANGUAGE, action_kind='answer')

RULES = r'''
operation: number OP number "=" number
answer: "answer:" expr "=" number
?expr: product (("+" | "-") product)*
?product: unary (("*" | "/") unary)*
?unary: "+" unary | "-" unary | atom
?atom: UNSIGNED | "(" expr ")"
number: SIGNED
OP: "+" | "-" | "*" | "/"
SIGNED: /-?(?:[0-9]+\/[0-9]+|[0-9]+(?:\.[0-9]+)?)/
UNSIGNED: /[0-9]+(?:\.[0-9]+)?/
%ignore /[ \t]+/
'''
GRAMMAR = 'start: (operation | answer) "\\n"\n' + RULES
FINAL_GRAMMAR = 'start: answer "\\n"\n' + RULES


def validate_language(spec):
    if spec not in (YELLOW_LANGUAGE, YELLOW_FINAL_LANGUAGE):
        raise ValueError('unknown Game24 arithmetic language')


def language_for_state(state):
    final = 'answer: <complete expression using the four original numbers> = 24'
    binary = '<available binary operation using two current numbers>'
    if state['available_actions'] == [final]:
        return dict(YELLOW_FINAL_LANGUAGE)
    if state['available_actions'] == [binary, final]:
        return dict(YELLOW_LANGUAGE)
    raise ValueError('Game24 public state has no supported Yellow action type')


def grammar_for(spec):
    validate_language(spec)
    if spec['action_kind'] == 'answer':
        return FINAL_GRAMMAR
    return GRAMMAR


class ArithmeticVocabulary:
    """Outlines token surfaces for the declared ASCII arithmetic language only."""
    def __init__(self, tokenizer):
        import json
        expected = dict(type='Sequence', decoders=[
            dict(type='Replace', pattern=dict(String='\u2581'), content=' '),
            dict(type='ByteFallback'), dict(type='Fuse'),
            dict(type='Strip', content=' ', start=1, stop=0)])
        decoder = json.loads(tokenizer.backend_tokenizer.to_str())['decoder']
        if decoder != expected:
            raise ValueError('unverified SentencePiece decoder for Game24 arithmetic')
        from vllm.model_executor.guided_decoding.outlines_logits_processors import _adapt_tokenizer
        adapted = _adapt_tokenizer(tokenizer)
        special = set(tokenizer.all_special_ids)
        self.token_bytes = {
            token: adapted.convert_token_to_string(text).encode('utf-8')
            for text, token in adapted.vocabulary.items() if token not in special}


def build_processor(tokenizer, vocabulary, *, spec=YELLOW_LANGUAGE):
    """Skip lexically impossible tokens before Outlines runs its CFG parser.

    The installed CFG backend otherwise reparses the entire 150k-token vocabulary
    at every generated token. ByteLevel token bytes give an exact lexical filter:
    no character outside this alphabet occurs anywhere in the declared grammar.
    Outlines still decides every remaining token, including parentheses and EOS.
    """
    from copy import copy
    from vllm.model_executor.guided_decoding.outlines_logits_processors import CFGLogitsProcessor
    alphabet = frozenset(b'0123456789+-*/()=answer: \t\n.')
    possible = {token for token, value in vocabulary.token_bytes.items()
                if value and set(value) <= alphabet}
    possible.add(tokenizer.eos_token_id)
    processor = CFGLogitsProcessor(grammar_for(spec), tokenizer)
    adapted = copy(processor._guide.tokenizer)
    adapted.vocabulary = {text: token for text, token in adapted.vocabulary.items() if token in possible}
    assert set(adapted.vocabulary.values()) == possible
    processor._guide.tokenizer = adapted

    class SingleLineProcessor:
        def __call__(self, output_ids, logits):
            # A complete public action ends at its first newline. Outlines'
            # ignored horizontal whitespace must not extend the next line.
            if any(b'\n' in vocabulary.token_bytes[token] for token in output_ids):
                import torch
                mask = torch.full_like(logits, float('-inf'))
                mask[tokenizer.eos_token_id] = logits[tokenizer.eos_token_id]
                return mask
            return processor(output_ids, logits)

    return SingleLineProcessor()

