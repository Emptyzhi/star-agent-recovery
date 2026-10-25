"""Opt-in Game24 syntax masking around the unchanged resident engine."""
from pathlib import Path
import sys


def install(runtime_package):
    runtime = Path(runtime_package)
    sys.path[:0] = [str(runtime/'source'), str(runtime/'vllm_service')]
    import star_vllm_service as service
    from .action_language import (ArithmeticVocabulary, YELLOW_LANGUAGE,
                                  validate_language, build_processor)

    original_validate = service.validate_spec
    original_language = service.ParameterLanguage
    original_processor = service.StarProcessor
    original_service = service.Service
    original_vocabulary = service.parameter_vocabulary
    tokenizer = {}

    def vocabulary(value):
        import json
        decoder = json.loads(value.backend_tokenizer.to_str())['decoder']
        if decoder['type'] == 'ByteLevel':
            return original_vocabulary(value)
        return ArithmeticVocabulary(value)

    class ArithmeticMask:
        def __init__(self, vocabulary, spec):
            self.spec = dict(spec)
            self.processor = build_processor(tokenizer['value'], vocabulary, spec=spec)

    def validate(spec):
        if spec.get('kind') == YELLOW_LANGUAGE['kind']:
            validate_language(spec)
        else:
            original_validate(spec)

    def language(spec, vocabulary, encode, eos):
        if spec.get('kind') == YELLOW_LANGUAGE['kind']:
            validate_language(spec)
            return ArithmeticMask(vocabulary, spec)
        if isinstance(vocabulary, ArithmeticVocabulary):
            raise ValueError('SentencePiece Game24 vocabulary does not expose WebShop parameter framing')
        return original_language(spec, vocabulary, encode, eos)

    class Processor(original_processor):
        def __call__(self, prompt_ids, output_ids, logits):
            if isinstance(self.pending.mask, ArithmeticMask):
                return self.pending.mask.processor(output_ids, logits)
            return super().__call__(prompt_ids, output_ids, logits)

    class Service(original_service):
        def __init__(self):
            super().__init__()
            self.parameter_vocab = vocabulary(self.tokenizer)
            self.parameter_language_supported = True
            tokenizer['value'] = self.tokenizer

        def health(self, experiment=None):
            result = super().health(experiment)
            if isinstance(self.parameter_vocab, ArithmeticVocabulary):
                result['parameter_language_supported'] = False
            return dict(result, game24_action_grammar_supported=True,
                        game24_action_grammar_version=YELLOW_LANGUAGE['version'])

        def response(self, pending, output):
            result = super().response(pending, output)
            if isinstance(pending.mask, ArithmeticMask):
                result['game24_action_language'] = pending.mask.spec
            return result

    service.validate_spec, service.ParameterLanguage = validate, language
    service.parameter_vocabulary = vocabulary
    service.StarProcessor, service.Service = Processor, Service
    return service


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] != '--runtime-package':
        raise SystemExit('Pass --runtime-package before resident arguments')
    runtime = sys.argv[2]
    del sys.argv[1:3]
    service = install(runtime)
    service.main(service.build_parser().parse_args())
