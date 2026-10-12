"""Finite complete-command prefix language used only by the alternative decoder."""


_END = object()


class CommandTrie:
    def __init__(self, sequences, eos_token_id):
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("trie EOS must be an explicit token ID")
        if type(sequences) is not tuple or not sequences:
            raise ValueError("trie requires a nonempty tuple of token sequences")
        self.eos = eos_token_id
        self.root = {}
        for sequence in sequences:
            if type(sequence) is not tuple or not sequence:
                raise ValueError("a command must have a nonempty token sequence")
            node = self.root
            for token in sequence:
                if type(token) is not int or token < 0 or token == self.eos:
                    raise ValueError("command contains an invalid or embedded EOS token")
                if token not in node:
                    node[token] = {}
                node = node[token]
            if _END in node:
                raise ValueError("distinct command entries have duplicate token sequences")
            node[_END] = True

    def allowed_next(self, generated):
        node = self.root
        for token in generated:
            if token not in node or token is _END:
                raise ValueError("generated prefix left the declared command trie")
            node = node[token]
        allowed = [token for token in node if token is not _END]
        if _END in node:
            allowed.append(self.eos)
        if not allowed:
            raise ValueError("command trie has a dead prefix")
        return allowed
