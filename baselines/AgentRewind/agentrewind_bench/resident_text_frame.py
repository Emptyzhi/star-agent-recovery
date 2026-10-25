"""Opt-in token-time boundary for native text commands; never edit sampled text."""
from .framing import decode_sectioned_reply

STOP_MODE='agentrewind_text_command'


def command_complete(text):
    command=text.lstrip()
    if command.startswith(('backtrack_candidates','backtrack_commit')):
        try:
            name,_=decode_sectioned_reply(text,whole_response=True)
        except ValueError:
            return False
        return name in ('backtrack_candidates','backtrack_commit')
    return '\n' in text


def install():
    """Extend only the new resident process; existing service files stay intact."""
    import icr_output_framing as framing
    if STOP_MODE in framing.STOP_MODES:
        raise RuntimeError('AgentRewind text boundary already installed')
    original=framing.frame_complete

    def frame_complete(text,mode):
        return command_complete(text) if mode==STOP_MODE else original(text,mode)

    framing.STOP_MODES=(*framing.STOP_MODES,STOP_MODE)
    framing.frame_complete=frame_complete
