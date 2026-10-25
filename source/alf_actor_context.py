"""Build ALFWorld native action context from ordinary history."""
from icr_core import ProtocolError


def memory_base(base, memory):
    if memory:
        base += "\nYour memory for the task below:"
        for index, item in enumerate(memory):
            base += f"\nTrial {index}:\n{item.strip()}"
    return base


def _actor_history(episode):
    return ''.join('> ' + row['action'] + '\n' + row['observation'] + '\n'
                   for row in episode.history)


def actor_input(episode, memory):
    text = memory_base(episode.base_prompt, memory)
    text += '\nHere is the task:\n' + episode.initial['observation'] + '\n'
    text += _actor_history(episode)
    return text
