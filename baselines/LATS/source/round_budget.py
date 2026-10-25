"""User-requested LATS search rounds with a fresh path budget per iteration."""
import copy

from .resident_transport import append


ROUND_ACTIONS = {'game24': 12, 'alfworld': 50, 'webshop': 15}


def install_round_search(core, actions, directory):
    def search(args, task, idx, iterations=2, to_print=False):
        observation = core.env.step(idx, 'reset')[0]
        root = core.Node(state=None, question=observation)
        root.env_state = copy.deepcopy(core.env.sessions)
        core.failed_trajectories, core.reflection_map = [], []
        endpoints = []
        for iteration in range(iterations):
            selected = core.select_node(root)
            if selected is None:
                break
            if selected.is_terminal:
                endpoints.append(selected)
                if selected.reward == 1:
                    return selected.state, selected.value, selected.reward, selected.em
                raise ValueError('LATS selected an unsuccessful terminal node')
            core.iteration_depth_limit = selected.depth + actions
            append(directory/'rounds.jsonl', dict(event='round_started', iteration=iteration+1,
                selected_depth=selected.depth, action_budget=actions,
                depth_limit=core.iteration_depth_limit))
            core.expand_node(selected, args, task, idx)
            if not selected.children:
                append(directory/'rounds.jsonl', dict(event='round_no_progress',
                    iteration=iteration+1, selected_depth=selected.depth,
                    reason='empty_candidate_batch'))
                continue
            core.evaluate_node(selected, args, task, idx)
            candidate = max(selected.children, key=lambda child: child.value)
            # Expansion already executed one action on the chosen path.
            remaining = core.iteration_depth_limit - candidate.depth
            endpoint = core.rollout(candidate, args, task, idx, max_depth=remaining)
            if endpoint.depth > core.iteration_depth_limit:
                raise ValueError('LATS exceeded its round action budget')
            endpoints.append(endpoint)
            core.backpropagate(endpoint, endpoint.reward)
            all_nodes = core.collect_all_nodes(root) + endpoints
            winners = [n for n in all_nodes if n.is_terminal and n.reward == 1]
            append(directory/'rounds.jsonl', dict(event='round_finished', iteration=iteration+1,
                selected_depth=selected.depth, endpoint_depth=endpoint.depth,
                path_actions=endpoint.depth-selected.depth,
                reward=endpoint.reward, success=bool(winners)))
            if winners:
                winner = winners[0]
                return winner.state, winner.value, winner.reward, winner.em
            # Budget exhaustion is an unsuccessful attempt, not an environment
            # terminal: keep the prefix restorable in the following round.
            if not any(f['r'] == endpoint.reward for f in core.failed_trajectories):
                core.failed_trajectories.append(dict(trajectory=core.collect_trajectory(endpoint),
                    final_answer=endpoint.state['action'], r=endpoint.reward))
        best = max(core.collect_all_nodes(root)+endpoints, key=lambda n: n.reward)
        return best.state, best.value, best.reward, best.em

    core.lats_search = search
