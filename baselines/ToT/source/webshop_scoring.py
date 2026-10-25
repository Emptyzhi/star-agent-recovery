"""Report rewards only from purchases actually executed in the branch world."""
import math


def purchase_report(rewards):
    rewards = list(rewards)
    if not all(type(r) in (int, float) and math.isfinite(r) and 0 <= r <= 1 for r in rewards):
        raise ValueError('actual purchase rewards must be finite numbers in [0,1]')
    best = max([0, *rewards])
    return dict(reward=best, success=best == 1, purchase_rewards=rewards,
                reward_policy='webshop_best_observed_purchase_v1')


def world_purchase_report(world):
    rewards = []
    for actions, receipt in world.cache.items():
        if not actions or actions[-1] != 'click[Buy Now]':
            continue
        states = list(world.snapshots[actions].values())
        if len(states) != 1:
            raise ValueError('WebShop purchase snapshot must contain exactly one session')
        # A rejected Buy Now has no purchase; partial purchases may have done=False.
        if states[0]['page_type'] == 'end':
            rewards.append(receipt['reward'])
    return purchase_report(rewards)
