import unittest
from dataclasses import replace

from test_game24.model_api import CallableModelClient, Generation
from test_game24.star_controller import Game24STARController
from source.experiment_settings import Experiment

SHORT_SETTINGS = replace(Experiment.for_dataset('game24'), initial_actions=4, recovery_actions=4)


class ScriptedModel:
    def __init__(self, actions, checkpoint_response="ACTION 0"):
        self.actions = list(actions)
        self.checkpoint_response = checkpoint_response
        self.roles = []

    def __call__(self, prompt, *, role, max_new_tokens, stop, commands, parameter_language=None):
        self.roles.append(role)
        if role == "red_reflection":
            return Generation("The current terminal result is inconsistent with the task; the upstream value dependency must be reconsidered.")
        if role == "red_checkpoint":
            return Generation(self.checkpoint_response)
        if role == "yellow_diagnosis":
            return Generation("The public proposal is locally invalid; try one current arithmetic action.")
        if not self.actions:
            raise AssertionError("scripted model ran out of actions")
        return Generation(self.actions.pop(0))


class STARControllerTests(unittest.TestCase):
    def test_green_can_solve_without_repair(self):
        model = ScriptedModel([
            "4 + 8 = 12",
            "6 - 4 = 2",
            "2 * 12 = 24",
            "answer: (6 - 4) * (4 + 8) = 24",
        ])
        controller = Game24STARController(CallableModelClient(model), settings=SHORT_SETTINGS)
        result = controller.run(0, "4 4 6 8")
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.red_attempts, 0)
        self.assertEqual(result.yellow_entries, 0)

    def test_red_second_reflection_selects_checkpoint(self):
        model = ScriptedModel([
            "1 + 1 = 2",
            "4 * 6 = 24",
            "24 / 2 = 12",
            "answer: (4 * 6) / (1 + 1) = 12",
            "1 - 1 = 0",
            "4 * 6 = 24",
            "24 + 0 = 24",
            "answer: (1 - 1) + (4 * 6) = 24",
        ])
        controller = Game24STARController(CallableModelClient(model), settings=SHORT_SETTINGS)
        result = controller.run(1, "1 1 4 6")
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(model.roles.count("red_reflection"), 1)
        self.assertEqual(model.roles.count("red_checkpoint"), 1)
        restored = [e for e in result.events if e["event"] == "red_checkpoint_restored"]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["checkpoint_index"], 0)
        self.assertEqual(restored[0]["checkpoint_selection_response"], "ACTION 0")
        delivered = [e for e in result.events if e["event"] == "event_subtree_delivered"]
        self.assertEqual(len(delivered), 1)

    def test_invalid_red_checkpoint_restores_root_and_resumes_green(self):
        model = ScriptedModel(
            [
                "1 + 1 = 2",
                "4 * 6 = 24",
                "24 / 2 = 12",
                "answer: (1 + 1) + (4 * 6) / 2 = 12",
                "1 - 1 = 0",
                "4 * 6 = 24",
                "24 + 0 = 24",
                "answer: (1 - 1) + (4 * 6) = 24",
            ],
            checkpoint_response="ACTION 99",
        )
        controller = Game24STARController(CallableModelClient(model), settings=SHORT_SETTINGS)
        result = controller.run(4, "1 1 4 6")
        self.assertEqual(result.status, "PROTOCOL_ERROR")
        self.assertFalse(result.success)
        self.assertEqual(result.cost['checkpoint_restores'], 0)
        self.assertTrue(any(e["event"] == "red_checkpoint_selection_failed" for e in result.events))

    def test_yellow_is_one_action_and_does_not_force_a_page_action(self):
        model = ScriptedModel([
            "1 + 1 = 99",
            "1 + 2 = 3",
            "3 + 3 = 6",
            "6 * 4 = 24",
            "answer: (1 + 2 + 3) * 4 = 24",
        ])
        controller = Game24STARController(CallableModelClient(model), settings=SHORT_SETTINGS)
        result = controller.run(2, "1 2 3 4")
        self.assertEqual(result.status, "SUCCESS")
        self.assertGreaterEqual(result.yellow_entries, 1)
        self.assertTrue(any(e['event'] == 'action_executed' and e['node']['scope'] == 'YELLOW' for e in result.events))

    def test_red_handoff_is_not_repeated_after_first_green_request(self):
        model = ScriptedModel([
            "1 + 1 = 2", "4 * 6 = 24", "24 / 2 = 12",
            "answer: (4 * 6) / (1 + 1) = 12",
            "1 - 1 = 0", "4 * 6 = 24", "24 + 0 = 24",
            "answer: (1 - 1) + (4 * 6) = 24",
        ])
        controller = Game24STARController(CallableModelClient(model), settings=SHORT_SETTINGS)
        result = controller.run(3, "1 1 4 6")
        self.assertEqual(
            len([e for e in result.events if e["event"] == "event_subtree_delivered"]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
