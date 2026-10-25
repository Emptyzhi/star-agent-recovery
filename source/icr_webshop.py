"""Table1 WebShop receipt adapter; public fork qualification never replaces native execution."""

from copy import deepcopy
import re

from icr_core import ActionSupport, ParsedAction, ProtocolError, SupportKind
from icr_prompts import web_failed_log, web_prompt
from webshop_public_env import WebShopPublicEnv, native_extract_action


INFORMATION = frozenset(("Description", "Features", "Reviews", "Attributes"))
BROWSING = frozenset(("search", "item", "item_sub"))


def publicly_callable(snapshot, command):
    match = re.fullmatch(r"(search|click)\[([^\r\n]*)\]", command)
    if match is None:
        return False
    tool, argument = match.groups()
    page = snapshot["page_type"]
    if tool == "search":
        return page == "init"
    if command not in snapshot["available_actions"]:
        return False
    if argument == "Back to Search":
        return page in BROWSING
    if argument == "< Prev":
        return page in {"item", "item_sub"}
    if argument == "Next >":
        return False
    if argument == "Buy Now" or argument in INFORMATION:
        return page == "item"
    if page == "search":
        return argument in snapshot["visible_asins"]
    if page == "item":
        return argument in snapshot["option_types"]
    return False


class WebShopEpisode:
    ordinary_command_constrained = True
    action_limit = 15
    unused_final_proposal = True

    def __init__(self, website, session_id, task_id, base_prompt):
        self.environment = WebShopPublicEnv(website, session_id, task_id)
        self.task = task_id
        self.base_prompt = base_prompt
        self.history = []
        self.initial = None

    @property
    def backend_requests(self):
        return self.environment.http_requests

    @staticmethod
    def _receipt(raw):
        receipt = raw.to_dict()
        receipt["backend_requests"] = receipt["http_requests"]
        receipt["native_accepted"] = not receipt["invalid_action"]
        return receipt

    def reset(self):
        if self.initial is not None:
            raise ProtocolError("use a unique WebShop episode/session for each real attempt")
        self.initial = self._receipt(self.environment.step("reset"))
        return deepcopy(self.initial)

    def snapshot(self):
        return self.environment.snapshot()

    def prompt(self, memory):
        if self.initial is None:
            raise ProtocolError("Actor prompt requested before actual reset")
        return web_prompt(self.base_prompt, self.initial["observation"], self.history, memory)

    def parse(self, text):
        parsed = native_extract_action(text)
        return ParsedAction(parsed["action"], parsed["grammar_valid"],
                            parsed["action"].startswith("think["), parsed)

    def support(self):
        snapshot = self.snapshot()
        if snapshot["page_type"] == "init":
            return ActionSupport(SupportKind.OPEN, (), "table1-webshop-root-search-v1")
        if snapshot["page_type"] not in BROWSING:
            raise ProtocolError("terminal/unknown page has no next-action interface")
        # A command alphabet is a set of executable strings. Repeated rendered
        # labels for one identical native command do not create a new action.
        commands = []
        for command in snapshot["available_actions"]:
            if publicly_callable(snapshot, command) and command not in commands:
                commands.append(command)
        return ActionSupport(SupportKind.FINITE, tuple(commands), "table1-webshop-rendered-commands-v1")

    def public_physical(self, action, support):
        if not action.grammar_valid or action.is_think:
            return False
        if support != self.support():
            raise ProtocolError("fork support no longer matches its actual page")
        return publicly_callable(self.snapshot(), action.action)

    def step(self, action):
        # Never use the smaller rendered catalog to reject a normal native action.
        receipt = self._receipt(self.environment.step_actor(action))
        self.history.append(receipt)
        return deepcopy(receipt)

    def checkpoint(self):
        """Capture environment state and the Actor-visible receipt history."""
        return {
            "environment": self.environment.checkpoint(),
            "initial": deepcopy(self.initial),
            "history": deepcopy(self.history),
        }

    def restore(self, checkpoint):
        """Restore state without issuing network or replay actions."""
        if not isinstance(checkpoint, dict):
            raise ProtocolError("episode checkpoint must be an object")
        if checkpoint.get("initial") != self.initial:
            raise ProtocolError("episode checkpoint initial receipt changed")
        history = checkpoint.get("history")
        if not isinstance(history, list):
            raise ProtocolError("episode checkpoint history is invalid")
        self.environment.restore(checkpoint["environment"])
        self.history = deepcopy(history)

    def context(self):
        return {"task": self.task, "reset": self.initial, "history": self.history,
                "actor_style": "table1_act_full_token_checked", "base_prompt": self.base_prompt}

    def failure_log(self):
        return web_failed_log(self.initial["observation"], self.history)

    def result_metadata(self):
        """Expose only public result fields and their public binding evidence."""
        snapshot = self.snapshot()
        asin = snapshot.get("asin", "")
        if not asin:
            return None

        domains = {"asin": (asin,)}
        for receipt in self.history:
            after = receipt.get("after", {})
            if after.get("asin") != asin:
                continue
            option_types = after.get("option_types", {})
            by_type = {}
            for label, option_type in option_types.items():
                by_type.setdefault(option_type, set()).add(label)
            for option_type, values in by_type.items():
                candidate = (None,) + tuple(sorted(values))
                previous = domains.get(option_type)
                if previous is not None and previous != candidate:
                    raise ProtocolError("public option domain changed for one result binding")
                domains[option_type] = candidate

        projection = {"asin": asin}
        projection.update(snapshot.get("selected_options", {}))
        complete_projection = dict(projection)
        for field in domains:
            if field != "asin":
                complete_projection.setdefault(field, None)

        bindings = {}
        for index, receipt in enumerate(self.history):
            action = receipt.get("action", "")
            before = receipt.get("before") or {}
            after = receipt.get("after") or {}
            if (before.get("page_type") == "search" and after.get("page_type") == "item"
                    and after.get("asin") == asin):
                bindings["asin"] = index
            before_options = before.get("selected_options", {})
            after_options = after.get("selected_options", {})
            for field, value in after_options.items():
                if before_options.get(field) != value:
                    bindings[field] = index

        return {
            "projection": projection,
            "complete_projection": complete_projection,
            "domains": {field: list(values) for field, values in domains.items()},
            "bindings": bindings,
            "environment_contract": "table1-webshop-fixed-task-price-v1",
        }

    @staticmethod
    def action_is_noncommitting(action):
        if action.startswith("think[") or action.startswith("search["):
            return True
        if not action.startswith("click["):
            return False
        return action[6:-1] in INFORMATION or action[6:-1] in {"Back to Search", "< Prev"}

    def parse_reflection(self, text):
        # Archived WebShop Local_llm applies the native first-line extraction
        # to both action and ordinary Reflection requests. Preserve that path.
        return native_extract_action(text)["action"]

    def close(self):
        self.environment.close()
