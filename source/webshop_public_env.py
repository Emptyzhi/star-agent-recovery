"""Table1 WebShop interface with a separate, rendered-observation-only state.

The private native session retains Table1's navigation bookkeeping. It is never
returned to the proposal selector. In particular, the full search-result ASIN
list used by the native wrapper is not a public action catalog.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import re
from typing import Any
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment


ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}
SCORE_LABEL = "Your score (min 0.0, max 1.0)"


class PublicProtocolError(RuntimeError):
    """Invalid interface evidence; no synthesized receipt or retry is allowed."""


def native_clean_str(value: str) -> str:
    """The declared Table1 observation encoding, including its failure modes."""
    return value.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def native_extract_action(raw: str) -> dict[str, Any]:
    """Expose every stage of the original Table1 parser without adding repairs."""
    first_line = raw.split("\n")[0]
    start = first_line.find(":")
    if start > 4:
        start = -1
    end = first_line.find("]", 1)
    extracted = (first_line[start + 1 : end + 1] if end != -1 else first_line[start + 1 :]).strip()
    lowered = extracted[0].lower() + extracted[1:] if len(extracted) > 1 else extracted
    action = lowered.lstrip(" ")
    grammar_ok = re.fullmatch(r"(?:think|search|click)\[[^\r\n]*\]", action) is not None
    return {
        "raw_text": raw,
        "first_line": first_line,
        "extracted": extracted,
        "action": action,
        "grammar_valid": grammar_ok,
        "protocol_failure": None if grammar_ok else "not_a_complete_native_action",
        "parser": "table1_first_line_extract_substring_first_letter_lowercase_v1",
    }


def _visible(text: Any) -> bool:
    if isinstance(text, Comment) or text.parent.name == "[document]":
        return False
    for parent in text.parents:
        if parent.name in {"style", "script", "head", "title", "meta"}:
            return False
        if parent.has_attr("hidden") or parent.get("aria-hidden") == "true":
            return False
        style = re.sub(r"\s+", "", parent.get("style", "")).lower()
        if "display:none" in style or "visibility:hidden" in style:
            return False
    return True


def _render(html: str, url: str, page_type: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Reproduce Table1 text layout, exposing metadata only for emitted text."""
    # Undo only our transport encoding for native selected-label highlighting.
    native_url = unquote(url, encoding="utf-8", errors="strict")
    soup = BeautifulSoup(html, "html.parser")
    texts = [text for text in soup.find_all(string=True) if _visible(text)]
    if page_type == "end":
        labels = [index for index, text in enumerate(texts) if str(text) == SCORE_LABEL]
        if len(labels) != 1 or labels[0] + 1 >= len(texts):
            raise PublicProtocolError("terminal page lacks exactly one visible aggregate score")
        score_text = str(texts[labels[0] + 1])
        score = float(score_text)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise PublicProtocolError("visible terminal score is outside [0,1]")
        observation = native_clean_str(SCORE_LABEL + ": " + score_text)
        return observation, {"reward": score}, {"visible_asins": [], "option_types": {}, "available_actions": [], "visible_buttons": []}

    observation = ""
    option_type = ""
    native_options: dict[str, str] = {}
    native_asins: list[str] = []
    public_options: dict[str, str] = {}
    public_asins: list[str] = []
    public_actions: list[str] = []
    public_buttons: list[str] = []
    count = product_count = since_product = 0
    for node in texts:
        value = str(node)
        if value == "\n" or value.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
            continue
        if node.parent.name == "button":
            processed = f"\n[{value}] "
            public_buttons.append(native_clean_str(value))
            if value != "Search":
                public_actions.append(f"click[{native_clean_str(value)}]")
        elif node.parent.name == "label":
            processed = f"[[{value}]]" if f"'{value}'" in native_url else f"[{value}]"
            native_options[value] = option_type
            public_options[native_clean_str(value)] = native_clean_str(option_type)
            public_actions.append(f"click[{native_clean_str(value)}]")
        elif node.parent.get("class") == ["product-link"]:
            processed = f"\n[{value}] " if product_count < 3 else ""
            if processed:
                public_asins.append(native_clean_str(value))
                public_actions.append(f"click[{native_clean_str(value)}]")
            product_count += 1
            native_asins.append(value)
            since_product = 0
        else:
            processed = "\n" + value + " "
            if count < 2 and page_type != "init":
                processed = ""
            if since_product <= 2 and product_count >= 4:
                processed = ""
            option_type = value
            count += 1
        since_product += 1
        observation += processed
    info: dict[str, Any] = {}
    if native_options:
        info["option_types"] = native_options
    if native_asins:
        info["asins"] = native_asins
    public = {
        "visible_asins": public_asins,
        "option_types": public_options,
        "available_actions": public_actions,
        "visible_buttons": public_buttons,
    }
    return native_clean_str(observation), info, public


@dataclass(frozen=True)
class PublicReceipt:
    action: str
    observation: str
    before: dict[str, Any] | None
    after: dict[str, Any]
    done: bool
    public_terminal_score: float | None
    invalid_action: bool
    invalid_detail: str | None
    http_requests: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WebShopPublicEnv:
    """One unique backend session per attempt, fixed_<task_id> suffix binding."""

    def __init__(self, website: str, session_id: str, task_id: int, timeout: int = 120):
        if not session_id.endswith(f"_fixed_{task_id}"):
            raise PublicProtocolError("session must end in _fixed_<task_id>")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise PublicProtocolError("session ID contains URL path syntax")
        if not website.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise PublicProtocolError("WebShop backend must be the local allocated service")
        self.website = website.rstrip("/")
        self.session_id = session_id
        self.task_id = task_id
        self.timeout = timeout
        self._http = requests.Session()
        self._session: dict[str, Any] | None = None
        self._public: dict[str, Any] | None = None
        self.http_requests = 0

    def close(self) -> None:
        self._http.close()

    def snapshot(self) -> dict[str, Any]:
        if self._public is None:
            raise PublicProtocolError("snapshot requested before reset")
        return deepcopy(self._public)

    def checkpoint(self) -> dict[str, Any]:
        """Capture opaque local session state without exposing private fields."""
        if self._session is None or self._public is None:
            raise PublicProtocolError("checkpoint requested before reset")
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "native_state": deepcopy(self._session),
            "public_state": deepcopy(self._public),
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore the exact logical session state without a replay request."""
        if not isinstance(checkpoint, dict):
            raise PublicProtocolError("checkpoint must be a state object")
        if (checkpoint.get("session_id") != self.session_id
                or checkpoint.get("task_id") != self.task_id):
            raise PublicProtocolError("checkpoint belongs to a different fixed task")
        native_state = checkpoint.get("native_state")
        public_state = checkpoint.get("public_state")
        if not isinstance(native_state, dict) or not isinstance(public_state, dict):
            raise PublicProtocolError("checkpoint is missing exact session state")
        if public_state.get("page_type") == "end":
            raise PublicProtocolError("terminal checkpoint cannot be restored")
        self._session = deepcopy(native_state)
        self._public = deepcopy(public_state)

    def _url(self) -> str:
        state = self._session
        if state is None:
            raise PublicProtocolError("native session is absent")
        page = state["page_type"]
        sid = quote(self.session_id, safe="")
        query, number = state.get("query_string", ""), state.get("page_num", 1)
        asin, options = state.get("asin", ""), state.get("options", {})
        # Keep native Python-literal options; encode each path component once.
        # Flask decodes one percent-encoding layer before route matching. Encode
        # the already encoded query once more so literal slashes remain inside
        # the <keywords> route variable; the backend unwraps one layer.
        query = quote(quote(query, safe=""), safe="")
        number = quote(str(number), safe="")
        asin = quote(asin, safe="")
        options = quote(str(options), safe="")
        if page == "init":
            return f"{self.website}/{sid}"
        if page == "search":
            return f"{self.website}/search_results/{sid}/{query}/{number}"
        if page == "item":
            return f"{self.website}/item_page/{sid}/{asin}/{query}/{number}/{options}"
        if page == "item_sub":
            subpage = quote(state["subpage"], safe="")
            return f"{self.website}/item_sub_page/{sid}/{asin}/{query}/{number}/{subpage}/{options}"
        if page == "end":
            return f"{self.website}/done/{sid}/{asin}/{options}"
        raise PublicProtocolError(f"unknown native page type {page!r}")

    def _native_transition(self, action: str) -> str | None:
        """The original wrapper transition rules; assertions are env failures."""
        if action == "reset":
            self._session = {"session": self.session_id, "page_type": "init"}
            return None
        if self._session is None:
            raise PublicProtocolError("action before reset")
        state = self._session
        if action.startswith("think["):
            return None
        if action.startswith("search["):
            assert state["page_type"] == "init", "search requires init"
            self._session = {"session": self.session_id, "page_type": "search", "query_string": action[7:-1], "page_num": 1}
        elif action.startswith("click["):
            button = action[6:-1]
            if button == "Buy Now":
                assert state["page_type"] == "item", "Buy Now requires item"
                state["page_type"] = "end"
            elif button == "Back to Search":
                assert state["page_type"] in {"search", "item_sub", "item"}, "Back to Search requires a browsing page"
                self._session = {"session": self.session_id, "page_type": "init"}
            elif button == "Next >":
                raise AssertionError("Table1 disables Next >")
            elif button == "< Prev":
                assert state["page_type"] in {"search", "item_sub", "item"}, "< Prev requires a browsing page"
                if state["page_type"] == "search":
                    raise AssertionError("Table1 disables < Prev on search")
                if state["page_type"] == "item_sub":
                    state["page_type"] = "item"
                else:
                    state["page_type"] = "search"
                    state["options"] = {}
            elif button in ACTION_TO_TEMPLATE:
                assert state["page_type"] == "item", "information control requires item"
                state["page_type"] = "item_sub"
                state["subpage"] = button
            elif state["page_type"] == "search":
                assert button in state.get("asins", []), "ASIN absent from native result list"
                state["page_type"] = "item"
                state["asin"] = button
            elif state["page_type"] == "item":
                assert "option_types" in state, "item has no native option labels"
                assert button in state["option_types"], "option label absent from native item"
                state.setdefault("options", {})[state["option_types"][button]] = button
                return f"You have clicked {button}."
        else:
            raise AssertionError("unrecognized native action")
        return None

    def step_actor(self, action: str) -> PublicReceipt:
        if not isinstance(action, str):
            raise PublicProtocolError("action must be text")
        if re.fullmatch(r"(?:think|search|click)\[[^\r\n]*\]", action) is None:
            before = self.snapshot()
            if before["page_type"] == "end":
                raise PublicProtocolError("action after terminal result")
            self._public = deepcopy(before)
            self._public["observation"] = "Invalid action!"
            return PublicReceipt(action, "Invalid action!", before, self.snapshot(), False, None,
                                 True, "malformed_or_unknown_native_action", 0)
        return self.step(action)

    def step(self, action: str) -> PublicReceipt:
        if not isinstance(action, str):
            raise PublicProtocolError("action must be text")
        before = None if self._public is None else self.snapshot()
        if before is not None and before["page_type"] == "end":
            raise PublicProtocolError("action after terminal result")
        try:
            override = self._native_transition(action)
        except AssertionError as exc:
            if before is None:
                raise PublicProtocolError("reset failed before public state exists") from exc
            self._public = deepcopy(before)
            self._public["observation"] = "Invalid action!"
            return PublicReceipt(action, "Invalid action!", before, self.snapshot(), False, None, True, str(exc), 0)
        url = self._url()
        self.http_requests += 1
        response = self._http.get(url, timeout=self.timeout)
        response.raise_for_status()
        page = self._session["page_type"]
        observation, info, rendered = _render(response.text, url, page)
        self._session.update(info)
        if action.startswith("think"):
            observation = "OK."
        elif override is not None:
            observation = override
        if before is None or action == "reset":
            public = {
                "page_type": "init", "query": "", "page_num": 1, "asin": "",
                "selected_options": {}, "subpage": "",
                "visible_asins": [], "option_types": {}, "available_actions": [],
                "visible_buttons": [],
            }
        else:
            public = deepcopy(before)
        # Only an observation actually returned to the Actor can expose a catalog.
        if not action.startswith("think") and override is None:
            public.update(rendered)
        if action.startswith("search["):
            public.update(page_type="search", query=action[7:-1], page_num=1, asin="", selected_options={}, subpage="")
        elif action.startswith("click["):
            button = action[6:-1]
            if button == "Back to Search":
                public.update(page_type="init", query="", page_num=1, asin="", selected_options={}, subpage="")
            elif button == "< Prev":
                public["page_type"] = page
                if page == "search":
                    public["selected_options"] = {}
            elif button in ACTION_TO_TEMPLATE:
                public.update(page_type="item_sub", subpage=button)
            elif button == "Buy Now":
                public["page_type"] = "end"
            elif before is not None and before["page_type"] == "search":
                public.update(page_type="item", asin=button)
            elif override is not None:
                # Native acceptance is authoritative; record its actual selected option.
                public["selected_options"] = deepcopy(self._session["options"])
        public["observation"] = observation
        self._public = public
        done = page == "end"
        score = info["reward"] if done else None
        return PublicReceipt(action, observation, before, self.snapshot(), done, score, False, None, 1)
