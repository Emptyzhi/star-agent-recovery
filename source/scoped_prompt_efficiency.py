"""Compact public evidence and one model assessment per exhausted path."""

import json
import re

from icr_core import ProtocolError, SupportKind
from tree_history import canonical
from webshop_public_env import native_extract_action


RED_DECLINE_CONTRACT = (
    "\nDecision-specific reason contract: RESUME must describe the cached position, remaining n actions, "
    "and an estimated public action chain that reaches a terminal better aligned with the task within n actions. "
    "For BACK or UNKNOWN, report the position, remaining budget and the specific public obstacle or missing evidence "
    "briefly; do not construct or repeat an action chain for a checkpoint you decline. "
    "An unbounded repetition of option clicks is not a plan. This decision-specific rule replaces any generic "
    "action-chain formatting instruction in the history/view. Return the decision on line1 and a free-text reason "
    "on line2, then end the message. Do not invent facts or a replacement decision."
)


def terminal_summary(evidence):
    rows = []
    for entries in evidence.values():
        if not entries:
            raise ProtocolError("empty terminal evidence group")
        rows.append({"submitted_state": entries[-1]["metadata"]["complete_projection"],
                     "count": len(entries), "scores": [e["score"] for e in entries]})
    return rows


def parse_red_batch(text, indices):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    rows = []
    expected_indices = set(indices)
    seen = set()
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].startswith("CHECKPOINT "):
            raise ProtocolError("Red repair requires a CHECKPOINT line")
        try:
            index = int(lines[cursor].split(None, 1)[1])
        except (IndexError, ValueError) as error:
            raise ProtocolError("Red CHECKPOINT must contain an integer index") from error
        if index not in expected_indices or index in seen:
            raise ProtocolError("Red CHECKPOINT must identify each supplied candidate exactly once")
        if cursor + 1 >= len(lines) or not lines[cursor + 1].startswith("RESTORE_STEP "):
            raise ProtocolError("Red repair requires one RESTORE_STEP line")
        try:
            restore_step = int(lines[cursor + 1].split(None, 1)[1])
        except (IndexError, ValueError) as error:
            raise ProtocolError("Red RESTORE_STEP must contain an integer") from error
        reported_restore_step = restore_step
        if restore_step != index:
            # Preserve the malformed candidate for audit; the controller skips it.
            restore_step = None
        seen.add(index)
        cursor += 2
        prefix = "REPLACEMENT_ACTION "
        if cursor >= len(lines) or not lines[cursor].startswith(prefix):
            raise ProtocolError("Red repair requires one concrete REPLACEMENT_ACTION line")
        action = lines[cursor][len(prefix):].strip()
        if not action:
            raise ProtocolError("Red replacement_action cannot be empty")
        cursor += 1
        evidence = ""
        if cursor < len(lines) and lines[cursor].startswith("EVIDENCE "):
            evidence = lines[cursor][len("EVIDENCE "):].strip()
            cursor += 1
        rows.append({"checkpoint_index": index, "restore_step": restore_step,
                     "reported_restore_step": reported_restore_step,
                     "replacement_action": action, "reason": evidence or "structured repair candidate",
                     "evidence": evidence})
    if seen != expected_indices:
        raise ProtocolError("Red repair must cover every supplied candidate")
    return rows


def parse_red_choice(text, indices):
    """Parse one bounded Red choice; keep non-control trailing text as evidence.

    Red's first three lines are the machine-readable control prefix.  The
    remaining text is model-authored evidence/reasoning and does not change
    the selected checkpoint or action.  A second control line remains an
    explicit protocol error rather than being silently interpreted.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ProtocolError("Red choice cannot be empty")

    control_line = re.compile(
        r"(?:CHECKPOINT|RESTORE_STEP)(?:\s+\S+)?|REPLACEMENT_ACTION(?:\s+.*)?|DECLINE"
    )

    def free_evidence(start):
        tail = lines[start:]
        if any(control_line.fullmatch(line) for line in tail):
            raise ProtocolError("Red choice contains an additional control line")
        evidence = []
        for line in tail:
            evidence.append(line[len("EVIDENCE "):].strip()
                            if line.startswith("EVIDENCE ") else line)
        return "\n".join(item for item in evidence if item)

    if lines[0] == "DECLINE":
        free_evidence(1)
        return None
    if len(lines) < 3 or not lines[0].startswith("CHECKPOINT "):
        raise ProtocolError("Red choice requires CHECKPOINT, RESTORE_STEP and REPLACEMENT_ACTION")
    try:
        index = int(lines[0].split(None, 1)[1])
        restore = int(lines[1].split(None, 1)[1]) if lines[1].startswith("RESTORE_STEP ") else None
    except (IndexError, ValueError) as error:
        raise ProtocolError("Red choice has invalid checkpoint or restore step") from error
    if index not in set(indices) or restore != index:
        raise ProtocolError("Red choice checkpoint and restore step must identify one supplied candidate")
    prefix = "REPLACEMENT_ACTION "
    if not lines[2].startswith(prefix) or not lines[2][len(prefix):].strip():
        raise ProtocolError("Red choice requires one replacement action")
    evidence = free_evidence(3)
    return {"checkpoint_index": index, "restore_step": restore,
            "reported_restore_step": restore, "replacement_action": lines[2][len(prefix):].strip(),
            "reason": evidence, "evidence": evidence}


class YellowChoiceError(ProtocolError):
    """A model-authored Yellow choice needs explicit protocol clarification."""


def parse_red_reason_first(text, indices):
    """Declared five-line selection or three-line decline; no output repair."""
    if type(text) is not str:
        raise ProtocolError('Red output must be text')
    lines=[line.strip() for line in text.strip().splitlines()]
    if len(lines) not in (3,5) or not lines[0].startswith('EVIDENCE ') or lines[1]!='DECISION':
        raise ProtocolError('Red requires one EVIDENCE line, DECISION, then one three-line choice or DECLINE')
    evidence=lines[0][len('EVIDENCE '):].strip()
    if not evidence:
        raise ProtocolError('Red evidence must be nonempty')
    control=lines[2:]
    if control==['DECLINE']:
        return None
    if len(control)!=3:
        raise ProtocolError('Red decision requires exactly three control lines or DECLINE')
    choice=parse_red_choice('\n'.join(control),indices)
    if choice is None:
        raise ProtocolError('Red selection unexpectedly declined')
    return {**choice,'reason':evidence,'evidence':evidence}


def parse_yellow_choice(text, support, *, progress_recovery=False):
    """Parse one canonical action field without guessing between encodings."""
    def unique_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise YellowChoiceError("Yellow choice contains duplicate JSON fields")
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=unique_fields)
    except json.JSONDecodeError as error:
        raise YellowChoiceError(
            f"Yellow choice requires one JSON object: {error.msg} at line {error.lineno}, "
            f"column {error.colno}. JSON string values require double quotes.") from error
    if type(value) is not dict or set(value) != {"role", "reason", "action"}:
        raise YellowChoiceError("Yellow requires exactly role, reason and one action field")
    role = value["role"].strip().upper() if type(value["role"]) is str else None
    if role not in (("PREPARE", "CORRECT", "RETURN_GREEN_UNRESOLVED") if progress_recovery else ("PREPARE", "CORRECT")):
        raise YellowChoiceError("Yellow choice has an unknown role")
    if type(value["reason"]) is not str or not value["reason"].strip():
        raise YellowChoiceError("Yellow choice requires a reason")
    action = value["action"]
    if role == "RETURN_GREEN_UNRESOLVED":
        if action is not None:
            raise YellowChoiceError("RETURN_GREEN_UNRESOLVED requires action=null; it executes no action")
        return {"diagnosis": value["reason"], "step_role": role, "current_action": None}
    if type(action) is not str or not action.strip():
        raise YellowChoiceError("The canonical action field must be nonempty text")
    if support.contract_id == "alfworld-public-admissible-v1":
        # Callability belongs to the controller's existing YellowCallabilityError
        # path, which records exhausted proposals without executing them.
        if action != action.strip() or '\n' in action or '\r' in action or action.startswith('think:'):
            raise YellowChoiceError("ALFWorld action must be one exact native command line")
        return {"diagnosis": value["reason"], "step_role": role, "current_action": action}
    parsed = native_extract_action(action)
    if (parsed['action'] != action or not parsed['grammar_valid']
            or action.startswith('think[')):
        raise YellowChoiceError("Yellow action arguments do not form one exact native action")
    return {"diagnosis": value["reason"], "step_role": role, "current_action": action}


def red_candidate_evidence(episode, candidates, path, terminals):
    # Exact duplicate page observations share a reference; none are truncated.
    observations, observation_ids = [], {}
    interfaces, interface_ids = [], {}
    rows = []
    for index, checkpoint in candidates:
        state = checkpoint["environment"]["public_state"]
        interface_keys = (("available_actions",) if state.get("benchmark") == "alfworld" else
                          ("available_actions", "option_types", "visible_asins", "visible_buttons"))
        interface = {k: state[k] for k in interface_keys}
        interface_key = canonical(interface)
        if interface_key not in interface_ids:
            interface_ids[interface_key] = len(interfaces)
            interfaces.append(interface)
        row = {"checkpoint_index": index, "restore_step": index,
               "remaining_actions": episode.action_limit - index,
               "state": {k: v for k, v in state.items() if k not in (*interface_keys, "observation")},
               "public_interface_id": interface_ids[interface_key]}
        page = state["observation"]
        if page not in observation_ids:
            observation_ids[page] = len(observations)
            observations.append(page)
        row["observation_id"] = observation_ids[page]
        # Option clicks expose only a short receipt; retain the actual item page.
        extra = episode._current_item_evidence(state, checkpoint["history"])
        extra += episode._current_search_evidence(state, checkpoint["history"])
        if extra:
            if extra not in observation_ids:
                observation_ids[extra] = len(observations)
                observations.append(extra)
            row["page_evidence_id"] = observation_ids[extra]
        rows.append(row)
    evidence = {"task": episode.initial["observation"], "candidates": rows,
            "failed_path_actions": [row["action"] for row in path],
            "public_interfaces": interfaces,
            "page_observations": observations, "executed_path": path,
            "observed_terminals": terminal_summary(terminals)}
    if episode.snapshot().get('benchmark')=='alfworld':
        from alf_public_receipts import add_checkpoint_receipts
        add_checkpoint_receipts(evidence,candidates)
    return evidence


def red_batch_prompt(episode, candidates, path, terminals):
    evidence = red_candidate_evidence(episode, candidates, path, terminals)
    if episode.snapshot().get("benchmark") == "alfworld":
        return episode.red_batch_prompt(evidence)
    system = (
        "你处于Red，当前真实执行路径已经失败。一次审查全部缓存候选，环境此时不会恢复。"
        "每个候选的state和remaining_actions由真实快照提供，是唯一的位置和剩余步数依据；"
        "checkpoint_index和restore_step是同一恢复位置的两个标签，数值必须完全相同，"
        "逐字复制所选候选给出的整数，不加1、不减1。它们表示已保留动作数，不是接下来动作的序号。"
        "不能将历史下一动作的结果当成候选当前位置。候选按从深到浅排列。"
        "只选择一个候选，输出三或四行简单文本：CHECKPOINT n、RESTORE_STEP n、"
        "REPLACEMENT_ACTION <一个具体动作>、EVIDENCE <可选自由文本证据>；"
        "没有有依据的选择时只输出DECLINE和可选EVIDENCE。不要输出JSON或Markdown。"
        "reason是自由文本，不要求中文字段名或固定标点。禁止捏造商品、选项、隐藏得分或已经执行未来动作。"
        "replacement_action必须是该checkpoint快照中公开可调用的动作。历史动作并非禁用动作；"
        "可以重用必要步骤，但应根据公开失败证据说明此次续跑准备改变什么。没有依据时选择DECLINE；不要输出NONE。"
        "公开动作可调用只说明能执行，不说明它能修复已观察的失败。"
        "执行回执与任务评价必须区分：accepted=true且done=true说明环境已经接受动作并返回终局，"
        "非满分是任务要求未全部满足，不是动作没执行；除非有明确拒绝或异常回执，"
        "不得归因为页面加载、请求未提交或环境故障。无法定位原因时明确保留不确定性。"
        "对照原任务与失败路径的公开结果，选择仍有预算落实改变的位置；"
        "EVIDENCE可简述已有失败依据、准备改变的决策和本步之后的续跑方向，不要求根因证明。"
        "在未改变状态或取得新证据时原样重放终局动作，通常只会重现已有结果；"
        "需要重用某个历史动作来导航或查证时仍然允许，但要区分必要前缀与原样重复失败后缀。"
        "输出中的动作必须是所选位置现在可调用的一步，不是未来状态的动作。"
        "输出模板（k须替换为候选的同一个整数；尖括号内容须替换为实际内容）：\n"
        "CHECKPOINT k\nRESTORE_STEP k\nREPLACEMENT_ACTION <当前可调用动作>\nEVIDENCE <简短公开依据与改变方向>\n"
        "前三行只能出现一次且定义唯一控制选择；EVIDENCE之后可以继续写多行自由证据或解释，"
        "这些内容会被保留为evidence。不要把解释放在REPLACEMENT_ACTION行，不要输出第二个候选或第二组控制字段。")
    if getattr(episode, "progress_recovery", False):
        system = (
            "Review cached checkpoints only. Select at most one supplied checkpoint and output three or four plain-text lines: "
            "CHECKPOINT n, RESTORE_STEP n, REPLACEMENT_ACTION followed by one callable native action, and optional "
            "EVIDENCE free text. If no candidate is justified, output DECLINE and optional free evidence. "
            "After the control prefix, additional non-control evidence lines are allowed and retained; "
            "do not output a second checkpoint, restore step, replacement action, or JSON/Markdown. ")
    return episode.review_formatter([{"role": "system", "content": system},
        {"role": "user", "content": "[RED_BATCH_REVIEW]\n" + canonical(evidence)}])


class RedSelectionError(ProtocolError):
    """An explicit joint checkpoint selection violates the public contract."""


def json_envelope(text):
    """Declared wire formats: bare JSON or exactly one complete JSON code fence."""
    value = text.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0] not in ('```json', '```') or lines[-1] != '```':
            raise ValueError('invalid JSON code-fence envelope')
        value = '\n'.join(lines[1:-1])
    return value


def parse_red_selection(text, remaining_by_index):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RedSelectionError("duplicate Red selection field")
            value[key] = item
        return value
    try:
        row = json.loads(json_envelope(text), object_pairs_hook=unique)
    except ValueError as error:
        raise RedSelectionError("Red must return one JSON selection") from error
    if type(row) is not dict or set(row) != {"checkpoint_index", "reason", "action_plan"}:
        raise RedSelectionError("Red selection requires checkpoint_index, reason, action_plan")
    if type(row['reason']) is not str or not row['reason'].strip():
        raise RedSelectionError("Red reason must be nonempty free text")
    index, plan = row['checkpoint_index'], row['action_plan']
    if type(plan) is not list or any(type(a) is not str or not a.strip() for a in plan):
        raise RedSelectionError("action_plan must be a list of nonempty action steps")
    if index is None:
        if plan:
            raise RedSelectionError("no selected checkpoint requires an empty action_plan")
    elif type(index) is not int or index not in remaining_by_index:
        raise RedSelectionError("checkpoint_index must identify a supplied candidate or be null")
    elif not 1 <= len(plan) <= remaining_by_index[index]:
        raise RedSelectionError("action_plan length must fit the selected checkpoint's remaining actions")
    return row


def red_selection_prompt(episode, candidates, path, terminals, *, feedback=None, phase=None, assessment=None, choices=None):
    if episode.tree_internal_repair:
        evidence = red_candidate_evidence(episode,candidates,path,terminals)
        evidence['CHOICES'] = list(choices)
        return episode.review_formatter([{'role':'system','content':
            'You are Red. Select exactly one checkpoint from CHOICES or UNKNOWN. '
            'Compare the supplied public snapshots and failed suffix. Select a useful recovery point with enough '
            'remaining actions for a bounded Yellow repair segment and task completion. '
            'The correct prefix may be reused. Do not require an untried immediate action or a guaranteed score. '
            'Only the selected checkpoint is restored. Yellow then repairs that suffix; you do not choose its actions. '
            'Output only a CHOICES token; no explanation, plan, or JSON.'},
            {'role':'user','content':'[RED_SELECT_V41]\n'+json.dumps(evidence,ensure_ascii=False,separators=(',',':'))}])
    if episode.efficient_recovery and phase == 'choose':
        evidence = {'task': episode.initial['observation'], 'assessment': assessment, 'CHOICES': list(choices),
            'candidates': [{'checkpoint_index':i, 'remaining_actions':episode.action_limit-i,
                'state':{k:v for k,v in cached['environment']['public_state'].items()
                         if k not in ('observation', 'option_types', 'visible_buttons') }} for i,cached in candidates]}
        return episode.review_formatter([{'role':'system','content':
            'The prior assessment inspected every public snapshot. Select one option from CHOICES. '
            'Current candidate states and callable actions are authoritative; assessment is not a verified plan. '
            'Output only the selected option, without explanation.'},
            {'role':'user','content':'[RED_CHECKPOINT_SELECTION]\n' + json.dumps(
                evidence, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}])
    evidence = red_candidate_evidence(episode, candidates, path, terminals)
    evidence['execution_graph'] = {"root_id": "root", "nodes": [
        {"id": "root", "parent_id": None}] + [
        {"id": f"o{row['occurrence_id']}",
         "parent_id": "root" if row['execution_parent_occurrence_id'] is None else f"o{row['execution_parent_occurrence_id']}",
         "action": row['action'], "scope_kind": row['scope_kind']} for row in path]}
    system = (
        "你处于Red。这里一次给出本次失败路径的所有结点及其公开快照，请比较所有位置后直接选择一个恢复点。"
        "不是逐格BACK，也不是逐个输出RESUME；控制器不替你选最深或最早的点。"
        "候选checkpoint_index表示执行该序号历史动作之前的状态；remaining_actions是恢复后可执行的真实动作数，不重置窗口。"
        "选择一个能在剩余步数内合理完成原任务的位置，兼顾已有证据、改变失败策略和完成所需步数。"
        "不要求保证满分或证明优于旧得分；未知未来结果可以写条件步骤，但不能把多个环境动作合成一步。"
        "只输出一个JSON对象，字段恰好checkpoint_index、reason、action_plan。"
        "checkpoint_index为所选整数；reason说明当前位置、还剩n步、为什么选择这里；action_plan是按顺序排列的行动字符串数组，"
        "数组每项恰好一个环境调用，必须把最后提交或完成任务的动作计入，数组长度不得超过n。"
        "第一项必须是该快照当前可调用的完整原生动作；后续未知目标可用条件描述，实际执行仍依赖之后的真实观察。"
        "不要把仅有一两步却需要搜索、打开、选择、提交多步的方案当作可完成方案。不要重放已经失败且无新依据的后缀。"
        "若所有位置均无可推荐的继续方案，输出checkpoint_index:null、action_plan:[]并说明原因。"
        "根结点也是可选候选，不存在向根之前BACK。快照只含公开证据，环境在审查时不会恢复，选定后只恢复一次。"
        "不输出未执行的观察、分数或虚构选项。公开快照通过引用去重，但没有删掉候选结点。")
    if phase is not None:
        if phase not in ('assess', 'choose'):
            raise ProtocolError('unknown Red typed protocol phase')
        system = (
            "你处于Red，所有结点和公开快照都在同一个输入里，环境尚未恢复。比较所有位置，"
            "选择能改变失败策略并保留足够完成步数的位置。不是逐格BACK，不优先最深点。"
            "remaining_actions是恢复后的真实步数预算，不会重置。计划中的每个环境调用和最终提交都计一步，"
            "不要把多个行动塞进一个步骤。第一步必须在所选快照可调用，未来未知信息可用条件描述。"
            "说明原失败原因、推荐位置、还剩n步、预计完成路线和不确定性，不要求保证满分。"
            "不捏造商品、观察或奖励。")
        if phase == 'assess':
            system += "本次只输出简洁自由文本分析，不输出JSON，不执行动作。随后将单独选择一个恢复点。"
        else:
            system += "参考分析并核对快照，只输出CHOICES中的一个完整选项。UNKNOWN表示你确实无法推荐任何恢复点。不要输出解释或JSON。"
            evidence['model_assessment'] = assessment
            evidence['CHOICES'] = list(choices)
    if feedback is not None:
        evidence['unexecuted_selection_feedback'] = feedback
        system += "上一次选择未执行，按明确错误重新选择一次；控制器不会替换索引或计划。"
    encoded = (json.dumps(evidence, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
               if episode.efficient_recovery else canonical(evidence))
    return episode.review_formatter([{"role": "system", "content": system},
        {"role": "user", "content": "[RED_JOINT_SELECTION]\n" + encoded}])
