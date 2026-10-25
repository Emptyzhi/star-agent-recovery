from dataclasses import asdict, dataclass
import math
from pathlib import Path


UPSTREAM_COMMIT = '6661046e2b30fcf5f17c93d51acb56babdea8c53'
DATASET_WINDOWS = {'game24': 12, 'alfworld': 50, 'webshop': 15}


@dataclass(frozen=True)
class Settings:
    dataset: str
    model: str
    backend: str
    initial_actions: int
    recovery_actions: int
    max_recoveries: int | None
    max_output_tokens: int
    actor_output_tokens: int
    task_timeout: float | None
    request_timeout: float
    response_grace_seconds: float
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    environment_seed: int
    candidate_limit: int
    memory_limit: int
    action_budget_policy: str = 'segment'
    history_format: str = 'flat'
    recovery_format: str = 'bracket-json'
    notebook_item_char_limit: int = 900
    rejected_history_keep: int = 0
    max_terminal_failures: int | None = None
    max_reflection_rejections: int | None = None
    prompt_format: str = 'native-chat'
    candidate_history: str = 'full'
    recovery_patience: int | None = None
    runtime_prompt: str = 'full'
    interface_format: str = 'native'
    recovery_prompt_policy: str = 'full'
    text_output_boundary: str = 'eos'
    text_history_format: str = 'sectioned'
    context_overflow: str = 'stop'

    def __post_init__(self):
        if self.context_overflow not in ('stop','head-tail'):
            raise ValueError('unknown context overflow policy')
        if self.context_overflow=='head-tail' and (
                self.backend!='resident' or self.prompt_format!='plain' or
                self.history_format!='dialogue' or self.text_history_format!='native'):
            raise ValueError('head-tail context requires resident native plain dialogue')
        if self.dataset not in DATASET_WINDOWS or self.backend not in ('resident', 'openai'):
            raise ValueError('unsupported dataset or model backend')
        if not self.model:
            raise ValueError('model must be explicit')
        if self.history_format not in ('flat','dialogue'):
            raise ValueError('unknown native history format')
        if self.prompt_format not in ('plain','native-chat'):
            raise ValueError('unknown model prompt format')
        if self.runtime_prompt not in ('full','compact'):
            raise ValueError('unknown runtime prompt presentation')
        if self.interface_format not in ('native','sectioned','json-tools'):
            raise ValueError('unknown interface format')
        if self.recovery_prompt_policy not in ('full','staged-compact'):
            raise ValueError('unknown recovery prompt policy')
        if self.text_output_boundary not in ('eos','command'):
            raise ValueError('unknown text output boundary')
        if self.text_history_format not in ('sectioned','native'):
            raise ValueError('unknown text history format')
        if self.text_history_format=='native' and (self.interface_format!='sectioned' or self.text_output_boundary!='command'):
            raise ValueError('native text history requires the sectioned command-boundary interface')
        if self.text_output_boundary=='command' and (self.interface_format!='sectioned' or self.backend!='resident'):
            raise ValueError('command boundary requires the sectioned resident text interface')
        if self.recovery_prompt_policy=='staged-compact' and (
                self.interface_format not in ('sectioned','json-tools') or self.notebook_item_char_limit!=900):
            raise ValueError('staged-compact requires a sectioned interface and the original 900-character notebook boundary')
        if self.interface_format in ('sectioned','json-tools') and (self.prompt_format!='plain' or
                self.history_format!='dialogue' or self.recovery_format!='lines' or self.runtime_prompt!='compact'):
            raise ValueError('sectioned interface requires plain dialogue, lines recovery and compact runtime')
        if self.runtime_prompt=='compact' and (self.prompt_format!='plain' or self.recovery_format!='lines'):
            raise ValueError('compact runtime requires plain text and lines recovery transport')
        if self.candidate_history not in ('full', 'once'):
            raise ValueError('unknown candidate history presentation')
        if self.candidate_history == 'once' and (self.prompt_format != 'plain' or self.history_format != 'dialogue'):
            raise ValueError('candidate references require plain recorded dialogue')
        if self.recovery_patience is not None:
            if type(self.recovery_patience) is not int or self.recovery_patience != 2:
                raise ValueError('recovery_patience must be 2 or None for the historical policy')
            if self.max_reflection_rejections is not None:
                raise ValueError('actor resumption replaces the historical reflection-rejection stop')
        if self.recovery_format not in ('bracket-json','python','lines'):
            raise ValueError('unknown recovery argument transport')
        if type(self.notebook_item_char_limit) is not int or self.notebook_item_char_limit not in (900, 4000):
            raise ValueError('notebook_item_char_limit must be 900 or 4000')
        if self.notebook_item_char_limit == 4000 and self.recovery_format != 'lines':
            raise ValueError('the 4000-character notebook protocol requires lines recovery transport')
        if type(self.rejected_history_keep) is not int or self.rejected_history_keep not in (0, 2):
            raise ValueError('rejected_history_keep must be 0 or 2')
        if self.rejected_history_keep and (self.history_format != 'dialogue' or self.recovery_format != 'lines'):
            raise ValueError('rejection compression requires literal dialogue and lines recovery transport')
        if self.action_budget_policy not in ('segment','trajectory'):
            raise ValueError('unknown native action budget policy')
        if self.action_budget_policy=='trajectory' and self.initial_actions!=self.recovery_actions:
            raise ValueError('trajectory policy requires the same initial and restored trajectory limit')
        for key in ('initial_actions', 'recovery_actions', 'max_output_tokens', 'actor_output_tokens', 'candidate_limit', 'memory_limit'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key+' must be a positive integer')
        if self.max_recoveries is not None and (type(self.max_recoveries) is not int or self.max_recoveries < 0):
            raise ValueError('max_recoveries must be a nonnegative integer or None (unlimited)')
        for key in ('max_terminal_failures', 'max_reflection_rejections'):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(key+' must be a positive integer or None')
        for key in ('environment_seed',):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(key+' must be a nonnegative integer')
        if self.task_timeout is not None and (not math.isfinite(self.task_timeout) or self.task_timeout<=0):
            raise ValueError('task_timeout must be positive and finite or None')
        for key in ('request_timeout', 'response_grace_seconds', 'temperature', 'top_p', 'repetition_penalty'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(key+' must be positive and finite')
        if self.top_p > 1 or (self.top_k != -1 and self.top_k < 1):
            raise ValueError('invalid sampling parameters')
        if self.actor_output_tokens > self.max_output_tokens:
            raise ValueError('actor output limit exceeds configured generation capacity')

    def as_dict(self):
        return dict(asdict(self), model_request_limit=None, upstream_commit=UPSTREAM_COMMIT,
                    environment_actions=('current retained trajectory includes every native execute attempt; selected checkpoint restores its exact depth; cumulative execution and model costs never decrease'
                        if self.action_budget_policy=='trajectory' else 'per-segment native actions; cumulative consumption never decreases'),
                    recovery_trigger='unsuccessful native terminal requires recovery and author candidate listing; agent selects checkpoint and memory; voluntary backtrack otherwise; exhausted windows allow recovery tools only',
                    adaptation_protocol=('native_act_only_v44_game24_initial_separator_1200'
                        if self.dataset=='game24' and self.prompt_format=='plain' and
                        self.interface_format=='sectioned' and self.text_history_format=='native' else
                        'native_act_only_v42_context_headtail_1200' if self.context_overflow=='head-tail' else
                        'native_act_only_v41_native_action_envelope_1200' if self.text_history_format=='native' else
                        'native_act_only_v36_text_command_boundary_1200' if self.text_output_boundary=='command' else
                        'native_act_only_v35_staged_text_tools_1200' if self.recovery_prompt_policy=='staged-compact' and self.interface_format=='sectioned' else
                        'native_act_only_v34_staged_compact_memory_1200' if self.recovery_prompt_policy=='staged-compact' else
                        'native_act_only_v33_game24_native_observation_1200' if self.interface_format=='json-tools' else
                        'native_act_only_v27_sectioned_runtime_1200' if self.interface_format=='sectioned' else
                        'native_act_only_v25_compact_runtime_1200' if self.runtime_prompt=='compact' else
                        'native_act_only_v24_bounded_recovery_actor_resume_1200' if self.recovery_patience is not None else
                        'native_act_only_v23_candidate_references_1200' if self.candidate_history=='once' else
                        'native_act_only_v22_plain_continuation_1200' if self.prompt_format=='plain' else
                        'native_act_only_v21_native_failure_stop_1200'
                        if self.max_terminal_failures is not None or self.max_reflection_rejections is not None
                        else 'native_act_only_v20_author_tools_1200'),
                    stopping_policy='native success; configured terminal/reflection rejection stops; two-no-progress actor resumption if configured, terminal if no native action can run; finite recovery allowance if configured; task/request deadlines and context capacity; no cumulative request limit',
                    repeated_failure_protocol=dict(terminal='count actual done=true receipts with reward below one; action-window exhaustion is not a terminal submission',
                        reflection=('count rejected literal commit submissions or author commit errors; reset only on environment-accepted action or successful restore; rejection count is diagnostic, not an independent stop'
                            if self.recovery_patience is not None else 'count rejected literal commit submissions or author commit errors; reset on any executed environment action or successful restore; candidate listing does not reset'),
                        authority='explicit benchmark adaptation, not the original ordered-checklist rule'),
                    candidate_tool_policy=('two no-progress recovery decisions pause both recovery tools until an environment-accepted native action; otherwise native author tools stay available'
                        if self.recovery_patience is not None else 'author prompt requests commit-only; adapter does not hard-disable otherwise legal native actions or candidate listing'),
                    recovery_progress_policy=dict(limit=self.recovery_patience,
                        counted='recovery-phase replies and literal recovery-tool proposals without actual restore or environment-accepted action; first successful necessary candidate listing exempt',
                        on_limit='pause recovery and require an environment-accepted native action before reopening; end task if terminal or action budget exhausted; no restore or budget refresh',
                        reset='actual restore or environment-accepted native action only',
                        authority='explicit user-authorized adaptation; overrides author commit-only prose while recovery is paused'),
                    recovery_response_validation='candidate requests consume the full response; commit body passed verbatim as one memory argument; native actions in recovery must be one complete line',
                    candidate_history_projection='latest full list plus exact entry references in earlier receipts; distinct historical entries retained; logical checkpoint messages and original logs unchanged',
                    actor_transport=('phase-scoped recovery guidance, original notebook and native action/observation transcript; direct text continuation; full author guidance and concise correction-first memory instruction in recovery'
                        if self.recovery_prompt_policy=='staged-compact' else
                        'literal author guidance, notebook and native action/observation transcript; direct text continuation for both action and recovery; no chat template'
                        if self.prompt_format=='plain' else
                        'unchanged native Act-only demonstrations and action/observation continuation in native model chat; plain continuation only for explicit zero-recovery control'),
                    model_feedback='native public observation; Game24 public arithmetic state; ALFWorld admissible commands; full receipts audit-only',
                    runtime_prompt_policy=('staged-compact: brief action-stage recovery rules; full literal author guidance during recovery; all 15 memory fields with executable correction first and a 900-character writing target; no memory repair'
                        if self.recovery_prompt_policy=='staged-compact' else
                        'compact: unchanged author guidance; one transcript observation, status/structured state only in receipt, short budget/command instructions, no repeated serialization example; full: historical verbose adapter runtime'),
                    invalid_tool_policy='explicit rejection feedback; no execution, repair, free retry or added recovery; every generation charged',
                    resident_output_boundary=('token-time complete candidate command or combined commit with final delimiter; native actions first_line; sampled text/tokens retained verbatim; no output repair'
                        if self.text_output_boundary=='command' else
                        'complete JSON tool object while recovery tools are enabled; literal first-line native action during Actor-only phase; Actor cap for actions/queries; reflection cap for combined commit; no output repair'
                        if self.interface_format=='json-tools' else 'first_line and Actor output cap for native actions and candidate-only queries; native EOS and reflection cap for combined commits with explicit final delimiter'
                        if self.interface_format=='sectioned' else 'first_line for native actions and voluntary candidate requests; native EOS for recovery phase including combined commit'),
                    output_cap_policy='literal bounded completion enters the native parser or author memory validator; token limit alone does not end a task; no completion repair',
                    context_exhaustion=('overflow-only deterministic head-tail view; keep original task, current controller, notebook, latest observation and candidate IDs; no summarizer calls; original logical history and checkpoints unchanged'
                        if self.context_overflow=='head-tail' else
                        'terminal before submission; only the explicitly configured rejected-history projection; no general prompt truncation or phantom model call'),
                    rejected_history_projection=dict(keep_complete=self.rejected_history_keep,
                        scope='one pending candidate-selection phase; earlier rejected proposals grouped by tool, target, exact error and source references; original records retained'),
                    request_deadline_policy=('resident server generation budget; separate HTTP receipt grace; no retry'
                        if self.backend == 'resident' else 'OpenAI HTTP timeout; resident receipt grace is not used'),
                    tool_definitions='current request only; native action syntax; author commit schema with record_uid and memory_summary submitted together',
                    notebook_limits=dict(items=self.memory_limit,item_characters=self.notebook_item_char_limit,
                                         total_characters=8000))


def resolve(args):
    if args.dataset not in DATASET_WINDOWS:
        raise ValueError('unknown dataset')
    window = DATASET_WINDOWS[args.dataset]
    settings = Settings(dataset=args.dataset, model=args.model, backend=args.backend,
        initial_actions=window if args.initial_actions is None else args.initial_actions,
        recovery_actions=window if args.recovery_actions is None else args.recovery_actions,
        max_recoveries=args.max_recoveries, max_output_tokens=args.max_output_tokens,
        actor_output_tokens=args.actor_output_tokens,
        task_timeout=args.task_timeout, request_timeout=args.request_timeout,
        response_grace_seconds=args.response_grace_seconds,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, environment_seed=args.environment_seed,
        candidate_limit=args.candidate_limit, memory_limit=args.memory_limit,
        action_budget_policy=args.action_budget_policy,history_format=args.history_format,
        recovery_format=args.recovery_format,notebook_item_char_limit=args.notebook_item_char_limit,
        rejected_history_keep=args.rejected_history_keep,
        max_terminal_failures=args.max_terminal_failures,
        max_reflection_rejections=args.max_reflection_rejections,prompt_format=args.prompt_format,
        candidate_history=args.candidate_history,recovery_patience=args.recovery_patience,
        runtime_prompt=args.runtime_prompt,interface_format=args.interface_format,
        recovery_prompt_policy=args.recovery_prompt_policy,text_output_boundary=args.text_output_boundary,
        text_history_format=args.text_history_format,context_overflow=args.context_overflow)
    tasks = [int(value) for value in args.task_ids.split(',')]
    if not tasks or len(set(tasks)) != len(tasks) or min(tasks) < 0:
        raise ValueError('task IDs must be explicit unique nonnegative indices')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if not 0 < args.prompt_limit < args.context_limit or args.max_output_tokens + args.prompt_limit > args.context_limit:
        raise ValueError('invalid context/prompt/output capacity')
    if args.backend == 'openai' and (args.context_limit != 32768 or args.prompt_limit != 31000):
        raise ValueError('context-limit/prompt-limit are resident tokenizer checks; configure API context at its server')
    if args.dataset != 'alfworld' and args.split != 'test':
        raise ValueError('--split applies only to ALFWorld')
    required = {'game24': 'csv', 'alfworld': 'manifest', 'webshop': 'website'}[args.dataset]
    if getattr(args, required) is None:
        raise ValueError('--'+required+' is required for '+args.dataset)
    applicable = {'game24': {'csv'}, 'alfworld': {'manifest', 'environment_config'}, 'webshop': {'website'}}[args.dataset]
    for name in ('csv', 'manifest', 'environment_config', 'website'):
        if getattr(args, name) is not None and name not in applicable:
            raise ValueError(name+' does not apply to '+args.dataset)
    if args.dataset == 'webshop' and max(tasks) >= 500:
        raise ValueError('WebShop task IDs must be in 0..499')
    if not (args.native_adapter_root/'source').is_dir():
        raise ValueError('native adapter root must contain the existing STAR environment source')
    if args.backend == 'resident':
        if not all((args.parent_job_id, args.node, args.step_id)) or args.gpu_index is None:
            raise ValueError('resident requires parent job, node, step and GPU index')
    return settings, tasks
