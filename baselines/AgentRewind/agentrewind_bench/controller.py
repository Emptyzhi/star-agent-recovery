from copy import deepcopy
import json
from pathlib import Path
import time

from . import author
from .models import ModelContextLimit, ModelDeadlineExpired
from .native_protocol import (RUNTIME_PREFIX, boundary_instruction,
    decode_line, initial_actor_prompt, append_actor_step, add_control_observation,
    without_runtime_allowance, available_command_instruction, unavailable_command_error,append_dialogue_feedback)
from .feedback import model_feedback
from .notebook import notebook_override
from .rejection_context import RejectedHistory
from .runtime_prompt import PROTOCOL_INSTRUCTION,compact_runtime_instruction
from .framing import sectioned_initial_prompt,decode_sectioned_reply,decode_native_action_reply,COMMIT_END,one_line_recovery
from .json_tools import decode_interface_reply,json_runtime_instruction,actor_only_request,declared_tool_name


class ProtocolError(ValueError):
    pass


class GenerationStopped(RuntimeError):
    def __init__(self, status, evidence):
        self.status, self.evidence = status, evidence


class Controller:
    def __init__(self, environment, model, settings, output, run_id):
        self.env, self.model, self.settings = environment, model, settings
        self.output, self.run_id = Path(output), run_id
        self.log_dir = self.output/'author_traces'
        self.log_dir.mkdir(exist_ok=False)
        self.trace = run_id+'_r0'
        self.records = []
        self.checkpoints = {}
        self.checkpoint_depths = {}
        self.record_index = 0
        self.recoveries = self.actions = self.window_actions = self.calls = 0
        self.trajectory_actions = 0
        self.input_tokens = self.output_tokens = 0
        self.queue_seconds = 0.0
        self.queue_known = True
        self.usage_complete = True
        self.purchases = []
        self.stages = []
        self.started = time.monotonic()
        self.writer = author.JsonlWriter(author.run_path(self.log_dir, self.trace), overwrite=False)
        self.events = (self.output/'events.jsonl').open('x', encoding='utf-8', buffering=1)
        self.last_candidates = set()
        self.last_candidate_uid = None
        self.terminal_failures = 0
        self.reflection_rejections = 0
        self.protocol_rejections = []
        self.dialogue = settings.history_format=='dialogue' and settings.max_recoveries!=0
        self.rejected_history = RejectedHistory(settings.rejected_history_keep)
        self.last_llm_uid = None
        self.recovery_no_progress = 0
        self.actor_until_accepted = False
        self.initial_listing_exempted = False

    def reset_recovery_progress(self, reason):
        if self.settings.recovery_patience is None:
            return
        previous = self.recovery_no_progress
        paused = self.actor_until_accepted
        self.recovery_no_progress = 0
        self.actor_until_accepted = False
        self.initial_listing_exempted = False
        self.emit('recovery_progress_reset', reason=reason, previous_no_progress=previous,
            was_paused=paused, cost=self.cost())

    def recovery_without_progress(self, reason, *, initial_listing=False):
        if self.settings.recovery_patience is None or self.actor_until_accepted:
            return
        if initial_listing and not self.initial_listing_exempted:
            self.initial_listing_exempted = True
            self.emit('recovery_initial_listing_exempt', cost=self.cost())
            return
        self.recovery_no_progress += 1
        self.actor_until_accepted = self.recovery_no_progress >= self.settings.recovery_patience
        self.emit('recovery_decision_without_progress', reason=reason,
            consecutive=self.recovery_no_progress, limit=self.settings.recovery_patience,
            recovery_paused=self.actor_until_accepted, cost=self.cost())

    def emit(self, event, **data):
        self.events.write(json.dumps(dict(event=event, trace=self.trace,time=time.time(), recovery_count=self.recoveries, **data),
                                     ensure_ascii=False, allow_nan=False)+'\n')

    def cost(self):
        elapsed = time.monotonic()-self.started
        queue = self.queue_seconds if self.queue_known else None
        return dict(model_requests=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    elapsed_seconds=elapsed, model_queue_seconds=queue,
                    queue_excluded_elapsed_seconds=None if queue is None else elapsed-queue,
                    model_request_limit=None, token_accounting_complete=self.usage_complete,
                    queue_time_accounting_complete=self.queue_known)

    def finish_stage(self, reason):
        stage = dict(recovery_count=self.recoveries, executed_actions=self.actions,
                     window_actions=self.window_actions, trajectory_actions=self.trajectory_actions,
                     best_purchase_reward=max([0.0, *self.purchases]),
                     reason=reason, cost=self.cost())
        self.stages.append(stage)
        self.emit('recovery_stage_finished', **{k:v for k,v in stage.items() if k!='recovery_count'})

    def request_submitted(self):
        self.calls += 1
        self.usage_complete = False
        self.emit('model_request_submitted',cost=self.cost())

    def record(self, kind, **data):
        self.record_index += 1
        row = dict(kind=kind, record_uid=author.make_record_uid(self.record_index), **data)
        self.records.append(row)
        self.writer.write(row)
        return row['record_uid']

    def rewind(self, request):
        if self.settings.max_recoveries is not None and self.recoveries >= self.settings.max_recoveries:
            raise ProtocolError('backtrack exceeded configured recovery allowance')
        fork = self.run_id+'_r'+str(self.recoveries+1)
        prepared = author.prepare_backtrack_replay(request, current_run_id=self.trace, log_dir=self.log_dir,
            fork_run=fork, memory_limit=self.settings.memory_limit, memory_policy='cumulative',
            context_policy='rollback', workspace_policy='rollback')
        effective_override, notebook_config = notebook_override(
            prepared, self.log_dir, item_char_limit=self.settings.notebook_item_char_limit,
            memory_limit=self.settings.memory_limit)
        self.finish_stage('AGENT_BACKTRACK')
        messages = json.loads(effective_override.read_text(encoding='utf-8'))['messages']
        snapshot = self.checkpoints[request.record_uid]
        previous_depth = self.trajectory_actions
        restored_depth = self.checkpoint_depths[request.record_uid]
        restoration = self.env.restore(snapshot)
        index = next(i for i,r in enumerate(self.records) if r['record_uid'] == request.record_uid)
        prefix = self.records[:index]
        self.writer.close()
        self.writer = author.JsonlWriter(author.run_path(self.log_dir, fork), overwrite=False)
        for record in prefix:
            self.writer.write(record)
        self.records = prefix
        # Upstream ReplaySession.on_enter seeds each materialized fork's counter
        # from its retained prefix, not from the abandoned suffix's final UID.
        self.record_index = max((int(r['record_uid'].removeprefix('rec_')) for r in prefix),default=0)
        retained = {r['record_uid'] for r in prefix if r['kind']=='llm'}
        self.checkpoints = {uid:s for uid,s in self.checkpoints.items() if uid in retained}
        self.checkpoint_depths = {uid:d for uid,d in self.checkpoint_depths.items() if uid in retained}
        self.trace = fork
        self.recoveries += 1
        self.window_actions = 0
        self.trajectory_actions = restored_depth
        self.last_candidates = set()
        self.last_candidate_uid = None
        self.reflection_rejections = 0
        self.rejected_history.clear()
        remaining = (self.settings.recovery_actions-restored_depth
                     if self.settings.action_budget_policy=='trajectory' else self.settings.recovery_actions)
        if remaining < 0:
            raise ProtocolError('checkpoint depth exceeds the configured trajectory limit')
        if remaining:
            self.env.set_window(remaining)
        self.emit('checkpoint_restored', target=request.record_uid, restoration=restoration,
                  previous_trajectory_actions=previous_depth,trajectory_actions=restored_depth,
                  native_actions_remaining=remaining,
                  context_source=prepared['backtrack_context_source_record_uid'],
                  original_override_input_file=str(prepared['override_input_file']),
                  effective_override_input_file=str(effective_override),
                  notebook_render_config=notebook_config,
                  fork_record_counter=self.record_index,cost=self.cost())
        self.reset_recovery_progress('checkpoint_restored')
        # Restore the selected request and its action depth. Costs stay cumulative.
        return without_runtime_allowance(messages), self.env.observe()

    def generate(self, messages, purpose, tools):
        single_line=purpose=='action' or (self.settings.interface_format=='sectioned' and one_line_recovery(messages)) or (
            self.settings.interface_format=='json-tools' and not any(t['name']=='backtrack_commit' for t in tools))
        output_limit = (self.settings.actor_output_tokens if single_line
                        else self.settings.max_output_tokens)
        self.emit('model_input_prepared', messages=messages, purpose=purpose, tools=tools,
                  trajectory_actions=self.trajectory_actions,
                  recovery_policy_state=dict(no_progress=self.recovery_no_progress,
                    paused=self.actor_until_accepted,initial_listing_exempted=self.initial_listing_exempted),
                  output_limit=output_limit)
        snapshot = self.env.checkpoint()
        remaining = (self.settings.request_timeout if self.settings.task_timeout is None else
                     self.settings.task_timeout-(time.monotonic()-self.started))
        if remaining <= 0:
            raise GenerationStopped('TASK_TIMEOUT', dict(submitted=False,error='task deadline'))
        try:
            reply = self.model.generate(messages,output_limit,
                timeout=min(self.settings.request_timeout,remaining),on_submit=self.request_submitted,
                purpose=purpose)
        except (ModelContextLimit,ModelDeadlineExpired) as error:
            context = isinstance(error,ModelContextLimit)
            evidence = dict(error_type=type(error).__name__,error=str(error),submitted=False)
            self.emit('model_context_limit' if context else 'model_time_limit',**evidence,cost=self.cost())
            raise GenerationStopped('MODEL_CONTEXT_LIMIT' if context else 'MODEL_TIME_LIMIT',evidence) from error
        self.input_tokens += reply.input_tokens
        self.output_tokens += reply.output_tokens
        self.usage_complete = True
        if reply.queue_seconds is None:
            self.queue_known = False
        else:
            self.queue_seconds += reply.queue_seconds
        self.emit('model_request_completed',text=reply.text,raw_response=reply.raw,purpose=purpose,
                  input_tokens=reply.input_tokens,output_tokens=reply.output_tokens,cost=self.cost())
        if 'client_context_compression' in reply.raw:
            self.emit('model_context_compressed',**reply.raw['client_context_compression'],cost=self.cost())
        request = dict(messages=deepcopy(messages),model=self.settings.model,max_tokens=output_limit,
                       temperature=self.settings.temperature,top_p=self.settings.top_p,
                       top_k=self.settings.top_k,repetition_penalty=self.settings.repetition_penalty)
        if self.settings.backend == 'resident':
            request['stop_mode'] = ('agentrewind_text_command' if self.settings.text_output_boundary=='command' else
                ('first_line' if actor_only_request(messages) else 'json_object') if self.settings.interface_format=='json-tools' else
                'first_line' if single_line else None)
        else:
            request['stop'] = ['\n'] if single_line else None
        if 'client_prompt' in reply.raw:
            request['prompt'] = reply.raw['client_prompt']
            request['model_messages'] = deepcopy(reply.raw['client_model_messages'])
        uid = self.record('llm',input=request,
            output={'content':reply.text,'usage':dict(input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens),'stop_reason':reply.stop_reason},
            adapter=dict(purpose=purpose,available_text_commands=tools))
        self.last_llm_uid = uid
        self.checkpoints[uid] = snapshot
        self.checkpoint_depths[uid] = self.trajectory_actions
        self.emit('model_request_checkpointed',record_uid=uid,purpose=purpose,
                  trajectory_actions=self.trajectory_actions)
        if reply.stop_reason in ('length','eos_at_limit'):
            # Native benchmark callers parse the literal bounded completion.
            # A token boundary is not a task deadline or an inferred bad action.
            self.emit('model_output_boundary',purpose=purpose,stop_reason=reply.stop_reason,
                      output_limit=output_limit,cost=self.cost())
        if reply.stop_reason == 'time':
            status = 'MODEL_TIME_LIMIT'
            if self.settings.task_timeout is not None and time.monotonic()-self.started >= self.settings.task_timeout:
                status = 'TASK_TIMEOUT'
            evidence = dict(submitted=True,error='generation stopped at '+reply.stop_reason)
            self.emit('model_resource_limit',status=status,**evidence,cost=self.cost())
            raise GenerationStopped(status,evidence)
        return reply

    def reject(self,messages,raw,error):
        evidence = dict(error=str(error),raw_text=raw)
        self.protocol_rejections.append(evidence)
        self.emit('model_protocol_rejected',**evidence,cost=self.cost())
        if self.dialogue:
            return append_dialogue_feedback(messages,raw,'Protocol error: '+str(error)+
                '. No environment action or recovery was executed. Follow the current allowance.',
                self.settings.dataset,self.trajectory_actions+1)
        # A tool error is a public observation, not a silently repaired action.
        return add_control_observation(messages, 'Rejected model output: '+raw+
            '\nProtocol error: '+str(error)+
            '. No environment action or recovery was executed. Follow the current allowance.')

    def begin_rejected_history(self, messages, candidate_uid, continuation_uid=None):
        report = self.rejected_history.begin(messages, trace=self.trace, candidate_record_uid=candidate_uid,
            continuation_record_uid=continuation_uid)
        if report is not None:
            self.emit('rejected_history_phase_started', **report)

    def finish_rejected_history(self, messages, reason):
        if self.rejected_history.base is None:
            return messages
        projected, report = self.rejected_history.project(messages)
        self.emit('rejected_history_phase_finished', stage=self.rejected_history.stage,
            reason=reason, projection=report)
        self.rejected_history.clear()
        return projected

    def reject_reflection(self, error, origin):
        self.reflection_rejections += 1
        self.emit('reflection_submission_rejected', error=str(error), origin=origin,
            consecutive_rejections=self.reflection_rejections,
            limit=self.settings.max_reflection_rejections, cost=self.cost())

    def register_rejected_history(self, messages, raw, tool, target, error):
        if self.settings.rejected_history_keep and self.last_candidates:
            report = self.rejected_history.register(messages, trace=self.trace,
                record_uid=self.last_llm_uid, raw_reply=raw, tool=tool, target=target,
                error_type=type(error).__name__, error=str(error))
            self.emit('rejected_history_registered', **report)

    def run(self):
        try:
            model_resource_error = None
            current = self.env.reset()
            self.env.set_window(self.settings.initial_actions)
            messages = ([{'role':'system', 'content':author.instruction(self.settings.recovery_format,
                commit_boundary=COMMIT_END if self.settings.interface_format=='sectioned' else None,
                json_tools=self.settings.interface_format=='json-tools')}]
                        if self.settings.max_recoveries != 0 else [])
            if self.settings.max_recoveries!=0 and self.settings.runtime_prompt=='compact':
                messages.append(dict(role='system',content=PROTOCOL_INSTRUCTION))
            initial_prompt=(sectioned_initial_prompt if self.settings.interface_format in ('sectioned','json-tools') else initial_actor_prompt)
            messages.append({'role':'user','content':initial_prompt(
                self.settings.dataset,self.env.initial_prompt(),current)})
            try:
                while True:
                    if self.settings.task_timeout is not None and time.monotonic()-self.started >= self.settings.task_timeout:
                        status = 'TASK_TIMEOUT'
                        break
                    if current['done'] and current['reward'] == 1:
                        status = 'SUCCESS'
                        break
                    if (self.settings.max_terminal_failures is not None and
                            self.terminal_failures >= self.settings.max_terminal_failures):
                        status = 'TERMINAL_FAILURE_LIMIT_REACHED'
                        break
                    if (self.settings.max_reflection_rejections is not None and
                            self.reflection_rejections >= self.settings.max_reflection_rejections):
                        status = 'REFLECTION_REJECTION_LIMIT_REACHED'
                        break
                    window = self.settings.initial_actions if self.recoveries == 0 else self.settings.recovery_actions
                    consumed = (self.trajectory_actions if self.settings.action_budget_policy=='trajectory'
                                else self.window_actions)
                    blocked = current['done'] or consumed >= window
                    if self.actor_until_accepted:
                        messages = self.finish_rejected_history(messages, 'recovery_paused')
                        self.last_candidates = set()
                        self.last_candidate_uid = None
                        if blocked:
                            status = 'RECOVERY_STALLED_NO_ACTION'
                            self.emit('recovery_cannot_resume_actor', done=current['done'],
                                native_actions_remaining=max(0,window-consumed),cost=self.cost())
                            break
                    remaining_recoveries = (None if self.settings.max_recoveries is None
                                            else self.settings.max_recoveries-self.recoveries)
                    recovery_available = remaining_recoveries != 0 and not self.actor_until_accepted
                    if blocked and remaining_recoveries == 0:
                        status = 'RECOVERY_LIMIT_REACHED' if current['done'] else 'ACTION_LIMIT_REACHED'
                        break
                    if current['done'] and current['reward'] != 1 and not self.last_candidates:
                        # The benchmark harness requests recovery after an unsuccessful
                        # terminal. The agent still selects the author's checkpoint.
                        if current['reward'] is None:
                            raise ProtocolError('terminal environment receipt has no reward')
                        reason='Native terminal attempt has reward '+str(current['reward'])+' below full success 1.'
                        candidates=author.backtrack_candidates(reason,run_id=self.trace,
                            log_dir=self.log_dir,mode='deterministic',limit=self.settings.candidate_limit)
                        self.last_candidates={row['record_uid'] for row in candidates}
                        if not self.last_candidates:
                            raise ProtocolError('unsuccessful terminal has no author recovery candidates')
                        self.emit('terminal_recovery_required',reward=current['reward'],
                            action_accepted=current['accepted'],candidates=candidates,cost=self.cost())
                        candidate_uid=self.record('tool',input={'tool_name':'backtrack_candidates','arguments':{'reason':reason},
                            'invoked_by':'benchmark_terminal_protocol'},output={'value':candidates})
                        self.last_candidate_uid = candidate_uid
                        feedback=json.dumps(dict(recovery_required=True,reason=reason,
                            tool='backtrack_candidates',result=candidates),ensure_ascii=False)
                        messages=(append_dialogue_feedback(messages,None,feedback,self.settings.dataset,
                            self.trajectory_actions+1) if self.dialogue else add_control_observation(messages,feedback))
                        self.begin_rejected_history(messages, candidate_uid)
                    allowed = [tool for tool in author.tools() if
                               (tool['name']=='execute_action' and not blocked) or
                               (tool['name']=='backtrack_candidates' and recovery_available) or
                               (tool['name']=='backtrack_commit' and recovery_available and self.last_candidates)]
                    boundary = boundary_instruction(remaining=max(0,window-consumed),
                        recoveries=remaining_recoveries,blocked=blocked,candidates=bool(self.last_candidates),
                        recovery_format=self.settings.recovery_format)
                    if self.actor_until_accepted:
                        boundary = (f'Native actions remaining: {max(0,window-consumed)}. '
                            'The current recovery phase ended after two decisions without a checkpoint '
                            'restore or an environment-accepted native action. Recovery tools are paused. '
                            'This current controller mode overrides earlier commit-only instructions. '
                            'Continue from the actual current observation with one native task action. '
                            'Do not call either recovery tool. Recovery reopens only after the environment '
                            'accepts a native action. Rejected environment actions consume the existing '
                            'native action allowance; no checkpoint was restored and no budget was refreshed.')
                    if self.settings.action_budget_policy=='trajectory':
                        boundary += (f' Current retained native trajectory length: {self.trajectory_actions}; '
                            f'trajectory limit: {window}. Restoring a checkpoint retains its prefix length; '
                            'only discarded suffix actions cease to occupy this trajectory allowance. '
                            'Recovery commands and memory do not occupy native action positions.')
                    payload, projection = self.rejected_history.project(messages)
                    if projection is not None:
                        self.emit('rejected_history_projected', **projection)
                    if self.settings.max_recoveries != 0:
                        index = next(i for i, message in enumerate(payload) if message['role']=='user')
                        if self.settings.runtime_prompt=='compact':
                            runtime_function=json_runtime_instruction if self.settings.interface_format=='json-tools' else compact_runtime_instruction
                            runtime=runtime_function(tools=allowed,remaining=max(0,window-consumed),
                                recoveries=remaining_recoveries,blocked=blocked,candidates=bool(self.last_candidates),
                                paused=self.actor_until_accepted,public_receipt=model_feedback(current,self.settings.dataset),
                                output_limit=self.settings.max_output_tokens,
                                commit_end=COMMIT_END if self.settings.interface_format=='sectioned' else None)
                        else:
                            runtime=(RUNTIME_PREFIX+boundary+
                            ('\nOutput phase: recovery-command.' if blocked or self.last_candidates else
                             '\nOutput phase: native-action-or-rewind-command.')+
                            '\n'+available_command_instruction(allowed,self.settings.max_output_tokens,self.settings.recovery_format)+
                            '\nCurrent observation: '+json.dumps(model_feedback(current,self.settings.dataset),ensure_ascii=False))
                        payload.insert(index,{'role':'system','content':runtime})
                    had_candidates = bool(self.last_candidates)
                    was_paused = self.actor_until_accepted
                    purpose = 'recovery' if blocked or self.last_candidates else 'action'
                    reply = self.generate(payload,purpose,allowed)
                    proposed_tool=(declared_tool_name(reply.text) if self.settings.interface_format=='json-tools' else None)
                    recovery_decision = not was_paused and (purpose=='recovery' or
                        reply.text.lstrip().startswith(('backtrack_candidates','backtrack_commit')) or
                        proposed_tool in ('backtrack_candidates','backtrack_commit'))
                    try:
                        if self.settings.interface_format=='json-tools':
                            name,argument=decode_interface_reply(reply.text,payload)
                            recovery_decision=not was_paused and (purpose=='recovery' or name in ('backtrack_candidates','backtrack_commit'))
                        elif self.settings.interface_format=='sectioned':
                            if purpose=='action' and self.settings.text_history_format=='native':
                                name,argument=decode_native_action_reply(reply.text)
                                self.emit('native_action_frame_decoded',
                                    framing='first_nonempty_line',
                                    leading_whitespace_characters=len(reply.text)-len(reply.text.lstrip(' \t\r\n')),
                                    tool=name,argument=argument,cost=self.cost())
                            else:
                                name,argument=decode_sectioned_reply(reply.text,whole_response=purpose=='recovery')
                        else:
                            name, argument = decode_line(reply.text,self.settings.recovery_format,
                                whole_response=purpose=='recovery')
                        if name not in {tool['name'] for tool in allowed}:
                            if self.actor_until_accepted:
                                raise ProtocolError('Recovery is paused. Return one native environment action only; '
                                    'recovery tools reopen after the environment accepts a native action.')
                            raise ProtocolError(unavailable_command_error(name,
                                recoveries=remaining_recoveries,blocked=blocked,
                                candidates=bool(self.last_candidates),recovery_format=self.settings.recovery_format))
                    except (ValueError,ProtocolError) as error:
                        messages = self.reject(messages,reply.text,error)
                        self.register_rejected_history(messages,reply.text,None,None,error)
                        if reply.text.lstrip().startswith('backtrack_commit') or proposed_tool=='backtrack_commit':
                            self.reject_reflection(error, 'transport')
                        if recovery_decision:
                            self.recovery_without_progress('protocol_rejected')
                        continue
                    arguments = (argument if name=='backtrack_commit' else
                        {dict(execute_action='action',backtrack_candidates='reason')[name]:argument})
                    tool_error = None
                    author_error = None
                    if name=='execute_action':
                        current = self.env.execute(argument)
                        messages = self.finish_rejected_history(messages, 'native_action_executed')
                        if self.settings.recovery_patience is None or current['accepted']:
                            self.reflection_rejections = 0
                        self.actions += 1
                        self.window_actions += 1
                        self.trajectory_actions += 1
                        if current['done']:
                            if current['reward'] is None:
                                raise ProtocolError('terminal environment receipt has no reward')
                            self.purchases.append(current['reward'])
                            if current['reward'] != 1:
                                self.terminal_failures += 1
                            self.emit('native_terminal_evaluated', reward=current['reward'],
                                success=current['reward']==1, failed_submissions=self.terminal_failures,
                                limit=self.settings.max_terminal_failures, cost=self.cost())
                        result = model_feedback(current,self.settings.dataset)
                        self.emit('native_action_executed',action=argument,receipt=current,
                                  trajectory_actions=self.trajectory_actions)
                        if current['accepted']:
                            self.reset_recovery_progress('native_action_accepted')
                        if self.dialogue:
                            messages=append_dialogue_feedback(messages,reply.text,current['observation'],
                                self.settings.dataset,self.trajectory_actions+1)
                        else:
                            transcript = next(message for message in messages if message['role']=='user')
                            executed = current['executed_action'] if self.settings.dataset in ('webshop','alfworld') else argument
                            transcript['content'] = append_actor_step(self.settings.dataset,
                                transcript['content'],executed,current['observation'])
                    elif name=='backtrack_candidates':
                        result = author.backtrack_candidates(argument,run_id=self.trace,
                            log_dir=self.log_dir,mode='deterministic',limit=self.settings.candidate_limit)
                        messages = self.finish_rejected_history(messages, 'candidate_list_refreshed')
                        self.last_candidates = {r['record_uid'] for r in result}
                    else:
                        self.emit('recovery_memory_generated',target=arguments['record_uid'],
                                  text=arguments['memory_summary'],combined_commit=True,
                                  stop_reason=reply.stop_reason,cost=self.cost())
                        try:
                            try:
                                author.backtrack_commit(**arguments,run_id=self.trace,log_dir=self.log_dir)
                            except author.BacktrackRequested as request:
                                messages,current = self.rewind(request)
                            else:
                                raise ProtocolError('upstream commit did not request a rewind')
                        except (author.BacktrackError,author.InvalidBreakpointError) as error:
                            author_error = error
                            tool_error = author.tool_error_to_record(error)
                            result = dict(rejected=True,error_type=type(error).__name__,error=str(error))
                            self.emit('author_tool_rejected',tool=name,arguments=arguments,**result,cost=self.cost())
                            self.reject_reflection(error, 'author')
                        else:
                            continue
                    tool_uid=self.record('tool',input={'tool_name':name,'arguments':arguments},
                                output={'value':result} if tool_error is None else None,error=tool_error)
                    if name=='execute_action' and self.last_candidates:
                        self.begin_rejected_history(messages, self.last_candidate_uid, tool_uid)
                    if name != 'execute_action':
                        receipt=(dict(tool=name,result=result) if self.dialogue else
                                 dict(tool=name,arguments=arguments,result=result))
                        feedback=json.dumps(receipt,ensure_ascii=False)
                        messages=(append_dialogue_feedback(messages,reply.text,feedback,self.settings.dataset,
                            self.trajectory_actions+1) if self.dialogue else add_control_observation(messages,feedback))
                        if name=='backtrack_candidates' and self.last_candidates:
                            self.last_candidate_uid = tool_uid
                            self.begin_rejected_history(messages, tool_uid)
                        elif author_error is not None:
                            self.register_rejected_history(messages,reply.text,name,arguments['record_uid'],author_error)
                    if recovery_decision and not (name=='execute_action' and current['accepted']):
                        self.recovery_without_progress(name,
                            initial_listing=name=='backtrack_candidates' and not had_candidates and bool(self.last_candidates))
            except GenerationStopped as stop:
                status,model_resource_error = stop.status,stop.evidence
            self.finish_stage(status)
            return dict(complete=True,status=status,success=max([0.0,*self.purchases])==1,
                        benchmark_reward=max([0.0,*self.purchases]), final_public_reward=current['reward'],
                        executed_actions=self.actions, completed_recoveries=self.recoveries,
                        current_trajectory_actions=self.trajectory_actions,
                        recovery_stages=self.stages,cost=self.cost(),environment_accounting=self.env.accounting(),
                        protocol_error=None,protocol_rejections=self.protocol_rejections,
                        failed_terminal_submissions=self.terminal_failures,
                        consecutive_reflection_rejections=self.reflection_rejections,
                        consecutive_recovery_no_progress=self.recovery_no_progress,
                        recovery_paused=self.actor_until_accepted,
                        model_resource_error=model_resource_error)
        finally:
            self.writer.close()
            self.events.close()
