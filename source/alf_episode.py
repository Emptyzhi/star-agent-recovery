"""ALFWorld 0.3.5 public interface and exact same-instance replay restoration.

All methods see the same native admissible-command catalog. No expert action,
PDDL fact, solution, or nonterminal score is exposed to a model. Snapshots contain
executed prefixes; replay is real execution and its cost is counted separately.
"""
from copy import deepcopy
import json
from icr_core import ActionSupport, BudgetExhausted, ParsedAction, ProtocolError, SupportKind
from tree_history import canonical
from green_history_experiment import render_active_history
from alf_recovery_evidence import factor_path_interfaces
from alf_public_receipts import event_state_evidence, receipt_references, RECEIPT_SEMANTICS
from alf_actor_context import actor_input
from r27_recovery_prompts import YELLOW_SYSTEM, RED_SYSTEM, CORRECTION_SYSTEM
from alf_action_selection import command_choices, parse_selection
from alf_checkpoint_boundaries import checkpoint_boundaries
from alf_restored_facts import restored_facts
from alf_failed_branch_event import failed_branch_event


def parse_action(text):
    if type(text) is not str:
        raise ProtocolError('action must be text')
    line=text.split('\n')[0]
    marker=line.find('>')
    if marker>2:
        marker=-1
    action=line[marker+1:].strip()
    if len(action)>1:
        action=action[0].lower()+action[1:]
    return ParsedAction(action,bool(action) and '\r' not in action,
                        action.startswith('think:'),{'raw_text':text,'parser':'author_reflexion_first_line'})


def process_ob(text):
    return text[text.find('. ')+2:] if text.startswith('You arrive at loc ') else text


class CachedView:
    def __init__(self,checkpoint):
        self.checkpoint=checkpoint
    def snapshot(self):
        return deepcopy(self.checkpoint['environment']['public_state'])
    def support(self):
        return ActionSupport(SupportKind.FINITE,tuple(self.snapshot()['available_actions']),
                             'alfworld-public-admissible-v1')
    def context(self):
        return {'benchmark':'alfworld','review_mode':'CACHED_CHECKPOINT',
                'initial':self.checkpoint['initial'],'history':self.checkpoint['history']}


class AlfEpisode:
    native_actor_history=True
    green_tree_mode='event'
    compact_context=False
    terminal_guidance=False
    yellow_action_ids=True
    efficient_recovery=False
    tree_internal_repair=False
    def __init__(self,env,task_identity,base_prompt,formatter,*,action_limit=49,emit):
        self.raw=env
        self.task_identity=task_identity
        self.task=task_identity
        self.base_prompt=base_prompt
        self.review_formatter=formatter
        self.action_limit=action_limit
        self.emit=emit
        self.initial=None
        self.history=[]
        self._state=None
        self.backend_requests=0
        self.replay_actions=0
        self.replay_resets=0
        self.prompt_token_counter=None
        self.prompt_context_limit=32768
        self.prompt_service_limit=24000
        self._scope_view=None
        self._yellow_selection_state=None
        self._yellow_selection_commands=None
        self.recovery_event_subtree={'events':[], 'nodes':[]}
        self.last_history_window={'truncated':False,'omitted_records':0,'retained_records':0}

    def _read(self,result,*,reset=False,action=None):
        if reset:
            obs,info=result
            done=False
        else:
            obs,_,dones,info=result
            if len(dones)!=1 or type(dones[0]) is not bool:
                raise ProtocolError('non-singleton or nonboolean done receipt')
            done=dones[0]
        if len(obs)!=1 or type(obs[0]) is not str:
            raise ProtocolError('invalid ALFWorld observation batch')
        if reset and info['extra.gamefile'][0]!=self.task_identity:
            raise ProtocolError('reset changed game identity')
        if reset:
            parts=obs[0].split('\n\n')
            if len(parts)<2:
                raise ProtocolError('reset header delimiter missing')
            observation='\n'.join(parts[1:])
        else:
            observation='OK.' if action.startswith('think:') else process_ob(obs[0])
        commands=info['admissible_commands'][0]
        if not isinstance(commands,list) or any(type(c) is not str for c in commands):
            raise ProtocolError('invalid native command catalog')
        if len(set(commands))!=len(commands):
            raise ProtocolError('duplicate native catalog actions')
        if type(info['won'][0]) is not bool:
            raise ProtocolError('won is not boolean')
        score=int(info['won'][0]) if done else None
        self._state={'benchmark':'alfworld','observation':observation,
                     'last_public_observation':observation,
                     'available_actions':commands,'done':done,'public_terminal_score':score}
        return observation,done,score

    def reset(self):
        if self.initial is not None:
            raise ProtocolError('episode reset twice; restoration requires a checkpoint')
        self.backend_requests+=1
        observation,done,score=self._read(self.raw.reset(),reset=True)
        self.initial={'action':'reset','observation':observation,'before':None,'after':self.snapshot(),
                      'done':done,'public_terminal_score':score,'invalid_action':False,
                      'invalid_detail':None,'native_accepted':True,'backend_requests':1}
        return deepcopy(self.initial)

    def snapshot(self):
        if self._state is None:
            raise ProtocolError('no valid environment state')
        return deepcopy(self._state)

    parse=staticmethod(parse_action)
    def support(self):
        return ActionSupport(SupportKind.FINITE,tuple(self.snapshot()['available_actions']),
                             'alfworld-public-admissible-v1')
    def public_physical(self,parsed,support):
        if support!=self.support():
            raise ProtocolError('stale ALFWorld command support')
        return parsed.grammar_valid and not parsed.is_think and parsed.action in support.commands

    def step(self,action):
        if self._state['done'] or len(self.history)>=self.action_limit:
            raise ProtocolError('step after terminal or path budget')
        before=self.snapshot()
        self.backend_requests+=1
        observation,done,score=self._read(self.raw.step([action]),action=action)
        accepted=action.startswith('think:') or observation.strip()!='Nothing happens.'
        receipt={'action':action,'observation':observation,'before':before,'after':self.snapshot(),
                 'done':done,'public_terminal_score':score,'native_accepted':accepted,
                 'invalid_action':not accepted,'invalid_detail':None if accepted else observation,
                 'backend_requests':1}
        self.history.append(receipt)
        return deepcopy(receipt)

    def checkpoint(self):
        return {'environment':{'task_identity':self.task_identity,'public_state':self.snapshot()},
                'initial':deepcopy(self.initial),'history':deepcopy(self.history)}

    def local_obstruction(self,receipt):
        if not self.history or receipt!=self.history[-1]:
            raise ProtocolError('local obstruction requires the latest actual receipt')
        confirmed=(receipt['invalid_action'] and len(self.history)>=2
                   and self.history[-2]['invalid_action']
                   and self.history[-2]['after']==receipt['after'])
        self.emit('alfworld_local_obstruction_assessed',confirmed=confirmed,
                  path_length=len(self.history),action=receipt['action'],
                  invalid_action=receipt['invalid_action'],
                  criterion='two consecutive rejected receipts with identical public after-state')
        return confirmed

    def restore(self,checkpoint):
        if checkpoint['initial']!=self.initial or checkpoint['environment']['task_identity']!=self.task_identity:
            raise ProtocolError('checkpoint belongs to a different task')
        expected=deepcopy(checkpoint)
        start=self.backend_requests
        self.backend_requests+=1
        self.replay_resets+=1
        self._read(self.raw.batch_env.reset(),reset=True)
        if self.snapshot()!=self.initial['after']:
            raise ProtocolError('reset replay state differs from original root')
        self.history=[]
        for row in expected['history']:
            actual=self.step(row['action'])
            self.replay_actions+=1
            if actual!=row:
                raise ProtocolError('replayed action/observation/catalog differs from checkpoint prefix')
        if self.checkpoint()!=expected:
            raise ProtocolError('restoration did not reproduce checkpoint')
        self.emit('alfworld_checkpoint_replayed',prefix_length=len(self.history),
                  backend_requests=self.backend_requests-start)
        return {'backend_requests':self.backend_requests-start}

    def context(self):
        return {'benchmark':'alfworld','task_identity':self.task_identity,'reset':self.initial,
                'history':deepcopy(self.history),'base_prompt':self.base_prompt,
                'public_interface':'native_admissible_commands_shared_across_arms'}

    def result_metadata(self):
        projection={'task':self.task_identity,'public_state':self.snapshot()}
        return {'projection':projection,'complete_projection':projection,
                'environment_contract':'alfworld-textworld-0.3.5-json2.1.1'}

    def set_scope_prompt(self,view,version,*,kind,directive,incident_kind=None):
        if type(view) is not str or kind not in {'GREEN','YELLOW','RED'} or (not view and kind != 'GREEN'):
            raise ProtocolError('invalid recovery scope binding')
        self._scope_view=view
        self._scope_kind=kind
        self._scope_directive=directive
        self._scope_incident_kind=incident_kind

    def catalog_text(self):
        return '\n[CURRENT_ADMISSIBLE_COMMANDS]\n'+canonical(self.snapshot()['available_actions'])+'\n'

    def prompt(self,memory):
        if memory!=() or self.initial is None or self._scope_view is None:
            raise ProtocolError('STAR requires a bound scope and no external memory')
        text=actor_input(self,())
        if self._scope_kind=='GREEN' and getattr(self,'red_repair_handoff',None) is not None:
            facts=restored_facts(self.history,self.snapshot())
            text+='\n[ACTIVE_RED_REPAIR_MEMORY]\n'+canonical(self.red_repair_handoff)
            text+='\n[CURRENT_TRANSFER_EVIDENCE]\n'+canonical({
                'known_held_objects':facts['known_held_objects'],
                'latest_successful_navigation':facts['latest_successful_navigation'],
                'latest_successful_transfer':facts['latest_successful_transfer'],
                'latest_receipt':facts['latest_receipt'],
                'semantics':facts['semantics']})
            text+=('\nThe historical lesson is conditional advice, not a current state assertion. '
                   'Read the newest receipt first. When an operation has succeeded, move to the '
                   'next unmet prerequisite instead of repeating that operation. Check your current '
                   'location and possession before taking or placing an object. A search location '
                   'already observed without the target is not new evidence on another visit. '
                   'Return one next native action for the original task.\n')
        if self._scope_view:
            if self._scope_kind=='GREEN':
                text+='\n'+self._scope_view+'\nReturn one next action using the current task and actual receipts.\n'
        text+='>'
        if self.prompt_token_counter(text)>min(self.prompt_context_limit-100,self.prompt_service_limit):
            raise BudgetExhausted('rendered ALFWorld Actor input exceeds declared context; no additional truncation')
        self.last_history_window={'truncated':False,'omitted_records':0,'retained_records':len(self.history)}
        return text

    def yellow_conclusion_prompt(self,path,terminals,protocol_feedback=None):
        system=YELLOW_SYSTEM
        self._yellow_selection_state=self.snapshot()
        self._yellow_selection_commands=tuple(self._yellow_selection_state['available_actions'])
        evidence={'task':self.initial['observation'],'incident':self._scope_incident_kind,
                  'event':self._scope_view,'history':self.native_history(),
                  'public_receipt_refs':receipt_references(self.history),
                  'public_receipt_semantics':RECEIPT_SEMANTICS,
                  'recovery_event_subtree':self.recovery_event_subtree,
                  'current_admissible_commands':command_choices(self._yellow_selection_commands),
                  'action_id_semantics':'IDs are zero-based and bound only to this unchanged current '
                      'state and menu. Select one listed integer; do not compose a command or copy '
                      'an ID from an earlier state. Every listed command is public interface evidence, '
                      'not a recommendation or proof it completes the task.',
                  'remaining_actions':self.action_limit-len(self.history),
                  'protocol_feedback':protocol_feedback}
        return self.review_formatter([{'role':'system','content':system},
                                      {'role':'user','content':self.recovery_examples()+canonical(evidence)}])

    def parse_yellow_selection(self,text):
        if self._yellow_selection_state is None or self.snapshot()!=self._yellow_selection_state:
            raise ProtocolError('Yellow command-ID response is not bound to the current state')
        conclusion=parse_selection(text,self._yellow_selection_commands)
        self.emit('alf_yellow_command_id_selected',action=conclusion['current_action'],
                  action_id=self._yellow_selection_commands.index(conclusion['current_action']),
                  native_menu_size=len(self._yellow_selection_commands))
        return conclusion

    def red_batch_prompt(self,evidence):
        self.red_phase1_messages=self.red_batch_messages(evidence)
        return self.review_formatter(self.red_phase1_messages)

    def red_causal_assessment_prompt(self, evidence):
        """Red explains the failure; a later bounded request identifies an action edge."""
        transcript = self.recovery_examples() + '\n[TASK]\n' + self.initial['observation'] + '\n'
        transcript += '[FAILED_BRANCH_EVENT_SUBTREE]\n' + canonical(failed_branch_event(self.history)) + '\n'
        transcript += '[LIVE_FAILURE_OBSERVATION]\n' + self.snapshot()['observation'] + '\n'
        transcript += '[FAILED_EXECUTION_RECEIPTS]\n' + canonical([
            {'index': i, 'action': receipt['action'], 'accepted': receipt['native_accepted'],
             'before': receipt['before'], 'after': receipt['after']}
            for i, receipt in enumerate(self.history)])
        transcript += ('\nExplain the unmet original task requirement and the observed failure. '
                       'Do not select a checkpoint, restore step, or action number. '
                       'Do not invent hidden causes or treat discarded actions as completed. '
                       'A separate controller request will identify the first public transition '
                       'that cannot be certified as correct.')
        return self.review_formatter([
            {'role': 'system', 'content': 'You are Red. Produce a concise, conditional failure assessment from public receipts.'},
            {'role': 'user', 'content': transcript}])

    def red_batch_messages(self,evidence):
        system=('You are Red, reviewing a failed attempt. Explain the unmet task requirement '
                'and a useful place to resume, using only the supplied public evidence. '
                'Write a brief natural-language assessment. No field names, JSON or fixed '
                'line format are required. A separate choice follows your assessment.')
        # Restricting the checkpoint choices must not restrict failure evidence.
        transcript=self.recovery_examples()
        transcript+='[TASK]\n'+self.initial['observation']+'\n'
        transcript+='[FAILED_BRANCH_EVENT_SUBTREE]\n'+canonical(
            failed_branch_event(self.history))+'\n'
        transcript+='[LIVE_FAILURE_OBSERVATION]\n'+self.snapshot()['observation']+'\n'
        outline=checkpoint_boundaries(self.history,evidence['candidates'],self.action_limit)
        rows=[]
        for row in outline['rows']:
            index=row[0]
            state=self.history[index]['before']
            rows.append([index,row[1],row[2],row[-1],state['observation']])
        self.red_checkpoint_choices=[{
            'choice':f'CHECKPOINT {row[0]}',
            'remaining_environment_actions':row[3],
            'last_completed_receipt':row[1],
            'historical_next_action_NOT_executed_at_restore':row[2]}
            for row in rows]
        transcript+='[CHECKPOINT_INDEX]\n'+canonical({
            'columns':['checkpoint_index','last_completed_receipt','next_action',
                       'remaining_actions','checkpoint_observation'],
            'rows':rows,
            'semantics':'These rows restrict selectable checkpoints only. The failed-branch '
                        'event above is independent of this list. The controller retains the '
                        'full ETT; current admissible actions are supplied in phase 2.'})
        transcript+=('\nExplain why the attempt failed and which recorded state offers a '
                     'plausible continuation within its remaining budget. Checkpoint k '
                     'restores BEFORE action k; only receipts 0 through k-1 have occurred. '
                     'Estimate how many environment actions your continuation needs, '
                     'including navigation and manipulation, and compare with the candidate budget. '
                     'If it needs more actions, consider an earlier supplied checkpoint. '
                     'Express uncertainty honestly, but recovery is required: choose one supplied '
                     'checkpoint and continue with a supported native action. Keep the assessment concise.')
        return [{'role':'system','content':system},{'role':'user','content':transcript}]

    def red_action_prompt(self,checkpoint,checkpoint_index,assessment):
        state=checkpoint['environment']['public_state']
        content=(self.recovery_examples()+'\n[FIXED_CHECKPOINT]\n'+canonical({
            'task':self.initial['observation'],'checkpoint_index':checkpoint_index,
            'remaining_actions':self.action_limit-checkpoint_index,
            'observation':state['observation'],
            'current_admissible_commands':state['available_actions']})+
            '\n[RESTORED_STATE_FACTS]\n'+canonical(restored_facts(checkpoint['history'],state))+
            '\n[FAILED_BRANCH_EVENT_SUBTREE]\n'+canonical(failed_branch_event(self.history))+
            '\n[RESTORED_BRANCH_RECEIPTS]\n'+canonical([
                {'receipt_index':i,'action':receipt['action'],
                 'observation':receipt['observation'],'accepted':receipt['native_accepted']}
                for i,receipt in enumerate(checkpoint['history'])])+
            '\nThese receipts are the actual retained branch, not the discarded future. '
            'Infer held objects and completed operations only from successful receipts; '
            'a visible object or a historical next-action label does not prove it was taken. '
            'The action budget includes the first action and every later action. '+
            '\n[PHASE1_ASSESSMENT]\n'+assessment+
            '\nThe assessment is model advice, not verified environment evidence. '
            'Use the task and public observation to evaluate it. The checkpoint is fixed. '
            'Write one brief conditional lesson from the FAILED attempt. Identify the observed '
            'mistake and what evidence must be checked before repeating it. Keep the task object '
            'and target unchanged. Write the lesson as a condition that can remain useful after '
            'the next action succeeds. Do not write a first action, an ordered plan, or present-tense '
            'claims about what is currently held or already completed. Do not invent object '
            'locations or changes to previously observed contents. The next action is chosen '
            'separately from the restored state. Natural language only; no fixed format is required.')
        action_system=('You are Red in phase 2. The checkpoint has already been selected and is fixed. '
                       'Use the historical failed-branch event to diagnose the failed attempt, '
                       'and the restored-state facts and current_admissible_commands to identify '
                       'what is true and callable now. Never treat discarded future actions as '
                       'completed in the restored state. '
            'Give a concise conditional lesson about a historical failure. It will remain advice, '
            'not a changing state summary or a plan to replay. A separate native-action choice '
            'will follow; the controller does not parse commands from this advice.')
        self.red_phase2_messages=[{'role':'system','content':action_system},
                                  {'role':'user','content':content}]
        return self.review_formatter(self.red_phase2_messages)

    def red_selection_prompt(self,phase,assessment,choices):
        if phase not in (1,2):
            raise ProtocolError('unknown Red selection phase')
        messages=self.red_phase1_messages if phase==1 else self.red_phase2_messages
        instruction=('Select exactly one supplied checkpoint to restore. Recovery is required.' if phase==1 else
                     'Use RESTORED_STATE_FACTS to select one supplied native action NOW. Recovery is required. '
                     'The historical lesson is not an action plan. Check current location, held '
                     'objects and successful receipts before choosing. This instruction expires '
                     'when the action executes; later actions will be chosen from new observations.')
        if phase==1:
            instruction+= ('\n[RESTORE_BOUNDARIES_AND_BUDGETS]\n'+canonical(self.red_checkpoint_choices)+
                '\nSelect a boundary whose retained progress and remaining environment-action '
                'budget support the continuation you just proposed. At CHECKPOINT k, action k '
                'has NOT happened. A one-action budget cannot execute a multi-action continuation. '
                'An earlier checkpoint remains available; the controller has not preselected one.')
        return self.review_formatter(messages+[
            {'role':'assistant','content':assessment},
            {'role':'user','content':f'[RED_CHOICE_PHASE_{phase}]\n'+instruction+
             '\nChoose exactly one supplied option; the decoder handles formatting.\n'+canonical(list(choices))}])

    def red_callability_prompt(self,evidence,raw_proposal,feedback,*,checkpoint):
        system=CORRECTION_SYSTEM
        messages=[{'role':'system','content':system},
                  {'role':'user','content':self.recovery_examples()+checkpoint['initial']['observation']+'\n[RETAINED_HISTORY]\n'
                   +self.native_history(checkpoint['history'])+'[PUBLIC_RECEIPT_REFS]\n'
                   +canonical(receipt_references(checkpoint['history']))+'\n'+RECEIPT_SEMANTICS},
                  {'role':'assistant','content':raw_proposal},
                  {'role':'user','content':feedback}]
        return self.review_formatter(messages)

    def recovery_examples(self):
        return ('[AUTHOR_INTERFACE_EXAMPLES]\n'+self.base_prompt+
                '\n[AUTHOR_INTERFACE_EXAMPLES_END]\n'
                'The following is the current task and its actual evidence.\n')

    def native_history(self,history=None):
        history=self.history if history is None else history
        return ''.join(f'[{i}] > '+row['action']+'\n'+row['observation']+'\n' for i,row in enumerate(history))

    def checkpoint_review_view(self,checkpoint):
        if checkpoint['initial']!=self.initial:
            raise ProtocolError('foreign review checkpoint')
        return CachedView(checkpoint)
    def checkpoint_action_callable(self,checkpoint,action):
        return action in self.checkpoint_review_view(checkpoint).support().commands
    def _current_item_evidence(self,snapshot,history):
        # ALFWorld has no item-page supplement; all evidence is in actual receipts.
        return ''
    def _current_search_evidence(self,snapshot,history):
        # No WebShop search-page supplement exists in the ALFWorld interface.
        return ''
    def close(self):
        self.raw.close()
