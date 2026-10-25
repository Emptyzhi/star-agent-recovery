"""Offline state-machine tests using native WebShop parsing and mocked HTTP."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'source'))

from icr_core import Generation, Ledger, ProposalOracle, ProtocolError
from icr_prompts import reflection_functions
from scoped_recovery import ScopeKind
from web_v16_controller import WebV16Controller
from web_v16_episode import WebV16Episode, author_reflection
from web_v16_evidence import decision_key, new_evidence, option_resume_event, product_review_event
from webshop_semantics import (checkpoint_kind, key_checkpoint_indices, progress_kind,
                               public_fault_action_index, yellow_local_commands)
from web_v16_run import task_limits, verify_runtime, verify_backend


def formatter(messages):
    return '\n'.join(m['role']+': '+m['content'] for m in messages)


def make_episode():
    episode = WebV16Episode('http://127.0.0.1:9999', 'unit_fixed_0', 0, 'AUTHOR ACT EXAMPLE', formatter)
    def get(url, timeout):
        page = episode.environment._session['page_type']
        if page == 'init':
            html = '<div>WebShop</div><div>Instruction:</div><div>Buy a red cup below 20 dollars.</div><button>Search</button>'
        elif page == 'search':
            html = '<div>WebShop</div><div>Instruction</div><button>Back to Search</button><button>Next &gt;</button>'
            html += ''.join('<a class="product-link">%s</a><div>%s cup</div><div>$10</div>' % (s, s)
                            for s in ('A', 'B', 'C', 'HIDDEN'))
        elif page == 'item':
            html = ('<div>WebShop</div><div>Instruction</div><button>Back to Search</button><button>&lt; Prev</button>'
                '<div>color</div><label>red</label><label>blue</label><div>Cup</div><div>Price: $10</div>'
                '<button>Description</button><button>Attributes</button><button>Buy Now</button>')
        elif page == 'item_sub':
            html = '<div>WebShop</div><div>Instruction</div><button>&lt; Prev</button><div>Public cup material</div>'
        elif page == 'end':
            options = episode.environment._session.get('options', {})
            score = '1.0' if 'red' in options.values() else '0.5'
            html = '<div>Your score (min 0.0, max 1.0)</div><div>'+score+'</div>'
        else:
            raise AssertionError(page)
        return SimpleNamespace(text=html, raise_for_status=lambda: None)
    episode.environment._http.get = get
    episode.prompt_token_counter = lambda p: len(p)//5
    episode.prompt_context_limit = 32768
    episode.prompt_service_limit = 24000
    episode.ordinary_command_constrained = False
    return episode


def make_controller(outputs,settings=None):
    episode = make_episode()
    events, requests = [], []
    iterator = iter(outputs)
    def emit(event, **data):
        events.append(dict(event=event, **data))
    def generate(request):
        requests.append(request)
        text = next(iterator)
        return Generation(request, text, len(request.prompt)//5, min(len(text)//4+1, request.output_limit), {})
    oracle = ProposalOracle(generate, master_seed=1, emit=emit, token_counter=lambda p: len(p)//5,
        context_limit=32768, prompt_limit=24000, request_namespace='test', generation_versions={})
    c = WebV16Controller(episode, oracle, Ledger(task_limits(settings=settings)), task=0, emit=emit,settings=settings)
    c.start()
    return c, requests, events


class WebV16Tests(unittest.TestCase):
    def test_webshop_semantic_snapshot_contains_public_product_state(self):
        e = make_episode()
        e.reset(); e.step('search[cup]'); e.step('click[A]')
        state = e.snapshot()
        semantics = state['webshop_semantics']
        self.assertEqual(semantics['asin'], 'A')
        self.assertEqual(semantics['price_text'], 'Price: $10')
        self.assertEqual(semantics['product_title'], 'Cup')
        e.step('click[< Prev]'); e.step('click[A]'); e.step('click[red]')
        semantics = e.snapshot()['webshop_semantics']
        self.assertEqual(semantics['selected_options'], {'color': 'red'})
        self.assertIn('red', semantics['option_domains']['color']['labels'])

    def test_yellow_menu_only_contains_current_item_validation(self):
        e = make_episode(); e.reset(); e.step('search[cup]'); e.step('click[A]')
        commands = yellow_local_commands(e.snapshot())
        self.assertIn('click[red]', commands)
        self.assertIn('click[Description]', commands)
        self.assertNotIn('click[Back to Search]', commands)
        self.assertNotIn('click[Buy Now]', commands)
        e.step('click[< Prev]')
        self.assertEqual(yellow_local_commands(e.snapshot()), ())

    def test_webshop_progress_taxonomy_distinguishes_information_and_decision(self):
        e = make_episode(); e.reset(); e.step('search[cup]'); e.step('click[A]')
        before = e.history[-1]['before']; after = e.history[-1]['after']
        self.assertEqual(progress_kind(before, after, 'click[A]', accepted=True, done=False),
                         'PRODUCT_CHANGED')
        e.step('click[Description]')
        self.assertEqual(progress_kind(e.history[-1]['before'], e.history[-1]['after'],
                                       'click[Description]', accepted=True, done=False),
                         'SUBPAGE_VIEWED')
        e.step('click[< Prev]'); e.step('click[A]'); e.step('click[red]')
        self.assertEqual(progress_kind(e.history[-1]['before'], e.history[-1]['after'],
                                       'click[red]', accepted=True, done=False),
                         'OPTION_CHANGED')

    def test_webshop_key_checkpoints_exclude_pure_information_view(self):
        e = make_episode(); e.reset(); e.step('search[cup]'); e.step('click[A]')
        e.step('click[Description]'); e.step('click[< Prev]'); e.step('click[red]')
        occurrences = [SimpleNamespace(action=row['action'], receipt=row)
                       for row in e.history]
        kinds = {index: checkpoint_kind(item) for index, item in enumerate(occurrences)}
        candidates = key_checkpoint_indices(occurrences)
        self.assertEqual(kinds[2], 'INFORMATION_VIEW')
        self.assertNotIn(2, candidates)
        self.assertIn(0, candidates)
        self.assertIn(1, candidates)
        self.assertIn(4, candidates)
    def test_author_reflection_identical(self):
        reference = reflection_functions(PACKAGE/'vendor/webshop_reflection_prompt.py',
            PACKAGE/'prompts/webshop/webshop_reflexion_examples.txt')
        transcript = 'Instruction: buy a cup\nAction: click[Buy Now]\nObservation: score 0.5'
        self.assertEqual(author_reflection(transcript), reference(transcript+'\n\nSTATUS: FAIL', []))
        self.assertNotIn('stoveburner', author_reflection(transcript))

    def test_initial_success_no_repair(self):
        c, requests, _ = make_controller(['search[cup]', 'click[A]', 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(len(requests), 4)
        self.assertEqual(c.checkpoint_restore_count, 0)
        self.assertIn('[EVENT_SUBTREE]', requests[2].prompt)
        self.assertNotIn('[EVENT_SUBTREE]', requests[3].prompt)

    def test_partial_score_reflect_locate_restore_then_green(self):
        c, requests, events = make_controller(['search[cup]', 'click[A]', 'click[Buy Now]',
            'Select the requested red color before buying.', 'ACTION 2', 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(c.selected_fault_index, 2)
        self.assertEqual(c.episode.action_limit, 17)
        self.assertEqual(state['forward_trials'][0]['forward_actions_granted'], 15)
        self.assertEqual(c.episode.snapshot()['selected_options'], {'color': 'red'})
        self.assertEqual(c.episode.backend_requests, 6)  # reset + five actions, no replay
        self.assertEqual([o.receipt['public_terminal_score'] for o in c.occurrences.values() if o.receipt['done']], [.5, 1.])
        recovery = [r for r in requests if 'Actions in this recovery attempt:' in r.prompt]
        self.assertEqual(len(recovery), 2)
        self.assertIn('"kind":"RED_HANDOFF"', recovery[0].prompt)
        self.assertIn('Select the requested red color', recovery[0].prompt)
        self.assertIn('terminal_purchase_evidence', recovery[0].prompt)
        self.assertNotIn('"kind":"RED_HANDOFF"', recovery[1].prompt)
        self.assertNotIn('Select the requested red color', recovery[1].prompt)
        self.assertTrue(all('fault_transition' not in r.prompt for r in recovery))
        self.assertTrue(all('discarded_failed_suffix' not in r.prompt for r in recovery))
        self.assertIn('HISTORICAL_BRANCH', recovery[0].prompt)
        self.assertNotIn('Action: click[Buy Now]', recovery[1].prompt.split('Actions in this recovery attempt:')[1])
        self.assertEqual(len([e for e in events if e['event']=='red_memory_delivered']), 0)
        self.assertFalse(any(e['event'] == 'red_replacement_submitted' for e in events))

    def test_production_webshop_ordinary_uses_native_finite_commands(self):
        e = make_episode()
        e.ordinary_command_constrained = True
        e.reset(); e.step('search[cup]'); e.step('click[A]')
        self.assertEqual(e.support().kind.value, 'FINITE')

    def test_red_audits_all_mismatches_and_restores_before_condition_selection(self):
        reflection = ('The selected blue color mismatches the requested red color; the displayed '
                      'price matches. Correct the color before buying.')
        c, requests, events = make_controller(['search[cup]', 'click[A]', 'click[blue]',
            'click[Buy Now]', reflection, 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(c.selected_fault_index, 2)
        self.assertEqual(c.episode.snapshot()['selected_options'], {'color': 'red'})
        self.assertEqual([o.receipt['public_terminal_score'] for o in c.occurrences.values()
                          if o.receipt['done']], [.5, 1.])
        reflection_prompt = next(r.prompt for r in requests if r.route == 'reflection'
                                 and 'You will be given the history' in r.prompt)
        self.assertIn('List all confirmed mismatches', reflection_prompt)
        self.assertLess(reflection_prompt.index('UNKNOWN'), reflection_prompt.rindex('New plan:'))
        self.assertFalse(any(r.route == 'command_trie' and 'Select the action to change' in r.prompt
                             for r in requests))
        mapped = next(e for e in events if e['event'] == 'red_action_index_mapped')
        self.assertEqual(mapped['checkpoint_index'], 2)
        restored = next(e for e in events if e['event'] == 'red_checkpoint_restored')
        self.assertEqual(restored['checkpoint_index'], 2)

    def test_phase1_public_evidence_reaches_fallback_index_selection(self):
        c, requests, events = make_controller(['search[cup]', 'click[A]', 'click[Buy Now]',
            'The selected product does not match the requested item.', 'ACTION 1',
            'click[A]', 'click[red]', 'click[Buy Now]'])
        c.run()
        selection = next(r for r in requests if r.route == 'command_trie')
        self.assertIn('[RED_PHASE1_EVIDENCE]', selection.prompt)
        self.assertIn('public_transitions', selection.prompt)
        recorded = next(e for e in events if e['event'] == 'red_phase1_evidence_recorded')
        self.assertEqual(recorded['evidence']['schema'], 'webshop_red_phase1_public_evidence_v1')

    def test_yellow_diagnosis_feedback_and_green_handoff(self):
        c, requests, events = make_controller(['search[cup]', 'click[Next >]',
            'The next button is rejected; inspect a visible item instead.',
            'click[A]', 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(len(c.local_outcomes), 1)
        self.assertEqual(c.local_outcomes[0]['actions'], 1)
        diagnosis = [r for r in requests if r.stop_mode=='first_nonempty_line']
        self.assertEqual(len(diagnosis), 1)
        self.assertEqual(diagnosis[0].output_limit, 100)
        choice = next(r for r in requests if r.route == 'command_trie')
        self.assertEqual(choice.route, 'command_trie')
        self.assertNotIn('click[Next >]', choice.commands)
        self.assertNotIn('click[HIDDEN]', choice.commands)
        self.assertTrue(any('YELLOW_HANDOFF' in r.prompt for r in requests))
        self.assertTrue(any('PRODUCT_REQUIREMENT_REVIEW' in r.prompt for r in requests))
        correction_events = [e for e in events if e['event'] == 'yellow_correction_delivered']
        self.assertEqual(len(correction_events), 1)

    def test_all_blank_diagnosis_returns_to_green_without_guessing(self):
        c, _, events = make_controller(['search[cup]', 'click[Next >]', ' \n',
            'click[A]', 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertNotEqual(state['status'], 'PROTOCOL_ERROR')
        self.assertTrue(any(e['event'] == 'yellow_local_diagnosis'
                            and e['diagnosis']['status'] == 'EMPTY_DIAGNOSIS_PROTOCOL_EVENT'
                            for e in events))
        self.assertFalse(any(e['event'] == 'fatal' for e in events))

    def test_invalid_yellow_outputs_are_audited_and_return_to_green(self):
        c, _, events = make_controller(['Inspect the current page.', '', 'not an action', 'still not an action'])
        green = c._green_scope()
        for action in ('search[cup]', 'click[A]', 'click[< Prev]', 'click[A]', 'click[< Prev]'):
            c._execute(green, c.episode.parse(action), 'setup'+str(len(c.lineage)))
        incident = c._proposal_incident('click[A]')
        self.assertIsNotNone(incident)
        c._local_repair(incident, 'click[A]')
        self.assertEqual(c.active_scope.kind, ScopeKind.GREEN)
        self.assertEqual(c.local_outcomes[-1]['outcome'], 'UNRESOLVED_YELLOW_PROTOCOL')
        self.assertEqual(len(c.occurrences), 5)
        failures = [e for e in events if e['event'] == 'yellow_protocol_failure']
        self.assertEqual(len(failures), 1)
        self.assertTrue(all(not e['executed'] for e in failures))
        self.assertTrue(all('public_checkpoint' in e for e in failures))

    def test_public_mismatch_maps_to_first_value_transition(self):
        c, _, _ = make_controller([])
        green = c._green_scope()
        for action in ('search[cup]', 'click[A]', 'click[blue]', 'click[Buy Now]'):
            c._execute(green, c.episode.parse(action), 'setup'+str(len(c.lineage)))
        mapping = public_fault_action_index(c.lineage,
            'The selected blue color mismatches the requested red color.')
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping['action_index'], 2)

    def test_malformed_green_action_not_repaired(self):
        c, _, events = make_controller(['search[cup',
            'Use the public search form.', 'search[cup]', 'click[A]', 'click[red]', 'click[Buy Now]'])
        self.assertEqual(c.run()['status'], 'SUCCESS')
        self.assertNotIn('search[cup', [o.action for o in c.occurrences.values()])
        incidents = [e for e in events if e['event'] == 'yellow_incident_detected']
        self.assertEqual(incidents[0]['kind'], 'MODEL_ACTION_PROTOCOL_FAILURE')
        self.assertEqual(incidents[0]['action'], 'search[cup')

    def test_native_parser_keeps_original_evidence(self):
        e = make_episode()
        parsed = e.parse(' search[cup]')
        self.assertEqual(parsed.action, 'search[cup]')
        self.assertEqual(parsed.evidence['raw_text'], ' search[cup]')
        self.assertFalse(e.parse('search[cup').grammar_valid)

    def test_exact_checkpoint_restores_product_options_and_history(self):
        e = make_episode()
        e.reset(); e.step('search[cup]'); e.step('click[A]'); e.step('click[red]')
        checkpoint = e.checkpoint()
        e.step('click[blue]'); e.step('click[Buy Now]')
        calls = e.backend_requests
        e.restore(checkpoint)
        self.assertEqual(e.checkpoint(), checkpoint)
        self.assertEqual(e.snapshot()['selected_options'], {'color': 'red'})
        self.assertFalse(e.snapshot()['done'])
        self.assertEqual(e.backend_requests, calls)

    def test_page_cycle_not_new_evidence_and_options_change_state(self):
        e = make_episode(); e.reset(); e.step('search[cup]'); e.step('click[A]')
        e.step('click[< Prev]'); e.step('click[A]')
        first = decision_key(e.history, e.snapshot(), e.initial)
        e.step('click[< Prev]')
        old = list(e.history)
        receipt = e.step('click[A]')
        self.assertFalse(new_evidence(receipt, old, e.initial))
        self.assertEqual(first, decision_key(e.history, e.snapshot(), e.initial))
        e.step('click[red]')
        self.assertNotEqual(first, decision_key(e.history, e.snapshot(), e.initial))

    def test_private_search_results_never_enter_public_prompt(self):
        e = make_episode(); e.reset(); e.step('search[cup]')
        self.assertIn('HIDDEN', e.checkpoint()['environment']['native_state']['asins'])
        e.set_scope_prompt('', 0, kind='GREEN', directive='')
        self.assertNotIn('HIDDEN', e.prompt(()))
        self.assertNotIn('HIDDEN', json.dumps(e.context()))
        self.assertNotIn('click[HIDDEN]', e.support().commands)

    def test_yellow_repeat_returns_green_no_early_red(self):
        c, _, events = make_controller([])
        green = c._green_scope()
        for action in ('search[cup]', 'click[A]', 'click[< Prev]', 'click[A]', 'click[< Prev]'):
            c._execute(green, c.episode.parse(action), 'setup'+str(len(c.lineage)))
        incident = c._proposal_incident('click[A]')
        self.assertIsNotNone(incident)
        scripted = iter(['click[A]', 'click[< Prev]', 'click[A]'])
        def ordinary(scope, directive):
            c.yellow_diagnoses.setdefault(scope.scope_id, dict(text='Check the page.', request_id='test'))
            return c.episode.parse(next(scripted)), 'script'
        c._ordinary = ordinary
        c._local_repair(incident, 'click[A]')
        self.assertEqual(c.active_scope.kind, ScopeKind.GREEN)
        self.assertEqual(c.local_outcomes[-1]['outcome'], 'UNRESOLVED_REPEATED_INCIDENT_EDGE')
        self.assertEqual(c.local_outcomes[-1]['actions'], 0)
        self.assertEqual(c.attempt_scope_entries['RED'], 0)
        self.assertEqual(c._forward_window()['new_actions_remaining'], 10)

    def test_yellow_four_action_cap_and_red_window_boundaries(self):
        c, requests, _ = make_controller([])
        green = c._green_scope()
        for action in ('search[cup]', 'click[A]'):
            c._execute(green, c.episode.parse(action), 'setup'+str(len(c.lineage)))
        incident = dict(kind='ENVIRONMENT_REJECTION', action='click[missing]', incident_id='cap',
                        occurrence_id=0, receipt=c.lineage[0].receipt)
        # Every real transition is retained, while this fixture tests an unresolved
        # repair irrespective of novelty. No extra environment budget is granted.
        c._yellow_receipt_outcome = lambda *args: None
        actions = iter(['click[red]', 'click[blue]', 'click[Description]', 'click[< Prev]'])
        def ordinary(scope, directive):
            c.yellow_diagnoses.setdefault(scope.scope_id, dict(text='Inspect.', request_id='fixture'))
            return c.episode.parse(next(actions)), 'fixture'
        c._ordinary = ordinary
        c._local_repair(incident, incident['action'])
        self.assertEqual(c.local_outcomes[-1]['actions'], 1)
        self.assertEqual(c._forward_window()['new_actions_remaining'], 12)
        self.assertEqual(c.attempt_scope_entries['RED'], 0)

    def test_exhaustion_allows_only_one_red_and_two_complete_windows(self):
        c, requests, _ = make_controller(['think[step %d]' % i for i in range(15)] +
            ['Try a concrete shopping action.', 'ACTION 0'] + ['think[retry %d]' % i for i in range(15)])
        state = c.run()
        self.assertEqual(state['status'], 'RECOVERY_WINDOW_EXHAUSTED')
        self.assertEqual(len(c.occurrences), 30)
        self.assertEqual(len(c.lineage), 15)
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertEqual(c.attempt_scope_entries['RED'], 1)
        self.assertEqual(len(requests), 32)

    def test_context_overflow_explicit(self):
        c, _, _ = make_controller([])
        c.episode.prompt_service_limit = 1
        with self.assertRaisesRegex(ProtocolError, 'exceeds context'):
            c._bind_scope(c._green_scope(), '')

    def test_requested_option_resume_is_same_product_and_goal_grounded(self):
        c, _, _ = make_controller([])
        green = c._green_scope()
        for action in ('search[cup]', 'click[A]', 'click[< Prev]', 'click[red]', 'click[A]'):
            c._execute(green, c.episode.parse(action), 'setup'+str(len(c.lineage)))
        event = option_resume_event(c)
        self.assertEqual(event['action'], 'click[red]')
        self.assertEqual(event['evidence']['option_label'], 'red')
        c.delivered_green_events.add(event['event_key'])
        self.assertIsNone(option_resume_event(c))
        c.delivered_green_events.clear()
        c.episode.instruction = 'Buy a blue cup below 20 dollars.'
        self.assertIsNone(option_resume_event(c))
        c.episode.instruction = 'Buy a red cup below 20 dollars.'
        c.lineage[-2].receipt['before']['asin'] = 'OTHER'
        self.assertIsNone(option_resume_event(c))

    def test_product_review_cap_and_no_navigation_resume(self):
        c, _, _ = make_controller([])
        green = c._green_scope()
        for action in ('search[cup]', 'click[A]'):
            c._execute(green, c.episode.parse(action), 'fixture')
        a = product_review_event(c)
        c.delivered_green_events.add(a['event_key'])
        self.assertIsNone(product_review_event(c))
        c._execute(green, c.episode.parse('click[< Prev]'), 'fixture')
        c._execute(green, c.episode.parse('click[B]'), 'fixture')
        b = product_review_event(c)
        c.delivered_green_events.add(b['event_key'])
        c._execute(green, c.episode.parse('click[< Prev]'), 'fixture')
        c._execute(green, c.episode.parse('click[C]'), 'fixture')
        self.assertIsNone(product_review_event(c))
        self.assertIsNone(option_resume_event(c))
        self.assertIn('click[Buy Now]', c.episode.support().commands)

    def test_open_root_search_decision_returns_to_green(self):
        c, requests, _ = make_controller(['search[cup]', 'click[Back to Search]',
            'search[cup]', 'click[Back to Search]',
            'search[red cup]', 'click[A]', 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(len(c.local_outcomes), 1)
        self.assertGreaterEqual(len([r for r in requests if '[YELLOW_EVENT_SUBTREE]' in r.prompt]), 1)

    def test_yellow_limit_does_not_veto_green_and_resets_after_restore(self):
        c, _, events = make_controller(['search[cup]', 'click[Next >]', 'click[Next >]',
            'click[A]', 'click[Buy Now]', 'Choose red.', 'ACTION 4', 'click[red]', 'click[Buy Now]'])
        c.attempt_scope_entries['YELLOW'] = 2
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual([o.action for o in c.archived_lineages[0]][1:3], ['click[Next >]', 'click[Next >]'])
        self.assertEqual(c.attempt_scope_entries['YELLOW'], 0)

    def test_two_yellow_entries_per_window_are_a_hard_limit(self):
        c, _, _ = make_controller([])
        # Native caller and return frames are exercised by the ordinary feedback
        # tests; here verify that the underlying ETT cannot grant a third scope.
        c.attempt_scope_entries['YELLOW'] = 2
        with self.assertRaisesRegex(ProtocolError, 'entry exceeds'):
            c._enter_yellow('test', 'search[cup]', 'third')

    def test_invalid_checkpoint_id_has_no_root_default(self):
        c, _, _ = make_controller(['search[cup]', 'click[A]', 'click[Buy Now]',
                                  'Choose red.', 'ACTION 99'])
        with self.assertRaisesRegex(ProtocolError, 'unlisted executed action'):
            c.run()
        self.assertEqual(c.checkpoint_restore_count, 0)

    def test_runner_never_loads_weights_or_reseeds_model(self):
        import ast
        module = ast.parse((PACKAGE/'source/web_v16_run.py').read_text())
        called = [ast.unparse(node.func) for node in ast.walk(module) if isinstance(node, ast.Call)]
        self.assertFalse(any('AutoModel' in name or 'manual_seed' in name or name.endswith('.seed') for name in called))
        self.assertIn('load_model_bundle', called)
        backend = ast.parse((PACKAGE/'source/model_backend.py').read_text())
        backend_calls = [ast.unparse(n.func) for n in ast.walk(backend) if isinstance(n, ast.Call)]
        self.assertFalse(any('AutoModel' in name or 'manual_seed' in name for name in backend_calls))

    @unittest.skipUnless(os.environ.get('STAR_TRAJECTORY_FIXTURES'), 'historical trajectories are intentionally external to the code-only export')
    def test_real_r52_repeated_search_and_product_are_not_new_evidence(self):
        rows = [json.loads(line) for line in (Path(os.environ['STAR_TRAJECTORY_FIXTURES'])/'r5_2_task004_events.jsonl').read_text(encoding='utf-8').splitlines()]
        initial, = [r['initial_receipt'] for r in rows if r['event']=='scoped_recovery_started']
        actions = [r for r in rows if r['event']=='scoped_action_executed']
        # Public post-rollback segment: Back, search, product, Back, same search,
        # same product. Terminal scores are retained as evidence, not goal labels.
        suffix = actions[3:]
        history = []
        for row in suffix[:4]:
            history.append(row['receipt'])
        repeated_search = suffix[4]['receipt']
        self.assertTrue(repeated_search['action'].startswith('search['))
        self.assertFalse(new_evidence(repeated_search, history, initial))
        history.append(repeated_search)
        self.assertFalse(new_evidence(suffix[5]['receipt'], history, initial))
        transcript = '\n'.join('Action: '+r['action']+'\nObservation: '+r['receipt']['observation'] for r in actions)
        prompt = author_reflection(initial['observation']+'\n'+transcript)
        self.assertIn('Your score (min 0.0, max 1.0): 0.5', prompt)
        self.assertIn('New plan:', prompt)

    @unittest.skipUnless(os.environ.get('STAR_TRAJECTORY_FIXTURES'), 'historical trajectories are intentionally external to the code-only export')
    def test_real_r52_option_selection_is_public_change(self):
        rows = [json.loads(line) for line in (Path(os.environ['STAR_TRAJECTORY_FIXTURES'])/'r5_2_task006_events.jsonl').read_text(encoding='utf-8').splitlines()]
        initial, = [r['initial_receipt'] for r in rows if r['event']=='scoped_recovery_started']
        actions = [r['receipt'] for r in rows if r['event']=='scoped_action_executed']
        self.assertEqual(actions[2]['action'], 'click[redwood]')
        self.assertTrue(new_evidence(actions[2], actions[:2], initial))
        self.assertNotEqual(actions[2]['before']['selected_options'], actions[2]['after']['selected_options'])

    def test_runtime_rejects_wrong_step_and_backend(self):
        b = dict(parent_job_id='10', step_id='2', hostname='gpu-test', gpu=0)
        env = dict(SLURM_JOB_ID='10', SLURM_STEP_ID='2', CUDA_VISIBLE_DEVICES='0')
        verify_runtime(b, env, 'gpu-test')
        with self.assertRaises(ProtocolError):
            verify_runtime(b, dict(env, SLURM_STEP_ID='3'), 'gpu-test')
        record = dict(parent_job_id='10', step_id='2', hostname='gpu-test',
            interface='table1_webshop_attributes_v1', initialization_seed=1, show_attributes=True)
        verify_backend(record, b)
        with self.assertRaises(ProtocolError):
            verify_backend(dict(record, show_attributes=False), b)


if __name__ == '__main__':
    unittest.main()
