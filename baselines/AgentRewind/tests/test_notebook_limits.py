"""Explicit notebook rendering limits preserve upstream records and guidance."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment, ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.notebook import notebook_override
from agentrewind_bench.native_protocol import without_runtime_allowance
from replay.backtrack import backtrack_memory_item, write_backtrack_memory_notebook


LONG_NOTE = ('failed_check: observed incorrect state.\nevidence_observed: '
             + ' '.join('detail_%03d' % i for i in range(150))
             + '\ncorrected_strategy: choose good after checking the current state.'
             + '\nfirst_3_tool_calls: good'
             + '\nsuccess_invariant: current value is two.')


class NotebookLimitsTests(unittest.TestCase):
    def settings(self, *extra):
        return fixtures.AdaptationTests().config(extra=[
            '--history-format', 'dialogue', '--recovery-format', 'lines', *extra])

    def prepared(self, directory, prior_notes=(), note=LONG_NOTE):
        log_dir = Path(directory) / 'author_traces'
        log_dir.mkdir()
        source = 'notebook_source'
        writer = author.JsonlWriter(author.run_path(log_dir, source), overwrite=False)
        try:
            writer.write(dict(kind='llm', record_uid='rec_000001',
                input=dict(messages=[dict(role='system', content=author.instruction('lines')),
                                     dict(role='user', content='Task and observation\nAct 1>')],
                           model='fixture', max_tokens=100),
                output=dict(content='bad')))
        finally:
            writer.close()
        if prior_notes:
            items = [backtrack_memory_item(from_run=source, request=author.BacktrackRequested(
                record_uid='rec_000001', memory_summary=note, run_id=source, log_dir=log_dir))
                for note in prior_notes]
            write_backtrack_memory_notebook(log_dir=log_dir, run_id=source, memory_items=items)
        request = author.BacktrackRequested(record_uid='rec_000001', memory_summary=note,
                                            run_id=source, log_dir=log_dir)
        prepared = author.prepare_backtrack_replay(request, current_run_id=source, log_dir=log_dir,
            fork_run='notebook_fork', memory_limit=20, memory_policy='cumulative',
            context_policy='rollback', workspace_policy='rollback')
        return prepared, log_dir

    def render(self, prepared, memory, item_limit):
        return author.render_backtrack_memory_notebook(memory, limit=20, item_char_limit=item_limit,
            notebook_char_limit=8000, context_policy='rollback', workspace_policy='rollback',
            target_record_uid=prepared['breakpoint_record_uid'],
            context_source_record_uid=prepared['backtrack_context_source_record_uid'])

    def test_cli_default_and_explicit_variant_are_recorded(self):
        original = self.settings()
        self.assertEqual(original.notebook_item_char_limit, 900)
        self.assertEqual(original.as_dict()['adaptation_protocol'], 'native_act_only_v20_author_tools_1200')
        selected = self.settings('--notebook-item-char-limit', '4000')
        self.assertEqual(selected.notebook_item_char_limit, 4000)
        self.assertEqual(selected.as_dict()['adaptation_protocol'],
                         'native_act_only_v20_author_tools_1200')
        self.assertEqual(selected.as_dict()['notebook_limits'],
                         dict(items=20, item_characters=4000, total_characters=8000))
        self.assertEqual(selected.as_dict()['notebook_item_char_limit'], 4000)
        self.assertEqual((selected.actor_output_tokens, selected.max_output_tokens), (100, 1200))

    def test_invalid_limits_and_incompatible_transport_are_explicit_errors(self):
        original = self.settings()
        for value in (0, -1, 901, 8001, True, '4000', 4000.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(original, notebook_item_char_limit=value)
        with self.assertRaises(ValueError):
            replace(original, notebook_item_char_limit=4000, recovery_format='python')
        for value in ('0', '-1', '4001', 'not-a-number'):
            with self.subTest(cli=value), self.assertRaises(SystemExit):
                self.settings('--notebook-item-char-limit', value)

    def test_default_900_uses_original_file_without_rerender_or_effective_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, log_dir = self.prepared(directory)
            original_path = prepared['override_input_file']
            original_bytes = original_path.read_bytes()
            with patch('agentrewind_bench.notebook.author.render_backtrack_memory_notebook') as renderer:
                effective, config = notebook_override(prepared, log_dir, item_char_limit=900, memory_limit=20)
            renderer.assert_not_called()
            self.assertEqual(effective, original_path)
            self.assertEqual(original_path.read_bytes(), original_bytes)
            self.assertEqual(config, dict(items=20, item_char_limit=900, notebook_char_limit=8000))
            self.assertEqual(list(log_dir.glob('*.effective.json')), [])

    def test_4000_changes_only_unique_notebook_and_keeps_complete_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, log_dir = self.prepared(directory)
            original_path = prepared['override_input_file']
            original_bytes = original_path.read_bytes()
            memory_path = log_dir / (prepared['fork_run'] + '.backtrack_memory.jsonl')
            memory_bytes = memory_path.read_bytes()
            memory = author.load_backtrack_memory(log_dir, prepared['fork_run'])
            effective_path, config = notebook_override(prepared, log_dir, item_char_limit=4000, memory_limit=20)
            original = json.loads(original_bytes)
            effective = json.loads(effective_path.read_text(encoding='utf-8'))
            indices = [i for i, message in enumerate(original['messages'])
                       if author.is_backtrack_memory_message(message)]
            self.assertEqual(len(indices), 1)
            index = indices[0]
            self.assertNotIn('corrected_strategy', original['messages'][index]['content'])
            self.assertNotIn('first_3_tool_calls', original['messages'][index]['content'])
            self.assertIn('corrected_strategy: choose good', effective['messages'][index]['content'])
            self.assertIn('first_3_tool_calls: good', effective['messages'][index]['content'])
            self.assertEqual(effective['messages'][index]['content'], self.render(prepared, memory, 4000))
            expected = deepcopy(original)
            expected['messages'][index]['content'] = self.render(prepared, memory, 4000)
            self.assertEqual(effective, expected)
            self.assertEqual(original_path.read_bytes(), original_bytes)
            self.assertEqual(memory_path.read_bytes(), memory_bytes)
            self.assertEqual(memory[0]['memory_summary'], LONG_NOTE)
            self.assertEqual(config, dict(items=20, item_char_limit=4000, notebook_char_limit=8000))
            self.assertEqual(effective['messages'][0]['content'], author.instruction('lines'))

    def test_total_8000_marker_and_twenty_item_selection_use_author_renderer(self):
        notes = [f'NOTE_{i:02d}: ' + ('observed evidence ' * 180) for i in range(21)]
        with tempfile.TemporaryDirectory() as directory:
            prepared, log_dir = self.prepared(directory, notes)
            memory = author.load_backtrack_memory(log_dir, prepared['fork_run'])
            memory_before = deepcopy(memory)
            effective_path, _ = notebook_override(prepared, log_dir, item_char_limit=4000, memory_limit=20)
            effective = json.loads(effective_path.read_text(encoding='utf-8'))
            notebook = next(m['content'] for m in effective['messages'] if author.is_backtrack_memory_message(m))
            self.assertEqual(notebook, self.render(prepared, memory, 4000))
            marker = '\n...[backtrack memory truncated]'
            self.assertTrue(notebook.endswith(marker))
            self.assertLessEqual(len(notebook), 8000 + len(marker))
            self.assertNotIn('NOTE_00:', notebook)
            self.assertNotIn('NOTE_01:', notebook)
            self.assertIn('NOTE_02:', notebook)
            self.assertEqual(len(memory), 22)
            self.assertEqual(author.load_backtrack_memory(log_dir, prepared['fork_run']), memory_before)

    def test_short_note_with_missing_strategy_is_not_completed_or_reordered(self):
        note = 'failed_check: state differs.\nfirst_3_tool_calls: good'
        with tempfile.TemporaryDirectory() as directory:
            prepared, log_dir = self.prepared(directory, note=note)
            effective_path, _ = notebook_override(prepared, log_dir, item_char_limit=4000, memory_limit=20)
            effective = json.loads(effective_path.read_text(encoding='utf-8'))
            notebook = next(m['content'] for m in effective['messages'] if author.is_backtrack_memory_message(m))
            self.assertNotIn('corrected_strategy', notebook)
            self.assertIn('failed_check: state differs. first_3_tool_calls: good', notebook)
            self.assertEqual(author.load_backtrack_memory(log_dir, prepared['fork_run'])[0]['memory_summary'], note)

    def test_missing_or_duplicate_notebook_is_rejected_not_repaired(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                prepared, log_dir = self.prepared(directory)
                path = prepared['override_input_file']
                original = json.loads(path.read_text(encoding='utf-8'))
                notebook = next(m for m in original['messages'] if author.is_backtrack_memory_message(m))
                if duplicate:
                    original['messages'].append(deepcopy(notebook))
                else:
                    original['messages'].remove(notebook)
                path.write_text(json.dumps(original), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'exactly one notebook'):
                    notebook_override(prepared, log_dir, item_char_limit=4000, memory_limit=20)
                self.assertEqual(list(log_dir.glob('*.effective.json')), [])

    def test_modified_original_notebook_is_rejected_without_an_effective_override(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, log_dir = self.prepared(directory)
            path = prepared['override_input_file']
            original = json.loads(path.read_text(encoding='utf-8'))
            notebook = next(m for m in original['messages'] if author.is_backtrack_memory_message(m))
            notebook['content'] += '\nAltered notebook evidence'
            path.write_text(json.dumps(original), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the complete fork memory'):
                notebook_override(prepared, log_dir, item_char_limit=4000, memory_limit=20)
            self.assertEqual(list(log_dir.glob('*.effective.json')), [])

    def test_controller_uses_effective_override_and_records_both_paths(self):
        for item_limit in (900, 4000):
            with self.subTest(item_limit=item_limit), tempfile.TemporaryDirectory() as directory:
                settings = self.settings('--notebook-item-char-limit', str(item_limit))
                model = ScriptedModel(['bad', 'backtrack_candidates[Observed failure]',
                    'backtrack_commit[rec_000001]\n' + LONG_NOTE, 'good'])
                result = Controller(CounterEnvironment(), model, settings, directory, 'notebook_case').run()
                events = [json.loads(line) for line in (Path(directory) / 'events.jsonl').read_text().splitlines()]
                restored = next(e for e in events if e['event'] == 'checkpoint_restored')
                original_path = Path(restored['original_override_input_file'])
                effective_path = Path(restored['effective_override_input_file'])
                self.assertEqual(original_path == effective_path, item_limit == 900)
                effective = json.loads(effective_path.read_text(encoding='utf-8'))
                self.assertEqual(without_runtime_allowance(model.requests[-1]),
                                 without_runtime_allowance(effective['messages']))
                self.assertEqual(restored['notebook_render_config'],
                    dict(items=20, item_char_limit=item_limit, notebook_char_limit=8000))
                memory = author.load_backtrack_memory(Path(directory) / 'author_traces', restored['trace'])
                self.assertEqual(memory[0]['memory_summary'], LONG_NOTE)
                for messages in model.requests:
                    self.assertEqual(sum(m['role'] == 'system' and m['content'] == author.instruction('lines')
                                         for m in messages), 1)
                self.assertEqual((result['success'], result['completed_recoveries'],
                                  result['cost']['model_requests'], result['executed_actions']), (True, 1, 4, 2))

    def test_4000_keeps_real_author_exception_schema_and_rejected_note_out_of_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings('--notebook-item-char-limit', '4000')
            rejected = 'backtrack_commit[rec_000001]\nThe task is not feasible.'
            model = ScriptedModel(['bad', 'backtrack_candidates[Observed failure]', rejected,
                'backtrack_commit[rec_000001]\n' + LONG_NOTE, 'good'])
            result = Controller(CounterEnvironment(), model, settings, directory, 'notebook_error').run()
            log_dir = Path(directory) / 'author_traces'
            records = [json.loads(line) for line in (log_dir / 'notebook_error_r0.jsonl').read_text().splitlines()]
            error = next(r for r in records if r['kind'] == 'tool' and r['error'] is not None)
            self.assertIsNone(error['output'])
            self.assertEqual(set(error['error']), {'type', 'message', 'repr'})
            self.assertEqual(error['error']['type'], 'BacktrackError')
            self.assertIn('memory_summary rejected:', error['error']['message'])
            stored = author.load_backtrack_memory(log_dir, 'notebook_error_r1')
            self.assertEqual([item['memory_summary'] for item in stored], [LONG_NOTE])
            self.assertTrue(result['success'])
            self.assertEqual(result['cost']['model_requests'], 5)


if __name__ == '__main__':
    unittest.main()
