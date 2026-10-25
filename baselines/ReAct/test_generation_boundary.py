"""Verify role-specific request framing and the service's token-boundary stop."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from resident_transport import ROLE_STOP_MODES, ResidentTransport
from star_vllm_framing import apply_boundary


class RequestReachedService(Exception):
    pass


class BoundaryTests(unittest.TestCase):
    def test_actual_wire_payload_frames_actions_but_not_analysis(self):
        for role, expected in ROLE_STOP_MODES.items():
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                session = Mock()
                session.post.side_effect = RequestReachedService
                transport = ResidentTransport('http://resident','test',{}, {},Path(directory),session=session)
                transport.role = role
                with self.assertRaises(RequestReachedService):
                    transport.generate('exact native prompt',100)
                payload = session.post.call_args.kwargs['json']
                self.assertEqual(payload['prompt'],'exact native prompt')
                self.assertEqual(payload['stop_mode'],expected)
                self.assertEqual(payload['max_new_tokens'],100)
                self.assertNotIn('allowed_commands',payload)

    def test_token_loop_stops_before_sampling_observation(self):
        for mode, expected_count in [('first_line',3),(None,5)]:
            tokens=[]
            seq=SimpleNamespace(status='RUNNING',get_output_token_ids=lambda:tokens)
            pending=SimpleNamespace(request={'stop_mode':mode},submitted=0,stop_reason=None)
            vocabulary={1:'search[',2:'blue cup]',3:'\n',4:'Observation:',5:' invented receipt'}
            for token in [1,2,3,4,5]:
                tokens.append(token)
                reason=apply_boundary(seq,pending,lambda ids:''.join(vocabulary[i] for i in ids),
                                      [99],1,'STOPPED')
                if reason is not None:
                    break
            self.assertEqual(len(tokens),expected_count)
            if mode == 'first_line':
                self.assertEqual(seq.output_text,'search[blue cup]\n')
                self.assertEqual(pending.stop_reason,'frame')

    def test_boundary_preserves_raw_text_including_leading_newline(self):
        seq=SimpleNamespace(get_output_token_ids=lambda:[1],status='RUNNING')
        pending=SimpleNamespace(request={'stop_mode':'first_line'},submitted=0,stop_reason=None)
        apply_boundary(seq,pending,lambda ids:'\n', [99],1,'STOPPED')
        self.assertEqual(seq.output_text,'\n')
        self.assertEqual(seq.status,'STOPPED')

    def test_undeclared_role_is_rejected_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            session=Mock()
            transport=ResidentTransport('http://resident','test',{}, {},Path(directory),session=session)
            transport.role='unspecified'
            with self.assertRaises(ValueError):
                transport.generate('P',100)
            session.post.assert_not_called()


if __name__=='__main__':
    unittest.main()
