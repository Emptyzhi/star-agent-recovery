from types import SimpleNamespace
import unittest
from icr_output_framing import frame_complete
from star_vllm_framing import apply_boundary


class LabelStopTests(unittest.TestCase):
    def test_complete_label_stops_before_same_line_explanation(self):
        for parts in [[' im','possible',' # explanation'],[' like','ly',' because'],[' sure',' # extra']]:
            tokens=[]
            sequence=SimpleNamespace(status='RUNNING',get_output_token_ids=lambda:tokens)
            pending=SimpleNamespace(request={'stop_mode':'tot_label'},submitted=0,stop_reason=None)
            for i in range(len(parts)):
                tokens.append(i)
                reason=apply_boundary(sequence,pending,lambda ids:''.join(parts[j] for j in ids),[],1,'STOPPED')
                if reason is not None:
                    break
            self.assertEqual(tokens,list(range(len(parts)-1)))
            self.assertEqual(pending.stop_reason,'frame')
            self.assertNotIn('#',sequence.output_text)

    def test_no_posthoc_repair_or_label_search(self):
        for text in ['im','like','Certainly sure','likely # bundled malformed suffix','surely']:
            self.assertFalse(frame_complete(text,'tot_label'))
        self.assertTrue(frame_complete('\n','tot_label'))
        self.assertTrue(frame_complete(' impossible','tot_label'))
        self.assertFalse(frame_complete(' sure','first_line'))


if __name__=='__main__':
    unittest.main()
