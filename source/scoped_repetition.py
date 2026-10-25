"""Public-evidence repetition incidents; no action blacklist or hidden reward."""

import hashlib
from tree_history import canonical


def operation_key(state):
    return canonical({k: v for k, v in state.items() if k != "observation"})


class RepetitionEvidence:
    def __init__(self):
        self.observations = set()
        self.revision = 0
        self.transitions = []
        self.root_attempt_start = 0

    def begin_root_attempt(self):
        # Prior attempts remain public evidence, not an immediate repeat trigger.
        self.root_attempt_start = len(self.transitions)

    def record_initial(self, receipt):
        if self.observations or self.transitions:
            raise ValueError('initial public evidence must be recorded before execution')
        evidence=canonical({'state':operation_key(receipt['after']),
                            'observation':receipt['observation']})
        self.observations.add(evidence)
        self.revision=1

    def record(self, occurrence):
        receipt = occurrence.receipt
        evidence = canonical({"state": operation_key(receipt["after"]),
                              "observation": receipt["observation"]})
        if evidence not in self.observations:
            self.observations.add(evidence)
            self.revision += 1
        self.transitions.append({"before": operation_key(receipt["before"]),
            "after": operation_key(receipt["after"]), "action": occurrence.action,
            "accepted": receipt["native_accepted"], "done": receipt["done"],
            "occurrence_id": occurrence.occurrence_id, "revision": self.revision})

    def repeated(self, state, action):
        current = operation_key(state)
        # A Root boundary is a controller transition, not new public evidence.
        # Repeated state/action decisions must remain visible across attempts.
        for index in range(len(self.transitions) - 1, -1, -1):
            previous = self.transitions[index]
            if (previous["before"] == current and previous["action"] == action
                    and previous["accepted"] and not previous["done"]):
                # New observations elsewhere do not change this exact ALFWorld
                # decision state. A changed local public state does.
                segment = self.transitions[index:]
                region = sorted({current, *(e["before"] for e in segment),
                                 *(e["after"] for e in segment)})
                signature = canonical({"entry_state": current, "action": action,
                                       "region": region})
                return {"incident_id": "repeat-" + hashlib.sha256(signature.encode()).hexdigest()[:16],
                        "kind": "REPEATED_DECISION_WITHOUT_NEW_PUBLIC_EVIDENCE",
                        "action": action, "region": region,
                        "observed_actions": [e["action"] for e in segment],
                        "observed_occurrence_ids": [e["occurrence_id"] for e in segment],
                        "evidence_revision": self.revision}
        return None
