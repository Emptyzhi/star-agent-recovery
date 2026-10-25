# External dependencies

This repository publishes the recovery framework, our benchmark adapters,
adaptation edits, and anonymized evaluation records. It does not redistribute
the upstream implementations, their demonstrations, or the website source.

`upstreams.json` records the exact repository URL and commit for each external
dependency. `fetch_upstreams.py` retrieves those revisions into an ignored local
cache and materializes the required files outside Git tracking. The upstream
license files remain with locally obtained dependencies. No permission to
redistribute an external project is granted by this repository.

| Dependency | Pinned revision | Used for |
| --- | --- | --- |
| GA-Rollback | `f6fb280d6c5a03a515e5a7db213585ccd9b0f068` | Native ReAct, Reflexion, and GA controllers; task demonstrations and configuration |
| Tree of Thoughts | `8050e67d0e3a0fddc424d7fa5801538722a4c4cc` | Search implementation |
| Language Agent Tree Search | `853d81614607dd27433faf17c7b0a7d660f95d22` | Search implementation |
| AgentRewind library | `6661046e2b30fcf5f17c93d51acb56babdea8c53` | Recording, checkpoint tools, and recovery memory |
| WebShop | `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` | Website source used as the reconstruction base for compatibility edits |

The WebShop revision identifies the base used to reconstruct the released
compatibility layer, not an assertion about the original installation's Git
history. Each edit recipe checks its base and resulting content. Only inserted
or replacement lines are stored here, not unchanged website code.

Some upstream repositories do not declare a redistribution license. Obtain
access under their applicable terms and request permission for redistribution
when necessary. The acquisition helper is not a grant of permission.

Benchmark data, model weights, search indexes, and external environment
installations must be obtained separately. Model and benchmark names identify
evaluated resources and do not identify the submission's authors.
