# SPDX-License-Identifier: Apache-2.0
"""logwatch — small-LLM log triage harness.

Map/reduce over logs: a stateless small model classifies one chunk per call
(map), plain Python aggregates verdicts into findings (reduce), findings
become tracker issues, and issue outcomes feed back into training data.
"""
