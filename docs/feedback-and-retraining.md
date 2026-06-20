# Feedback & Retraining

## Post-Event Feedback Collection

The system includes a feedback collection module (`dashboard/services/feedback_store.py`) that stores, after an event has concluded:

```text
actual duration
actual officers deployed
actual barricades used
actual road closure
actual incident count
officer notes
```

Output file: `data/post_event_feedback.csv`

## Current Scope

**Important honesty note:** the current prototype stores feedback for audit, analysis, and future retraining. It does **not** automatically retrain the ML model after each feedback entry.

### Correct wording to use when describing this feature

```text
feedback collection system with retraining capability planned
```

### Wording to avoid

```text
automatic learning loop
self-learning model
continuous retraining
```

This distinction matters for accurately describing the system to reviewers, collaborators, or in any documentation — overstating this capability is the easiest way to lose credibility on an otherwise well-evidenced system.

## How This Data Could Be Used Later

The collected feedback is structured so that it *could* support:

- scheduled (not automatic) retraining of the forecasting model
- replacing the current EIS severity proxy (see [Event Impact Scoring](event-impact-scoring.md#severity-proxy)) with real officer-labelled outcomes
- auditing how accurate past risk/EIS predictions were against what actually happened

None of this is implemented yet — see [Limitations & Roadmap](limitations-and-roadmap.md) for the current state and planned direction.

## Related Docs

- [Event Impact Scoring](event-impact-scoring.md) — the severity proxy this feedback could eventually replace
- [Limitations & Roadmap](limitations-and-roadmap.md) — full list of what's not yet implemented