# Issue Working Document Template


<!--
ISSUE WORKING DOCUMENT (IWD) — template v3.
-->

# <ISSUE-KEY> — <issue title>

| Field | Value |
|-------|--------|
| **Issue** | [<ISSUE-KEY>](<tracker URL>) |
| **Branch** | `feature/<ISSUE-KEY>-<slug>` |
| **Author** | <name> |
| **Status** | Drafting |
| **Scope Tier** | <T0 / T1 / T2 / T3 — see process §2> |
| **QA Level** | Low |
| **Estimate** | <n> days |
| **Created / Updated** | <YYYY-MM-DD> / <YYYY-MM-DD> |

---

## 1. Abstract
Summarize the problem and proposed solution. Required for every tier; 3–5 sentences for T2, shorter for T1 (process §2.3).
Prompts:
- What problem is being solved?
- What is the proposed approach?
- Who benefits and how?

### 1.1 Scope
Written by the developer, by hand, before anything else (process §5.1) — restating scope is how you find out whether you understand it.

**In scope.**
-

**Out of scope.**
<!-- The tempting adjacent things you are deliberately NOT doing. Nothing in later
     sections may fall in this list — an agent that drifts here should be stopped. -->
-

**Done when.**
<!-- One sentence. -->

---

## 2. Concept of Operations
Written first by the developer before AI elaboration.
Prompts:
- Operational scenarios
- Actors and workflows
- Intended system behavior
- Assumptions and constraints
- Changes relative to current behavior

---

## 3. External Requirements and Design Evaluation Criteria
Guidance (T2): at most 8 requirements (process §2.3).

### 3.1 External Requirements
Atomic, verifiable success criteria. Each requirement must be one that a test can demonstrate — if you cannot say how it would be tested, it is not yet stated precisely enough. The verification method itself is specified in Architecture and Design (§6), not here; this section states *what* must hold, §6 states *how it will be shown to hold*.
Prompts:
- Functional requirements
- Performance requirements
- Interface/API requirements
- Environmental or mission constraints

### 3.2 Design Evaluation Criteria
Omit this subsection when only one design is plausible for this issue — that is expected for most T1/T2 issues (process §2.3, §2.4). When this subsection is omitted, the Frame and Design phases are normally performed in a single combined session (process §4, §5.2, §5.3).
Prompts:
- Criteria for comparing alternative designs
- Trade-off principles
- Constraints that shape acceptable solutions

---

## 4. Context
Populated during Exploration and updated in phases with exploration autonomy only when findings meet the materiality criteria of process §3.2. Guidance (T2): at most 15 bullets — a bullet that does not constrain a Requirement or Design decision should be omitted (process §2.4).
Prompts:
- Relevant architecture elements
- Existing modules and APIs
- Constraints discovered during exploration
- Missing information added by agents
- External dependencies or interactions

---

## 5. Critical Design Decisions (when applicable)
List key decisions where **genuine alternatives** exist. Guidance (T2): at most 3 decisions. Omit this section when no genuine alternative exists; a single plausible option belongs in Architecture and Design (§6) as a design fact, not here (process §2.3, §2.4).
Prompts:
- Decision question
- Options considered
- Final choice and rationale
- Risks and alternatives

---

## 6. Architecture and Design
Provide the traditional, human-approved implementation contract, properly scaled to the scope and risk of the issue. One page is a target for ordinary T2 issues; additional detail is justified when it improves implementation or human understanding. Describe the delta, not existing behavior already captured in Context.

The design should include, as applicable: design summary; affected components and responsibilities; control and data flow; interfaces and data; behavior and invariants; implementation outline; risks, rollout, and migration; and the implementation decision envelope. Genuine consequential alternatives and their human rationale may be recorded in §5 or here.

**Approval.** For T2, the developer shall record `Approved by <name> on <date>` before Implementation begins; a design drafted by hand (without agent involvement) may be self-approved and the checkbox below serves as the record. For T1, approval may be recorded after implementation has begun, provided it is present before merge. A material implementation deviation requires a design amendment and renewed approval.

- [ ] Design reviewed and approved by _______________ on _______________ .

**Every External Requirement (§3.1) must be mapped here to a verification method** — the test (unit, integration, or end-to-end) that will demonstrate it, or, where automated testing is impractical, the explicit manual-check steps that will. This mapping is part of the design, not an afterthought for Implementation. For T1, a one-line note per requirement suffices; for T2, use a short table, e.g.:

| Requirement | Verification |
|---|---|
| <requirement id/summary> | <unit test name/description, or manual-check steps> |

Prompts:
- Components, interfaces, algorithms
- Data structures and workflow descriptions
- Preconditions, invariants, and postconditions
- IEEE 1471-2000 style "views"
- UML or traditional software engineering diagrams in mermaid
- Examples or doctests where helpful
- Requirement-to-test mapping (required — see above)

---

## 7. Acceptance Criteria and Evidence
Testable criteria confirming completion. Required whenever an IWD exists (T1 and above — T0 has no IWD, see process §2). Guidance (T2): at most 8 criteria (process §2.3).

Each criterion must be demonstrated by an actual test result, not merely asserted. Record the evidence here or in the requirement-to-verification table in §6; do not duplicate the same criterion merely to preserve template symmetry.

Prompts:
- Required behaviors (covering all "External Requirements"), each with a cited passing test or recorded manual-check result
- Edge cases
- API expectations
- Performance thresholds
- Non-functional criteria

---

## 8. Open Questions
Gaps an agent could not resolve, raised here rather than answered by guessing or resolved only in conversation (process §3.4). Tactical questions that do not affect architecture, external requirements, or acceptance criteria may be resolved by the agent and noted in the Implementation Notes. Write `None.` if there are no outstanding blocking questions. Resolve each with a developer answer, or promote it to a Critical Design Decision (§5) if it turns out to be a genuine trade-off.

**Q-1.** <question>
- *Impact:* <what is blocked until this is answered>
- *Answer:* <developer's answer, or "→ see §5">

---

## 9. Notes, Risks, and Future Considerations (Optional)
Prompts:
- Known risks and ambiguities
- Expected extensions
- Dependencies on other issues
- Assumptions that may change

---

## 10. Change Log (Optional)
Also the record of anything promoted to project documentation during Documentation (process §5.6) — one line per item, naming the target file.
Prompts:
- Date / author / summary of changes
- Harvested items: target file and short description

---

## Implementation Notes

The implementation agent shall provide this section at closeout:

- Design steps completed:
- Files and symbols changed:
- Tests added or updated and results:
- Outcome evidence:
- Approved design amendments:
- Unresolved deviations:

## Definition of Done

- [ ] Scope (§1.1) respected — nothing implemented from the out-of-scope list.
- [ ] Architecture and Design (§6) approved; for T2 before implementation, for T1 before merge.
- [ ] Every External Requirement (§3.1) has a passing test or a recorded manual check (§6, §7).
- [ ] Material deviations from Architecture and Design (§6) are approved and recorded.
- [ ] A targeted diff review was completed; detailed review was performed for applicable risk triggers.
- [ ] Durable content has been promoted to project documentation where useful and logged in the Change Log (§10).
- [ ] CI is green; the change has been reviewed per the review discipline (process §4.1).


