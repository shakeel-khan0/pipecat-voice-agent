# VoiceMem Integration Implementation Plan

## Goal

Integrate the already-tested standalone VoiceMem implementation into the existing `pipecat-voice-agent` without breaking the current voice, RAG, LangGraph, Calendar, Sheets, or interruption behavior.

Final target:

```text
Caller / Terminal User
        |
        v
Deepgram Flux STT + Silero VAD
        |
        v
Final user transcript
        |
        +-------------------------+
        |                         |
        v                         v
Agentix RAG                  VoiceMem
Company knowledge            Caller memory
Qdrant hybrid retrieval      Left + Right Brain
        |                         |
        +------------+------------+
                     |
                     v
             Temporary context
                     |
                     v
                  Groq LLM
                     |
          +----------+----------+
          |                     |
          v                     v
     Normal reply          LangGraph
                           Booking workflow
                              |
                    Calendar / Sheets
                              |
                              v
                         Final response
                              |
                              v
                        Deepgram TTS
```

## Non-Negotiable Rules

- Do not copy the VoiceMem repository into the voice-agent source tree.
- Keep VoiceMem behind a small adapter/service layer.
- Do not change the existing Agentix RAG retrieval logic unless integration requires a minimal orchestration hook.
- Do not change LangGraph booking behavior.
- Do not change Deepgram STT/TTS, Silero VAD, barge-in, Calendar, or Sheets behavior.
- VoiceMem failure must never stop the voice agent.
- RAG failure must never stop the voice agent.
- Caller memories must be isolated by a stable `caller_id`.
- Never identify a caller by name alone.
- For terminal testing, use a manual stable test caller ID.
- Later, replace the terminal test ID with a stable ID derived from the incoming phone number.
- Do not log API keys, raw secrets, or unnecessary sensitive caller data.
- Retrieved VoiceMem context must be temporary inference context, not permanently appended to conversation history.
- Existing Agentix RAG remains the source of truth for company facts.
- VoiceMem remains the source of truth for caller-specific long-term memory.
- Groq is the LLM/reasoning provider. Memory is not stored inside Groq.
- Preserve the currently working Groq + local E5 VoiceMem configuration.
- No OpenAI service dependency should be introduced.

---

# Phase 1 - Safe Package Integration and Memory Adapter

## Objective

Make the existing voice-agent project able to initialize the tested VoiceMem implementation safely, without yet changing live conversation behavior.

## Work

1. Inspect dependency compatibility between:
   - existing `.pipevenv`
   - current Pipecat dependencies
   - modified VoiceMem package
   - local E5 dependencies

2. Install the modified VoiceMem repository into `.pipevenv` using an editable/local install only if dependency resolution is safe.

3. Create a dedicated memory module, for example:

```text
memory/
├── __init__.py
├── voicemem_adapter.py
└── caller_identity.py
```

4. `voicemem_adapter.py` should expose a minimal stable interface such as:

```python
initialize()
retrieve(caller_id, query)
ingest(caller_id, messages)
health()
close()
```

Exact signatures may differ after inspecting the actual VoiceMem API.

5. Add terminal caller identity support:

```env
TEST_CALLER_ID=terminal_test_001
```

6. `caller_identity.py` must:
   - accept the temporary terminal ID now
   - expose one stable function that returns the active caller ID
   - be designed so Asterisk/phone-number identity can replace the source later without changing VoiceMem integration code

7. Do not yet inject VoiceMem into Groq context.
8. Do not yet write live conversation turns into VoiceMem.

## Failure Handling

VoiceMem initialization must be fail-open:

```text
VoiceMem initializes
    |
    +-- success -> memory_enabled = true
    |
    +-- failure -> log warning
                  memory_enabled = false
                  voice agent continues normally
```

Never crash `main.py` because VoiceMem is unavailable.

## Acceptance Criteria

- Existing voice agent starts normally.
- Existing RAG still works.
- Existing booking still works.
- VoiceMem adapter can initialize independently.
- `TEST_CALLER_ID` resolves consistently.
- No memory is yet injected into the live agent.
- Existing tests remain green.

## Codex Prompt 1

```text
Integrate the already-tested modified VoiceMem package into my existing pipecat-voice-agent project at the infrastructure level only.

STRICT:
- First inspect both repositories and dependency versions.
- Do not modify the separate VoiceMem repository unless absolutely required for compatibility.
- Do not change Pipecat behavior, Deepgram STT/TTS, Silero VAD, barge-in, RAG retrieval, LangGraph, Calendar, or Sheets.
- Do not add OpenAI service usage.
- Do not hardcode secrets.
- Do not inject VoiceMem into the live LLM context yet.
- Do not ingest live conversations yet.
- VoiceMem failure must never prevent main.py from starting.

Current VoiceMem state:
- Left Brain standalone works.
- Right Brain standalone works.
- Groq is used for LLM/fact extraction.
- Local E5 embeddings work.
- Persistence across processes works.
- No OpenAI service is required.

Tasks:
1. Inspect dependency compatibility with the existing .pipevenv.
2. Use the modified local VoiceMem package through the cleanest local/editable package mechanism.
3. Create a small memory integration layer such as:
   memory/voicemem_adapter.py
   memory/caller_identity.py
4. Keep VoiceMem internals behind the adapter.
5. Add terminal identity support using TEST_CALLER_ID with a safe default only for local development.
6. Design caller_identity.py so the identity source can later change from TEST_CALLER_ID to normalized phone-number-based caller ID without changing the rest of the memory code.
7. Add fail-open initialization:
   - initialization failure -> warning + memory disabled
   - main voice agent continues
8. Add focused unit/smoke tests.
9. Run existing relevant tests to ensure there is no regression.

At the end report:
- files changed
- dependency compatibility result
- how VoiceMem is imported/initialized
- adapter interface
- caller ID behavior
- failure behavior
- tests run/results

Stop before live memory retrieval or ingestion.
```

---

# Phase 2 - VoiceMem Read Path

## Objective

Retrieve existing caller memory during the real voice-agent conversation and provide only relevant memory to Groq.

No live memory writes yet.

## Read Flow

```text
Final transcript
      |
      v
caller_id
      |
      v
VoiceMem retrieve
      |
      v
Relevant Left/Right Brain memories
      |
      v
Temporary inference context
      |
      v
Groq response
```

## Routing

VoiceMem should not necessarily run for every trivial turn.

Examples:

```text
"Hello"
-> memory lookup may be skipped

"What did I tell you about my business?"
-> VoiceMem lookup

"I prefer evenings. What time can you book me?"
-> VoiceMem may be useful + LangGraph continues

"What services does Agentix offer?"
-> RAG, VoiceMem usually unnecessary

"What Agentix service would fit my dental clinic?"
-> RAG + VoiceMem
```

The implementation should prefer a lightweight deterministic/router approach over adding another expensive LLM call.

## Temporary Context Rule

Memory must be inserted only into the copied/inference context, similar to the current RAG strategy.

Do not permanently append retrieved memory as assistant/user history.

Recommended conceptual block:

```text
CALLER MEMORY CONTEXT
Caller-specific historical memory. Use only when relevant.
Do not treat it as company knowledge.
Do not expose internal memory metadata.
...
```

## Failure Handling

Retrieval failure:

```text
VoiceMem retrieve
    |
    +-- success -> use relevant memory
    |
    +-- timeout/error -> log warning
                        use empty memory context
                        continue response
```

Set a bounded timeout appropriate for real-time voice interaction.

Memory retrieval must never block the caller indefinitely.

## Verification

Use the standalone-tested persisted memory or create a test caller memory.

Test:

```text
TEST_CALLER_ID=terminal_ali_001

Run 1 / existing persisted memory:
- Ali runs a dental clinic
- prefers evening meetings

Run main.py:
User: "What business did I tell you I run?"

Expected:
Agent correctly uses the caller-specific memory.
```

Then:

```text
TEST_CALLER_ID=terminal_other_002

User: "What business did I tell you I run?"

Expected:
Agent must NOT receive Ali's dental-clinic memory.
```

## Acceptance Criteria

- Same caller retrieves relevant old memory.
- Different caller cannot retrieve another caller's memory.
- Memory context is temporary.
- VoiceMem retrieval failure does not break the conversation.
- Existing RAG and booking still work.
- No live conversation writes yet.

## Codex Prompt 2

```text
Add the VoiceMem READ path to the existing pipecat-voice-agent using the adapter created in Phase 1.

STRICT:
- Do not add live VoiceMem writes yet.
- Do not modify VoiceMem's tested Left/Right Brain internals.
- Do not change RAG retrieval behavior.
- Do not change LangGraph booking behavior.
- Do not change Deepgram/Pipecat/audio behavior.
- VoiceMem must remain fail-open.
- Caller memory must be isolated by caller_id.
- Never match callers by name.
- Retrieved memory must be temporary inference context only.
- Do not permanently append retrieved memory to conversation history.
- Do not expose raw internal memory metadata to the caller.

Tasks:
1. Identify the correct point where a finalized user transcript is available before Groq inference.
2. Add lightweight routing so caller-memory retrieval runs only when potentially useful.
3. Retrieve relevant Left Brain and Right Brain memory for the active caller_id.
4. Inject selected memory into the copied/temporary Groq inference context.
5. Keep company RAG context and caller VoiceMem context clearly separated.
6. Add a bounded retrieval timeout.
7. On timeout/error:
   - log a concise warning
   - return no memory context
   - continue normal voice-agent execution
8. Prevent cross-caller leakage.
9. Add tests for:
   - same caller retrieval
   - different caller isolation
   - retrieval failure
   - timeout
   - temporary context only
   - existing RAG still works
   - booking flow still works
10. Provide a terminal validation procedure using TEST_CALLER_ID.

Do not add VoiceMem ingestion/write behavior yet.

At the end report:
- files changed
- retrieval hook location
- routing behavior
- temporary context format
- timeout/failure behavior
- caller-isolation test result
- regression test results
- exact manual validation steps
```

---

# Phase 3 - VoiceMem Write Path and Persistent Caller Memory

## Objective

Persist useful caller information from real voice-agent conversations into VoiceMem.

## Write Flow

```text
Caller turn / conversation
        |
        v
Successful conversational processing
        |
        v
VoiceMem ingest
        |
        +------------------+
        |                  |
        v                  v
Left Brain            Right Brain
facts                 preferences /
business info         emotional context
        |                  |
        +---------+--------+
                  |
                  v
            persistent store
```

## Write Strategy

Do not blindly save every raw STT fragment.

Only finalized, useful conversational content should reach VoiceMem.

Avoid:
- partial STT hypotheses
- duplicate frames
- repeated acknowledgements
- tool internals
- system prompts
- RAG chunks
- API responses
- Calendar IDs
- Sheets IDs
- secrets/tokens

Candidate memory content:
- caller name if explicitly provided
- business/company
- role
- recurring business problem
- preferences
- relevant prior decisions
- meeting preferences
- stable relationship/persona context
- emotional context where VoiceMem's Right Brain naturally extracts it

VoiceMem itself should perform its normal extraction rather than adding a second custom fact-extraction system.

## Ingestion Timing

Memory write should not delay spoken response.

Preferred pattern:

```text
response path -> caller hears answer normally

memory write -> post-response / non-blocking safe task
```

If the current Pipecat lifecycle makes asynchronous post-turn ingestion unsafe, use the safest lifecycle-supported mechanism. Do not introduce uncontrolled background tasks.

## Deduplication

Avoid repeated ingestion of the same finalized turn/context.

Use a deterministic turn/message identifier or existing frame identity where possible.

## Failure Handling

Ingestion failure:

```text
VoiceMem ingest fails
       |
       v
log warning
       |
       v
DO NOT fail current call
DO NOT repeat caller response
DO NOT corrupt booking state
```

Memory persistence is secondary to live conversation continuity.

## Persistence Test

### Caller A

```text
TEST_CALLER_ID=terminal_ali_001

Run main.py
User:
"My name is Ali. I run a dental clinic and I prefer evening meetings."

Exit main.py.
```

Restart:

```text
TEST_CALLER_ID=terminal_ali_001
python main.py
```

Ask:

```text
"What business do I run and when do I prefer meetings?"
```

Expected:
- dental clinic
- evening meetings

### Caller B Isolation

```text
TEST_CALLER_ID=terminal_bilal_002
```

Ask same question.

Expected:
- no Ali-specific memory.

## Acceptance Criteria

- Useful finalized caller information persists.
- Restart does not lose memory.
- Different caller IDs remain isolated.
- Duplicate turns are not repeatedly ingested.
- Ingestion failure does not affect spoken response.
- RAG and booking remain unchanged.

## Codex Prompt 3

```text
Add the VoiceMem WRITE path to the existing voice agent.

STRICT:
- Keep the working READ path unchanged unless a small fix is required.
- Do not store every raw STT frame.
- Do not store partial transcripts.
- Do not store system prompts, RAG chunks, API/tool internals, secrets, Calendar event IDs, spreadsheet IDs, or hidden metadata.
- Do not create a second custom fact-extraction system. Let VoiceMem perform its normal Left/Right Brain extraction.
- Memory ingestion must never block or break the live response.
- Do not change LangGraph state semantics.
- Do not change RAG semantics.
- Caller_id isolation is mandatory.

Tasks:
1. Identify the safest finalized-turn/conversation point for VoiceMem ingestion.
2. Send only appropriate user conversational content into VoiceMem.
3. Preserve Left Brain + Right Brain behavior.
4. Add duplicate-ingestion protection.
5. Prefer non-blocking/post-response ingestion using a lifecycle-safe mechanism.
6. If ingestion fails:
   - log warning
   - do not retry aggressively in the live path
   - do not affect the spoken response
   - do not affect LangGraph/Calendar/Sheets
7. Add tests for:
   - persistence across restart/process
   - same caller recall
   - different caller isolation
   - duplicate protection
   - ingestion failure
   - no internal/tool/RAG context accidentally stored
8. Verify the existing RAG and booking tests still pass.
9. Provide exact terminal manual test commands using two TEST_CALLER_ID values.

At the end report:
- files changed
- ingestion hook
- what content is eligible for memory
- duplicate protection
- failure behavior
- persistence result
- caller isolation result
- regression test results
```

---

# Phase 4 - Parallel RAG + VoiceMem Retrieval, Hardening and Future Phone Identity

## Objective

Finalize the architecture so Agentix RAG and caller VoiceMem can retrieve independently and concurrently when both are required.

## Final Retrieval Architecture

```text
                     Final transcript
                           |
                           v
                     Routing decision
                           |
              +------------+------------+
              |                         |
              v                         v
          Agentix RAG                VoiceMem
        company knowledge           caller history
              |                         |
              +------------+------------+
                           |
                           v
                 Context composition
                           |
                           v
                         Groq
```

Use concurrent retrieval when both are required.

Conceptually:

```python
rag_result, memory_result = await asyncio.gather(
    rag_task,
    memory_task,
    return_exceptions=True,
)
```

Use the actual project async model/API after inspection. Do not force this exact code if it conflicts with Pipecat architecture.

## Context Priority

Keep sources explicit:

```text
SYSTEM / RECEPTIONIST RULES
        |
        +-- authoritative behavior

AGENTIX KNOWLEDGE
        |
        +-- company facts from RAG

CALLER MEMORY
        |
        +-- caller-specific history/preferences

CURRENT CONVERSATION
        |
        +-- current turn/session

LANGGRAPH STATE
        |
        +-- current workflow/action state
```

Rules:
- VoiceMem must never override verified Agentix company facts.
- RAG must never be treated as caller history.
- Caller memory can personalize a response but cannot invent company capabilities.
- Current explicit caller statement should supersede stale caller memory when they conflict.
- Booking/tool results are authoritative for actual availability and booking status.

## Parallel Failure Matrix

```text
RAG OK + VoiceMem OK
-> use both

RAG OK + VoiceMem FAIL
-> use RAG, continue

RAG FAIL + VoiceMem OK
-> use caller memory, but do not invent company facts

RAG FAIL + VoiceMem FAIL
-> continue normal conversation/booking with no retrieved context
```

No retrieval subsystem may crash the real-time pipeline.

## Observability

Add concise structured logging/metrics for:

- RAG route: yes/no
- VoiceMem route: yes/no
- RAG latency
- VoiceMem latency
- number of RAG chunks injected
- number of caller memories injected
- retrieval failure/timeout
- memory ingestion success/failure
- caller ID only in safe/redacted form

Do not log raw private caller memory by default.

## Phone Identity Abstraction

Do not implement full Asterisk telephony in this phase unless already available.

Prepare only the identity boundary:

```text
NOW:
TEST_CALLER_ID
     |
     v
caller_identity.resolve()

LATER:
incoming phone number
     |
normalize E.164
     |
HMAC with server-side secret
     |
stable caller_id
     |
caller_identity.resolve()
```

Recommended future behavior:

```text
+923001234567
    |
HMAC-SHA256(secret, normalized_number)
    |
caller_<stable digest>
```

Do not store raw phone number as the vector-memory namespace if avoidable.

## Final End-to-End Tests

### Test 1 - RAG only

```text
"What services does Agentix Labs AI offer?"
```

Expected:
- RAG used
- VoiceMem optional/skipped
- grounded company answer

### Test 2 - VoiceMem only

```text
"What business did I tell you I run?"
```

Expected:
- VoiceMem used
- correct caller memory

### Test 3 - Both

Caller memory:
```text
I run a dental clinic.
```

Question:
```text
"Which Agentix service would fit my dental clinic?"
```

Expected:
- VoiceMem supplies dental-clinic context
- RAG supplies verified Agentix services
- Groq combines them without inventing services

### Test 4 - Isolation

Change `TEST_CALLER_ID`.

Expected:
- previous caller memory absent

### Test 5 - VoiceMem unavailable

Temporarily force memory adapter failure.

Expected:
- voice agent still responds
- RAG/booking remain available

### Test 6 - Qdrant unavailable

Expected:
- VoiceMem still usable
- company claims remain conservative/unverified
- agent stays alive

### Test 7 - Both retrieval systems unavailable

Expected:
- basic conversation + booking still operate where no company retrieval is required
- no crash

## Acceptance Criteria

- RAG and VoiceMem can execute independently.
- When both are needed they run concurrently where technically safe.
- Context sources remain separated.
- Failure matrix works.
- Latency is measured.
- Existing voice behavior remains stable.
- Manual caller identity can later be replaced by phone-number identity without changing VoiceMem or RAG internals.

## Codex Prompt 4

```text
Finalize the VoiceMem integration by adding parallel/independent orchestration with the existing Agentix RAG and hardening the full voice-agent pipeline.

STRICT:
- Do not redesign RAG.
- Do not redesign VoiceMem.
- Do not redesign LangGraph.
- Do not implement Asterisk telephony yet.
- Do not make phone number assumptions in current terminal mode.
- Do not allow one retrieval subsystem failure to crash the call.
- Do not log raw caller memories or secrets.
- Maintain strict separation:
  Agentix RAG = company knowledge
  VoiceMem = caller-specific memory
  LangGraph = workflow state
  Groq = reasoning/response generation

Tasks:
1. Review the existing routing for RAG and VoiceMem.
2. Allow four routing outcomes:
   - neither
   - RAG only
   - VoiceMem only
   - both
3. When both are needed, execute retrieval concurrently if compatible with the current Pipecat async architecture.
4. Compose temporary inference context with clearly separated sections for:
   - Agentix knowledge
   - caller memory
5. Enforce precedence:
   - system/receptionist rules remain highest
   - tool/Calendar booking results remain authoritative for actions
   - RAG is authoritative for Agentix company facts
   - current caller statement supersedes stale caller memory
   - caller memory may personalize but must not expand Agentix offerings
6. Implement full fail-open behavior:
   - RAG fail -> continue
   - VoiceMem fail -> continue
   - either timeout -> continue
   - both fail -> continue without retrieval context
7. Add concise observability:
   - route decisions
   - retrieval latency
   - number of injected RAG chunks
   - number of injected memories
   - failure/timeout counters/logs
   Do not log raw private memory by default.
8. Keep TEST_CALLER_ID for terminal testing.
9. Prepare caller_identity abstraction for later phone-number integration, but do not implement Asterisk.
10. Add tests covering:
    - RAG only
    - VoiceMem only
    - both
    - neither
    - caller isolation
    - VoiceMem failure
    - Qdrant/RAG failure
    - both retrievals failing
    - booking regression
    - temporary-context behavior
11. Run the relevant full test suite and provide a concise final architecture report.

At the end report:
- files changed
- final routing logic
- concurrency behavior
- context composition
- precedence rules
- failure matrix
- observability added
- terminal caller identity behavior
- test results
- any remaining production blockers before Asterisk integration
```

---

# Final Expected Project Structure

Exact filenames may differ after implementation, but the separation should remain similar:

```text
pipecat-voice-agent/
│
├── main.py
│
├── agent/
│   └── booking / LangGraph...
│
├── rag/
│   ├── integration.py
│   ├── retrieve.py
│   └── ...
│
├── memory/
│   ├── __init__.py
│   ├── voicemem_adapter.py
│   └── caller_identity.py
│
├── services/
│   ├── google_calendar.py
│   ├── google_sheets.py
│   └── ...
│
└── tests/
    ├── test_rag_integration.py
    ├── test_memory_integration.py
    └── ...
```

VoiceMem remains a separately maintained local package/repository:

```text
D:\AI Internship\Pipecat\
├── pipecat-voice-agent\
└── voicemem-test\
    └── VoiceMem\
```

---

# Production Identity Upgrade Later

Current terminal mode:

```text
TEST_CALLER_ID=terminal_test_001
```

Future Asterisk mode:

```text
Incoming phone number
        |
        v
normalize number
        |
        v
HMAC / stable irreversible caller key
        |
        v
caller_id
        |
        v
same VoiceMem adapter
```

No RAG, VoiceMem, Groq, or LangGraph redesign should be required when telephony is added.

---

# Definition of Done

VoiceMem integration is complete when all of the following are true:

- Voice agent starts even if VoiceMem is unavailable.
- Existing Pipecat/Deepgram flow is unchanged.
- Existing Agentix RAG still works.
- Existing LangGraph booking still works.
- Existing Calendar/Sheets integrations still work.
- Same caller ID retrieves its previous persistent memory.
- Different caller IDs cannot retrieve each other's memories.
- Left Brain facts work.
- Right Brain context works.
- Memory survives process restart.
- Memory writes do not delay or break live responses.
- Duplicate turns are not repeatedly ingested.
- RAG and VoiceMem can run independently.
- RAG and VoiceMem can run concurrently when both are needed.
- Retrieval context is temporary.
- Company facts and caller memories remain clearly separated.
- All retrieval failures are fail-open.
- Current terminal identity can later be replaced by phone-number-derived caller identity through one abstraction layer.
