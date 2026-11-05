# ResolveBot (LangGraph version)

A customer support chat assistant for an online electronics store ("ShopSphere"), served over HTTP
(`POST /chat`, port 8000). Python 3.10+, FastAPI, SQLite, LangGraph. No UI.

The assistant can track orders, check whether an item can be returned, answer warranty questions,
open return requests and support tickets (warranty or cancellation) **only after the customer says yes**,
and answer questions from the store's policy documents, citing the section it used.

> Status: work in progress. The service starts and answers live chats with Gemini (greeting, routing and the
> order-not-found reply have been tried). The remaining scenarios are being tested one by one; fixes follow in
> later commits. See "What does not work yet".

## Run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key
# put the data/ folder (policies.md, manifest.json, shopsphere.db) next to this README
uvicorn main:app --port 8000
```

Example call:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" -H "X-Customer-Id: C1009" \
  -d '{"session_id": "s1", "message": "Where is my order SS-10512?"}'
```

Status codes: 401 missing/unknown `X-Customer-Id`, 400 bad body, 403 session belongs to another customer.

### Settings (`.env`)

| Variable | Meaning |
|---|---|
| `POLICIES_PATH`, `MANIFEST_PATH`, `DB_PATH` | Data files (defaults point at `./data`) |
| `APP_TODAY` | The date treated as "today" (default `2026-07-01`); the computer clock is never used |
| `LLM_PROVIDER` | `anthropic`, `openai` or `gemini` |
| `LLM_MODEL` | Optional. Empty = the provider's default (see `config.py`) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Only the key for the chosen provider is read |
| `MAX_TOOL_CALLS`, `LLM_TIMEOUT_SECONDS`, `REPLY_TIME_BUDGET_SECONDS` | Limits |

Model names are retired from time to time. If a provider answers 404 "model not found", set `LLM_MODEL`
to a model your key can use. Temperature is 0 (a model that rejects the setting is retried without it).

### Testing by hand in Swagger

Open `http://localhost:8000/docs`, click **Authorize** (top right), enter a customer ID such as `C1009` once, and
then every request is sent with that `X-Customer-Id` header, so the request form shows only the two body fields
(`session_id`, `message`). Use the same `session_id` to continue a chat and a new one to start a new chat.
The Authorize box only fills the header; the service still checks it itself (missing or unknown -> 401).

## Design

| File | Job |
|---|---|
| `main.py` | HTTP layer, validation, status codes |
| `agent.py` | The LangGraph graph: state, reducers, memory, the "yes" gate, the tool loop |
| `tools.py` | The 5 tools + their schemas; all customer scoping lives here |
| `rules.py` | Return / warranty / date rules. Pure Python, no DB, no LLM |
| `policy_search.py` | Policy RAG: split by `§` paragraphs, manifest filtering, BM25 ranking |
| `llm.py` | LangChain chat models (Anthropic / OpenAI / Gemini / local OpenAI-compatible) |
| `prompts.py` | System prompt (behaviour only; no business numbers) |
| `config.py`, `db.py` | Settings from env; SQLite helpers |

### The graph (`agent.py`)

```
START -> entry --(waiting proposal + clear "yes")--> confirm ------------------------> END
            \--> start_turn -> agent --tool calls--> tools --> agent (loop)
                                  |                    \--(order not found)--> finalize
                                  |--> finalize / too_many / fail -------------------> END
```

* **State** (`ChatState`): the messages, the customer ID (from the header), the waiting proposal (`pending`) and a per-turn log.
* **Reducers**: `add_messages` appends messages and lets a node delete old ones (`RemoveMessage`); our own `accumulate` reducer adds to the per-turn tool log and is cleared by `None` at the start of each request.
* **Memory**: `MemorySaver` (LangGraph's in-memory checkpointer), keyed by `thread_id = session_id`. It keeps the chat messages and the waiting proposal between requests. At the end of a turn the history is trimmed to the user message plus one plain assistant reply, so tool calls from old turns are not replayed.
* **Conditional edges**: the "yes" gate at the start, the tool loop (`agent -> tools -> agent`) with a cap on tool calls, and a shortcut to the final reply as soon as an order is not found.

### Key decisions

* **Python decides, the LLM explains.** Eligibility, warranty and cancellation rules are code in `rules.py`/`tools.py`. Every number (return window, warranty months) is read from the `policy_parameters` table, never typed in code.
* **Customer safety.** The customer ID comes only from the `X-Customer-Id` header and is injected by the server; it is not a tool argument. Every order query is filtered by it, and "does not exist" and "belongs to someone else" return the identical error and reply text. After such an error the reply is fixed and the LLM is not called again.
* **Writes need a "yes".** `create_return_request` / `create_support_ticket` only *validate and store a proposal* in the graph state. Saving happens in the `confirm` node on the customer's next message, and only if that message is a clear "yes" (`is_clear_yes`). Anything else ("no", "just the cable", a new question) drops the proposal. The rules are re-checked right before saving.
* **Policy search.** `policies.md` is split into documents (`# ` titles, matched to `manifest.json` by Document ID) and paragraphs (`### §x.y`). Only `status: current` documents are searchable. A document whose `effective_date` is after "today" is returned flagged `in_effect: false` with its start date. FAQ paragraphs are dropped whenever a real policy paragraph matched (policy wins over FAQ).
* **Citations are verified.** Only `doc_id`/`section` pairs that `search_policy` actually returned in that turn survive; `no_answer` and `out_of_scope` replies carry none.
* **Limits.** Max 6 tool calls per message (polite reply if exceeded), 26 s total budget, LLM errors give a friendly 200 reply instead of a 500.

## Assumptions

* A return is allowed on day 30 and refused on day 31 (`days_since_delivery <= return_window_days`).
* `defective` returns still need to be inside the return window; the window is not extended for defects.
* Warranty runs from the delivery date; the last day is still covered; month-end dates clip (31 Jan + 1 month = 28 Feb).
* A warranty ticket is allowed if at least one item in the order still has active warranty.
* Orders that are not `processing` cannot be cancelled (shipped -> `ORDER_ALREADY_SHIPPED`).
* The delivery date is read from `shipments.delivered_at`, and only for orders whose status is `delivered`.
* If any tool returns `ORDER_NOT_FOUND` during a message, the whole reply is the fixed "I couldn't find that order on your account." (route `no_answer`), even if the message also asked something else.
* Memory is in-process (`MemorySaver`): chats reset when the server restarts, and are not shared between worker processes (run a single uvicorn worker).
* Table and column names follow the provided data description; if the real database differs, `tools.py` (`_load_order`) is the only place that reads them.

## What does not work yet

* **Only partly tested against a live LLM** (Gemini). The greeting, the routing and the order-not-found reply were tried live. Returns, warranty, cancellation, the yes-step and policy questions were checked only with a scripted fake LLM and a synthetic database (including different policy numbers) and are being run live next. Wording and tool choices depend on the model and on `prompts.py`, and have not been tuned.
* **Model availability and speed vary.** Hosted models can answer 503 or be slow. One automatic retry is built in; if the provider is still failing the reply is the friendly "having trouble" message, never an error code.
* **Policy search is keyword-based (BM25).** It does not understand synonyms, so an unusual phrasing may miss the right paragraph (the reply then falls back to "not covered, email support").
* **"Yes" detection is a fixed word list.** "Yes", "confirm", "go ahead", "please do", "ok", "sure" work; other phrasings are treated as not-a-yes and the proposal is dropped (the safe side).
* **One proposal at a time, and only the very next message can confirm it.**
* **Memory is in-process only** (lost on restart, single process).
* **No automated tests are included.**
* Not implemented: exchanges, refund amounts, emails, human hand-off, UI, login.

## Example chats

Greeting (live, Gemini):

```
POST /chat   X-Customer-Id: C1009   {"session_id": "g1", "message": "Hi, what can you help me with?"}
-> route: "clarify"
   answer: "Hello! I'm ResolveBot, your customer support assistant for ShopSphere. I can help you track orders,
            assist with returns, open warranty or cancellation support tickets, and answer questions about our
            store policies. How can I help you today?"
   citations: [], tool_calls: [], pending_confirmation: false
```

More examples will be added after the remaining live tests.

## Where AI coding assistants were used

This project was built with the help of Claude (an AI coding assistant): the code for all files, including the LangGraph agent, the debugging of the Gemini model setting, and a scripted check of the rules, tools, policy search and API. I am reviewing and testing the code and can explain each part (tools, the business rules, the policy search, the graph and the yes-step).

## Secrets

`.env` is git-ignored. Only `.env.example` (no keys) is committed. `data/` is also git-ignored because it is confidential.
