# Continuous Autonomous AI Agent Framework
Stateless, continuous, autonomous agent framework designed to execute on free stateless cloud platforms (such as GitHub Actions cron schedules or serverless workers).

## Architectural Components

1. **`state_manager.py`**:
   - Implements a deterministic Finite State Machine (FSM):
     `WAKE_UP` -> `PLAN` -> `EXECUTE` -> `EVALUATE` -> `SLEEP` (with `ERROR` and `PENDING_APPROVAL` handling).
   - Validates legal directional transitions.
   - Preserves state change audit logs and execution cycle context (`AgentContext`).
   - Owns the **Non-Blocking Multi-Task Proposal Ledger** (`AgentContext.proposals_ledger`): a persistent global
     array of human-gated tasks, each with a unique `task_id` and a `status` (`PENDING_APPROVAL` / `COMPLETED` /
     `DENIED`). Helper surface: `append_proposal()`, `get_proposal()`, `remove_proposal()`, `purge_proposals()`,
     `pending_proposals()`, `ledger_summary()`. The ledger is global state and deliberately survives
     `reset_for_new_cycle()`.

2. **`memory_bridge.py`**:
   - Defines `BaseMemoryBridge`, an abstract interface decoupling the agent logic from external database backends (Supabase, DynamoDB, Redis, S3, or GitHub artifacts).
   - Provides `LocalFileMemoryBridge` reference implementation storing short-term state in atomic JSON snapshots and long-term history in append-only JSONL files.

3. **`agent_core.py`**:
   - The primary autonomous orchestrator (`AutonomousAgent`).
   - Uses standard library `urllib` to invoke OpenAI-compatible LLM endpoints without paid or heavy SDK dependencies.
   - Enforces structured internal monologues (`THINKING PROCESS`, `STEP-BY-STEP PLAN`, `EXECUTION ASSESSMENT`).
   - Runs a **non-blocking** Human-in-the-Loop gate: `_register_gate()`, `_enter_pending_approval()`,
     `_evaluate_human_gate()`, `_process_proposal_ledger()` and `_queue_ledger_task()` park gated work in the ledger
     and immediately resume the plan, then reconcile human decisions on the next wake-up.
   - Implements fault-tolerant error containment: exceptions transition to `ERROR` and persist context before gracefully winding down to `SLEEP`.

   - Embeds standard Python `Toolbox` with 3 zero-dependency capabilities:
     - `web_browser_tool(url)`: Fetches and cleans textual content from public RSS feeds, job boards, or websites via `urllib.request`.
     - `file_writer_tool(filepath, content)`: Writes scripts, portfolios, reports, or application files into the local workspace with directory traversal containment.
     - `http_poster_tool(url, payload)`: Sends HTTP POST requests to simulate external platform interactions, registrations, or work submissions.
   - Implements a production-grade 5-provider `LLMClient` Fallback Priority Ledger (Groq -> Gemini -> Mistral -> OpenRouter -> xKiro). On HTTP 429/400-class errors or timeouts it prints "[WARNING] Active provider failed with error. Diverting prompt state to the next priority engine backup..." and replays the exact prompt against the next engine.
   - Instructs the agent with an open-ended discovery objective: analyze long-term memory, choose lucrative or educational digital tasks, execute tools dynamically, and log successes/failures for continuous self-improvement.

### Advanced Autonomy Features

- **Non-Blocking Multi-Task Proposal Ledger** — the single blocking `pending_proposal` dictionary is gone. When the
  agent meets a task that needs human intervention, an account wall, or external publishing, it packages the action
  and appends it to the global `proposals_ledger` with a unique ID and a status of `PENDING_APPROVAL`, then
  **immediately continues its next planned pipeline step in the same run**. The FSM only settles into `SLEEP` once
  the entire multi-step plan for that cycle is concluded, so many verifications can wait on the dashboard in
  parallel while the agent keeps exploring new opportunities. Gate registration is de-duplicated, so a step detected
  in both `PLAN` and `EXECUTE` produces exactly one ledger entry and one alert.
- **Account Wall Routing Gate** — when the agent finds a high-value platform (Upwork, Freelancer, GitHub Sponsors, etc.) that demands registration, sign-up, login, or verification, it is forbidden from skipping it. It raises an **Account Creation & Verification** proposal (`requires_account`, `registration_link`, `session_active`) so the human can approve it on the dashboard instead of the opportunity being silently dropped.
- **Multi-Step Batch Processing (tool chaining)** — a single run executes an internal tool chain of up to **5**
  sequential execution steps (`MAX_CHAIN_STEPS_PER_CYCLE`). Each step runs, its output is appended to the FSM
  `working_memory`, and the next link processes with that context. Bounded by `MAX_TOOL_CALLS_PER_CYCLE` (default
  **5**) so nothing can loop forever; the EVALUATE phase receives the chain summary before the agent enters `SLEEP`.
- **Invisible Cookie-Session Handshake** — `web_browser_tool` automatically injects the optional top-level
  `session_cookies` dictionary from the Gist state into outgoing request headers as standard `Cookie:` literals
  (`name=value; name2=value2`). The human signs in on their own machine, pastes the cookie string into the dashboard,
  and the agent wakes up already authenticated — it never receives a password. Values are sanitized and redacted in
  logs/proposals.


4. **`cloud_memory_bridge.py`**:
   - Implements `GitHubGistMemoryBridge` subclassing `BaseMemoryBridge` using Python's standard `urllib.request` (zero third-party dependencies).
   - Reads and writes short-term JSON states and append-only `.jsonl` audit logs directly to/from a private GitHub Gist via Bearer Authentication (`CUSTOM_GIT_TOKEN`).
   - Includes graceful fault tolerance: transparently mirrors updates locally and safely falls back to `LocalFileMemoryBridge` on network or credential failures.

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `AGENT_ID` | `default_stateless_agent` | Identifier for the agent instance. |
| `MAX_STEPS_PER_RUN` | `5` | Maximum atomic steps allowed per invocation. |
| `MAX_TOOL_CALLS_PER_CYCLE` | `5` | Safety bound on chained `TOOL_CALL` links executed inside one cloud run (capped at the 5-step batch window). |
| `MAX_CHAIN_STEPS_PER_CYCLE` | `5` | Hard ceiling on sequential execution steps per cycle (compile-time constant `MAX_CHAIN_STEPS_PER_CYCLE`). |
| `LLM_PROVIDER` | `openai` | LLM provider (`openai`, `groq`, `anthropic`). |
| `LLM_MODEL` | `gpt-4o-mini` | Target completion model name. |
| `LLM_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible endpoint. |
| `OPENAI_API_KEY` | `None` | API Key for model completions (falls back to offline heuristic if omitted). |
| `AGENT_LOG_LEVEL` | `INFO` | Standard logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `CUSTOM_GIT_TOKEN` | `None` | Custom GitHub Personal Access Token (with `gist` scope) for Gist API authentication. Uses a custom name to avoid collision with Actions' internal `GITHUB_TOKEN` scope. |
| `GIST_ID` | `None` | Private Gist ID used to persist state and logs across stateless cloud runs. |
| `GROQ_API_KEY` | `None` | Priority 1 Groq key (model `qwen/qwen3.8-27b` @ `https://api.groq.com/openai/v1`). |
| `GEMINI_API_KEY` | `None` | Priority 2 Gemini key (model `gemini-1.5-flash` @ `https://generativelanguage.googleapis.com/v1beta/openai`). |
| `MISTRAL_API_KEY` | `None` | Priority 3 Mistral key (model `mistral-small-latest` @ `https://api.mistral.ai/v1`). |
| `OPENROUTER_API_KEY` | `None` | Priority 4 OpenRouter key (model `openrouter/free` @ `https://openrouter.ai/api/v1`). |
| `XKIRO_API_KEY` | `None` | Priority 5 xKiro backup key (open-weights model `qwen/qwen3-32b` @ `https://api.xkiro.com/v1`). |
| `DISCORD_WEBHOOK_URL` | `None` | Discord webhook that receives an embedded approval card when a proposal enters `PENDING_APPROVAL`. |
| `SMTP_SERVER` / `SMTP_PORT` | `None` / `587` | SMTP endpoint for email alerts (STARTTLS; port 465 uses SSL). |
| `SMTP_USER` / `SMTP_PASSWORD` | `None` | SMTP credentials used to send the approval summary email. |
| `NOTIFY_EMAIL` | falls back to `SMTP_USER` | Recipient address for approval emails. |
| `DASHBOARD_URL` | `https://autonomous-agent-gateway.vercel.app` | Vercel Dashboard link embedded in Discord/email alerts. |


## Human-in-the-Loop Dashboard Gateway

When the agent detects a task that touches external platforms, money movement, or asset publishing, it routes it into the ledger without stopping:

1. The gated task is **appended to the global `proposals_ledger`** (top-level in the Gist state) with a unique
   `task_id` and `status: "PENDING_APPROVAL"`, alongside `human_decision: null` and the review payload: `type`
   (Freelance Pitch / Code Asset Sale / Trend Report / Account Creation & Verification), `target_platform`,
   `proposed_action`, `generated_files`, and `estimated_revenue`. Account-wall proposals additionally carry
   `requires_account` and `registration_link`.
2. The run **keeps going**. The agent immediately executes its remaining planned pipeline steps and only settles
   into `SLEEP` once the whole multi-step plan for the cycle has concluded — so several tasks can wait for you at
   once while the agent continues working. `task_id` is derived deterministically and de-duplicated, so a step
   detected in both `PLAN` and `EXECUTE` yields exactly one ledger entry.
3. Immediately after each task is parked, the zero-dependency `NotificationManager` fires instant alerts:
   - **Discord** (`DISCORD_WEBHOOK_URL`): an embedded approval card with Task ID, Platform, Estimated Revenue, and a direct Vercel Dashboard link (`DASHBOARD_URL`, default `https://autonomous-agent-gateway.vercel.app`).
   - **Email** (`SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, optional `NOTIFY_EMAIL`): a summary alert that a task awaits review (uses `smtplib`, STARTTLS on 587 / SSL on 465).
4. On the next wake-up `_evaluate_human_gate` / `_process_proposal_ledger` walks the ledger in one pass, matching
   decisions case-insensitively and whitespace-tolerantly (`.strip().upper() == "APPROVED"`):
   - `"APPROVED"` $\rightarrow$ prints `[SUCCESS] Human approval validated for task <id>. Clearing gate and launching task chain...`, immediately queues **that specific task** into the active execution window, runs its stored `proposed_action` payload, and flips its ledger status to `COMPLETED`.
   - `"DENIED"` $\rightarrow$ the entry is **purged** from the active ledger array.
   - `null` (no decision yet) $\rightarrow$ the entry is **left completely untouched**, waiting safely on the dashboard while the rest of the agent runs independently.
5. Meanwhile the un-gated remainder of the run proceeds normally; an approved task's execution results are appended
   to `execution_results` and summarised into `working_memory` like any other chain link.

Legacy compatibility: a state file still carrying the old single `context_data.pending_proposal` (plus
`context_data.human_decision`) is automatically migrated into the ledger on wake-up.

## Human-in-the-Loop Dashboard (Vercel)

**Live dashboard: https://autonomous-agent-gateway.vercel.app**

The gateway UI lives in a single self-contained file: **`index.html`** (Tailwind CSS via CDN + vanilla JS, zero build step).

### Deployment layout
- `index.html` — the dashboard (served as the static root).
- `vercel.json` — static project config (`framework: null`, no build command, `outputDirectory: "."`).
- `.vercelignore` — keeps backend Python, `.agent_memory/`, and docs out of the public deployment.
- Vercel project: `steven-songwe-s-projects/autonomous-agent-gateway` (linked + connected to this GitHub repo, so every push to `main` can be redeployed).


### Features
- **Command center UI** — dark responsive layout with an active agent connection card, a live lifecycle tracker (`WAKE_UP → PLAN → EXECUTE → PENDING_APPROVAL → EVALUATE → SLEEP`), and a metrics panel (Cycle Count, Est. Revenue, Task Type, Target Platform).
- **Lockable credentials panel** — `GIST_ID`, `GIT_TOKEN`, and optional `AGENT_ID` are stored exclusively in the browser's `localStorage` (never sent anywhere except `api.github.com`).
- **Proposal ledger render engine** — uses the GitHub Gists REST API (`GET /gists/{id}`, `PATCH /gists/{id}`) with `async/await`. Every `PENDING_APPROVAL` entry in `proposals_ledger` is rendered as its own card with Task ID, platform, proposed action, generated-file code previews, expected revenue, and — for account walls — the registration link. Entries already `COMPLETED` or `DENIED` are excluded.
- **Per-task 1-button controls** — each card owns its own `🟢 APPROVE TASK` / `🔴 DENY TASK` pair, writing `human_decision` into **that specific ledger entry** (`APPROVED` / `DENIED`, and `status: DENIED` on a denial) while leaving every other task untouched.
- **Injected Cookie String field** — a per-card textarea. Sign into the platform on your own PC, copy the `Cookie:` header from DevTools → Network, paste it in, and press **APPROVE**: the dashboard parses `name=value` pairs (attribute tokens like `Secure`/`HttpOnly`/`SameSite` are skipped) and writes them to the top-level `session_cookies` of the Gist state, so the agent wakes up already logged in and never sees a password. A 🍪 badge shows how many cookies are active (names only).

### Deploy to Vercel (free)
```bash
# Option A — Vercel CLI
npm i -g vercel
vercel            # framework preset: "Other" · output directory: ./
vercel --prod

# Option B — GitHub import
# Vercel Dashboard → Add New… → Project → Import the repo → Deploy
# (static site: no build command, no output directory needed)
```

Open the deployed URL (**https://autonomous-agent-gateway.vercel.app**), click **Settings**, paste your `GIST_ID` and a GitHub token with `gist` scope, then press **Save Credentials**. The dashboard polls the Gist every 60 seconds.

### Dashboard regression test
```bash
node dashboard_smoke_test.js   # 34/34 checks: ledger render + per-task APPROVE/DENY + cookie handshake (stubbed Gist)
```

## Running the Agent

```bash
# Run a single cycle (offline heuristic or live if API key set)
python agent_core.py

# Run the automated test suite
python test_framework.py
```

## GitHub Actions Automated Deployment & Scheduling

The agent is orchestrated to run continuously and autonomously using the scheduled workflow in `.github/workflows/agent_cron.yml`.

### Workflow Highlights
- **Schedule**: Executes automatically every 30 minutes (`cron: '*/30 * * * *'`).
- **Manual Trigger**: Supports `workflow_dispatch` to initiate a run on demand from the GitHub Actions tab.
- **Concurrency Control**: Protected by `concurrency: group: agent-loop, cancel-in-progress: false`, ensuring overlapping runs never collide or corrupt remote memory state.
- **Resource Protection**: Enforces `timeout-minutes: 10` on `ubuntu-latest` with caching disabled to prevent hanging processes and storage bloat.

### Configuring GitHub Repository Secrets

To enable cloud autonomy, configure your secrets in GitHub:

1. Navigate to your repository on GitHub.
2. Click **Settings** (top tab) $\rightarrow$ **Secrets and variables** (left sidebar) $\rightarrow$ **Actions**.
3. Under the **Repository secrets** section, click **New repository secret** and add the following:

| Secret Name | Value Instructions |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) (Priority 1 failover provider). |
| `GEMINI_API_KEY` | Your Google AI Studio key from [aistudio.google.com](https://aistudio.google.com) (Priority 2 backup). |
| `MISTRAL_API_KEY` | Your Mistral key from [console.mistral.ai](https://console.mistral.ai) (Priority 3 backup). |
| `OPENROUTER_API_KEY` | Your OpenRouter key from [openrouter.ai](https://openrouter.ai) (Priority 4 backup). |
| `XKIRO_API_KEY` | Your xKiro key from [xkiro.com](https://xkiro.com) (Priority 5 backup, OpenAI-compatible `https://api.xkiro.com/v1`). |
| `GIST_ID` | The 32-character ID of your private GitHub Gist created to store agent memory (e.g. from `https://gist.github.com/username/<GIST_ID>`). |
| `GIT_TOKEN` (mapped to `CUSTOM_GIT_TOKEN`) | A GitHub Personal Access Token (classic with `gist` scope or fine-grained token with Gist read/write permission) so the runner can update the Gist. Stored as `GIT_TOKEN` secret, injected as `CUSTOM_GIT_TOKEN` env var to avoid collision with Actions' reserved `GITHUB_TOKEN`. |
| `DISCORD_WEBHOOK_URL` (optional) | Discord channel webhook URL for instant approval cards (Channel Settings $\rightarrow$ Integrations $\rightarrow$ Webhooks). |
| `SMTP_SERVER` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `NOTIFY_EMAIL` (optional) | SMTP details for approval emails (e.g. `smtp.gmail.com`, `587`). |
| `DASHBOARD_URL` (optional) | Public URL of your Vercel Human-in-the-Loop Dashboard. |

#### Step-by-Step Gist Setup:
1. Go to [gist.github.com](https://gist.github.com).
2. Enter a filename (e.g., `default_stateless_agent_state.json`) and placeholder content `{}`.
3. Click **Create secret gist**.
4. Copy the alphanumeric ID from the end of the URL and save it as the `GIST_ID` secret in your repository.

## 🚀 Precision Timing with External Cron Providers

GitHub Actions' internal cron schedule engine runs on shared public worker pools, which can experience standard free-tier load scheduling delays (typically 5 to 30 minutes). If your agent requires precision timing or immediate execution, you can trigger the runner on-demand with to-the-minute accuracy using an external pinger service (such as [cron-job.org](https://cron-job.org), EasyCron, or custom webhooks).

External cron providers trigger GitHub's official REST API endpoint for `workflow_dispatch`.

### Step-by-Step Configuration (e.g. cron-job.org)

1. **Target Endpoint URL**:
   ```text
   https://api.github.com/repos/leonardted005-dev/autonomous-agent/actions/workflows/agent_cron.yml/dispatches
   ```
   *(General format: `https://api.github.com/repos/<OWNER>/<REPO>/actions/workflows/<WORKFLOW_FILE>/dispatches`)*

2. **HTTP Method**:
   `POST`

3. **Required Request Headers**:
   - `Authorization: Bearer <YOUR_GITHUB_TOKEN>` *(Personal Access Token with `repo` or `actions:write` scope)*
   - `Accept: application/vnd.github+json`
   - `X-GitHub-Api-Version: 2022-11-28`
   - `Content-Type: application/json`

4. **Required JSON Body Payload**:
   ```json
   { "ref": "main" }
   ```

5. **Schedule**:
   Set your exact desired interval (e.g., every 30 minutes, 15 minutes, or hourly) in the external pinger's dashboard.

Whenever the external cron fires, GitHub Actions will immediately initiate the runner, bypassing the native scheduling queue while respecting the concurrency lock (`group: agent-loop`) and timeout boundaries.

