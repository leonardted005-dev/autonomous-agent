# Continuous Autonomous AI Agent Framework
Stateless, continuous, autonomous agent framework designed to execute on free stateless cloud platforms (such as GitHub Actions cron schedules or serverless workers).

## Architectural Components

1. **`state_manager.py`**:
   - Implements a deterministic Finite State Machine (FSM):
     `WAKE_UP` -> `PLAN` -> `EXECUTE` -> `EVALUATE` -> `SLEEP` (with `ERROR` and `PENDING_APPROVAL` handling).
   - `PLAN` and `EXECUTE` may enter `PENDING_APPROVAL` when a task involves external platforms, financial transactions, or asset publishing (Human-in-the-Loop gate).
   - Validates legal directional transitions.
   - Preserves state change audit logs and execution cycle context (`AgentContext`).

2. **`memory_bridge.py`**:
   - Defines `BaseMemoryBridge`, an abstract interface decoupling the agent logic from external database backends (Supabase, DynamoDB, Redis, S3, or GitHub artifacts).
   - Provides `LocalFileMemoryBridge` reference implementation storing short-term state in atomic JSON snapshots and long-term history in append-only JSONL files.

3. **`agent_core.py`**:
   - The primary autonomous orchestrator (`AutonomousAgent`).
   - Uses standard library `urllib` to invoke OpenAI-compatible LLM endpoints without paid or heavy SDK dependencies.
   - Enforces structured internal monologues (`THINKING PROCESS`, `STEP-BY-STEP PLAN`, `EXECUTION ASSESSMENT`).
   - Implements fault-tolerant error containment: exceptions transition to `ERROR` and persist context before gracefully winding down to `SLEEP`.

   - Embeds standard Python `Toolbox` with 3 zero-dependency capabilities:
     - `web_browser_tool(url)`: Fetches and cleans textual content from public RSS feeds, job boards, or websites via `urllib.request`.
     - `file_writer_tool(filepath, content)`: Writes scripts, portfolios, reports, or application files into the local workspace with directory traversal containment.
     - `http_poster_tool(url, payload)`: Sends HTTP POST requests to simulate external platform interactions, registrations, or work submissions.
   - Implements a production-grade 4-provider `LLMClient` Fallback Priority Ledger (Groq -> Gemini -> Mistral -> OpenRouter). On HTTP 429/400-class errors or timeouts it prints "[WARNING] Active provider failed with error. Diverting prompt state to the next priority engine backup..." and replays the exact prompt against the next engine.
   - Instructs the agent with an open-ended discovery objective: analyze long-term memory, choose lucrative or educational digital tasks, execute tools dynamically, and log successes/failures for continuous self-improvement.


4. **`cloud_memory_bridge.py`**:
   - Implements `GitHubGistMemoryBridge` subclassing `BaseMemoryBridge` using Python's standard `urllib.request` (zero third-party dependencies).
   - Reads and writes short-term JSON states and append-only `.jsonl` audit logs directly to/from a private GitHub Gist via Bearer Authentication (`CUSTOM_GIT_TOKEN`).
   - Includes graceful fault tolerance: transparently mirrors updates locally and safely falls back to `LocalFileMemoryBridge` on network or credential failures.

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `AGENT_ID` | `default_stateless_agent` | Identifier for the agent instance. |
| `MAX_STEPS_PER_RUN` | `5` | Maximum atomic steps allowed per invocation. |
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

When the agent detects a task that touches external platforms, money movement, or asset publishing, it pauses for your approval:

1. The FSM enters `PENDING_APPROVAL` and persists a `pending_proposal` JSON dictionary inside the Gist state with: `task_id`, `type` (Freelance Pitch / Code Asset Sale / Trend Report), `target_platform`, `proposed_action`, `generated_files`, and `estimated_revenue`.
2. The moment the proposal is created, the zero-dependency `NotificationManager` fires instant alerts (before any SLEEP phase):
   - **Discord** (`DISCORD_WEBHOOK_URL`): an embedded approval card with Task ID, Platform, Estimated Revenue, and a direct Vercel Dashboard link (`DASHBOARD_URL`, default `https://autonomous-agent.vercel.app`).
   - **Email** (`SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, optional `NOTIFY_EMAIL`): a summary alert that a task awaits review (uses `smtplib`, STARTTLS on 587 / SSL on 465).
3. On the next wake-up the agent reads `human_decision` from the persisted state:
   - `null` $\rightarrow$ immediately `SLEEP` without executing (waiting for you).
   - `"APPROVED"` $\rightarrow$ proceeds to the execution block with the stored plan.
   - `"DENIED"` $\rightarrow$ clears the task and returns to planning.
4. To decide, edit your Gist state JSON and set `human_decision` inside `context_data` to `APPROVED` or `DENIED`.

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
- **Proposal render engine** — uses the GitHub Gists REST API (`GET /gists/{id}`, `PATCH /gists/{id}`) with `async/await`; when the agent state is `PENDING_APPROVAL` a prominent alert card shows Task ID, platform, proposed action, generated-file code previews, and expected revenue.
- **1-button controls** — `🟢 APPROVE TASK` / `🔴 DENY TASK` patch `context_data.human_decision` (`APPROVED` / `DENIED`) in the Gist and confirm that the cloud runner will act on the next 30-minute cron cycle.

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
node dashboard_smoke_test.js   # 15/15 checks: render + APPROVE patch flow (uses a stubbed Gist)
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

