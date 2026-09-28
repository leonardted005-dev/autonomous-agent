/* Temporary functional smoke test for index.html dashboard logic (Node, no browser). */
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync("index.html", "utf8");
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const appJs = blocks[blocks.length - 1];

const STATE = {
  run_id: "run_test_1",
  cycle_count: 14,
  current_state: "PENDING_APPROVAL",
  active_plan: ["1. Submit freelance pitch to Upwork"],
  context_data: {
    pending_proposal: {
      task_id: "abc123def456",
      type: "Freelance Pitch",
      target_platform: "Upwork",
      proposed_action: "Submit pitch to Upwork requesting $250 payment",
      generated_files: [{ name: "pitch.md", content: "# Pitch\n\nHello Upwork." }],
      estimated_revenue: "$250",
    },
    human_decision: null,
  },
  last_updated: "2026-09-28T00:00:00Z",
};

const GIST = {
  id: "gist_test",
  files: {
    "default_stateless_agent_state.json": {
      filename: "default_stateless_agent_state.json",
      content: JSON.stringify(STATE),
      truncated: false,
    },
  },
};

function makeEl(id) {
  const classes = new Set();
  return {
    id,
    value: "",
    type: "text",
    textContent: "",
    _innerHTML: "",
    disabled: false,
    children: [],
    set innerHTML(v) {
      this._innerHTML = String(v);
      this.children = []; // mirror browser behaviour: innerHTML="" clears children
    },
    get innerHTML() {
      return this._innerHTML;
    },
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c) => (classes.has(c) ? (classes.delete(c), true) : (classes.add(c), false)),
      contains: (c) => classes.has(c),
    },
    set className(v) {
      classes.clear();
      String(v).split(/\s+/).filter(Boolean).forEach((c) => classes.add(c));
      this._className = v;
    },
    get className() {
      return this._className || "";
    },
    addEventListener() {},
    appendChild(child) {
      this.children.push(child);
    },
  };
}

const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeEl(id));
    return elements.get(id);
  },
  createElement(tag) {
    return makeEl("created:" + tag);
  },
};

const store = new Map([
  ["hitl_gist_id", "gist_test"],
  ["hitl_git_token", "token_test"],
]);
const localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};

let patchedBody = null;
async function fetchStub(url, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  if (method === "GET") return { ok: true, status: 200, json: async () => GIST };
  if (method === "PATCH") {
    patchedBody = JSON.parse(options.body);
    return { ok: true, status: 200, json: async () => ({}) };
  }
  throw new Error("unexpected method " + method);
}

const sandbox = {
  document,
  localStorage,
  fetch: fetchStub,
  setInterval: () => 0,
  clearTimeout: () => {},
  setTimeout: (fn) => {
    try { fn(); } catch (_) {}
    return 0;
  },
  console,
  Date,
  JSON,
  Array,
  String,
  Number,
  Object,
  encodeURIComponent,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(appJs, sandbox);

const results = [];
function check(name, condition, extra) {
  results.push({ name, ok: !!condition, extra: extra === undefined ? "" : String(extra) });
}

function run() {
  return new Promise((resolve) => setTimeout(resolve, 50));
}

(async () => {
  await run();

  const card = elements.get("proposalCard");
  const empty = elements.get("emptyState");
  check("proposal card visible", card && !card.classList.contains("hidden"));
  check("empty state hidden", empty && empty.classList.contains("hidden"));
  check("task id rendered", elements.get("proposalTaskId").textContent === "abc123def456", elements.get("proposalTaskId").textContent);
  check("platform rendered", elements.get("proposalPlatform").textContent === "Upwork", elements.get("proposalPlatform").textContent);
  check("revenue rendered", elements.get("proposalRevenue").textContent === "$250", elements.get("proposalRevenue").textContent);
  check("cycle metric rendered", elements.get("metricCycles").textContent === 14, elements.get("metricCycles").textContent);
  check("state badge rendered", elements.get("stateBadge").textContent === "PENDING_APPROVAL", elements.get("stateBadge").textContent);
  check("tracker chips drawn", elements.get("tracker").children.length === 6, elements.get("tracker").children.length);
  check("file preview drawn", elements.get("proposalFiles").children.length === 1, elements.get("proposalFiles").children.length);
  check("decision buttons visible", !elements.get("decisionArea").classList.contains("hidden"));

  await sandbox.decide("APPROVED");
  await run();

  check("PATCH sent to gist", patchedBody && patchedBody.files, JSON.stringify(patchedBody).slice(0, 120));
  const patchedState = patchedBody && patchedBody.files["default_stateless_agent_state.json"]
    ? JSON.parse(patchedBody.files["default_stateless_agent_state.json"].content)
    : null;
  check("human_decision = APPROVED", patchedState && patchedState.context_data.human_decision === "APPROVED", patchedState && patchedState.context_data.human_decision);
  check("proposal preserved in patch", patchedState && patchedState.context_data.pending_proposal.task_id === "abc123def456");
  check("success state shown", !elements.get("decisionSuccess").classList.contains("hidden"));
  check("success text mentions cron", /30-minute cron/.test(elements.get("decisionSuccessText").textContent), elements.get("decisionSuccessText").textContent);

  const failed = results.filter((r) => !r.ok);
  results.forEach((r) => console.log((r.ok ? "PASS  " : "FAIL  ") + r.name + (r.ok ? "" : "  -> " + r.extra)));
  console.log("\n" + (results.length - failed.length) + "/" + results.length + " dashboard checks passed");
  process.exit(failed.length ? 1 : 0);
})();
