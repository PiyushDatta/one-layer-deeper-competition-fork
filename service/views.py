"""Server-rendered HTML for the prototype."""

from __future__ import annotations

from html import escape
import json

from .competition import (
    SUBMISSION_DEADLINE_LABEL,
    SUBMISSIONS_CLOSED_MESSAGE,
)
from .tiers import TIERS, TIER_BY_ID


def _layout(title: str, content: str, *, refresh: bool = False) -> str:
    refresh_tag = '<meta http-equiv="refresh" content="5">' if refresh else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_tag}
  <title>{escape(title)} · One Layer Deeper</title>
  <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header class="shell masthead">
    <a class="brand" href="/"><span class="brand-mark">1L↓</span> One Layer Deeper</a>
    <nav><a href="/">Leaderboard</a><a href="/problem">Problem</a><a href="/register">Sign in</a><a class="button small" href="/submit">Submit</a></nav>
  </header>
  <aside class="shell competition-deadline" aria-label="Competition submission deadline">
    <strong>Submission deadline:</strong> {escape(SUBMISSION_DEADLINE_LABEL)}.
  </aside>
  <main class="shell">{content}</main>
  <footer class="shell">
    <strong>Core Automation × Tilde Research</strong><br>
    Fixed model state. Variable architecture and optimizer. Evaluated on one H100.
    <span class="support">With thanks to <a href="https://modal.com/">Modal</a> for supporting GPU evaluations and <a href="https://northflank.com/">Northflank</a> for supporting the competition service and leaderboard.</span>
    <span class="support"><strong>Competition updates:</strong> join <a href="https://discord.gg/gpumode">discord.gg/gpumode</a> and follow <code>#one-layer-deeper</code>.</span>
  </footer>
</body>
</html>"""


def problem_page() -> str:
    content = """
    <section class="problem-hero">
      <div>
        <p class="eyebrow">The benchmark problem</p>
        <h1>Square it.<br><em>Then square it again.</em></h1>
        <p class="lede">Given a modulus <var>N</var>, a starting value <var>x</var>, and a step count <var>T</var>, predict the residue after squaring modulo <var>N</var> exactly <var>T</var> times.</p>
      </div>
      <div class="formula-card" aria-label="Repeated modular squaring recurrence">
        <span class="formula-label">Recurrence</span>
        <div><var>x</var><sub>0</sub> = <var>x</var> mod <var>N</var></div>
        <div><var>x</var><sub>t</sub> = <var>x</var><sub>t−1</sub><sup>2</sup> mod <var>N</var></div>
        <div class="formula-result"><var>y</var> = <var>x</var><sub>T</sub> = <var>x</var><sup>2<sup>T</sup></sup> mod <var>N</var></div>
      </div>
    </section>

    <section class="panel problem-example">
      <div class="panel-head"><h2>A small example</h2><span>N = 77 · x = 2 · T = 4</span></div>
      <div class="squaring-steps">
        <div><span>Start</span><strong>2</strong><code>x₀</code></div>
        <i aria-hidden="true">→</i>
        <div><span>Square 1</span><strong>4</strong><code>2² mod 77</code></div>
        <i aria-hidden="true">→</i>
        <div><span>Square 2</span><strong>16</strong><code>4² mod 77</code></div>
        <i aria-hidden="true">→</i>
        <div><span>Square 3</span><strong>25</strong><code>16² mod 77</code></div>
        <i aria-hidden="true">→</i>
        <div class="answer-step"><span>Square 4</span><strong>9</strong><code>25² mod 77</code></div>
      </div>
      <p class="example-caption">The answer is <strong>9</strong>. Each arrow is one serial step: the next value depends on the residue produced by the previous square.</p>
    </section>

    <section class="problem-grid">
      <article class="panel problem-copy">
        <div class="panel-head"><h2>What the model sees</h2><span>Decimal tokens</span></div>
        <div class="problem-card-body">
          <p>Prompts use field markers followed by decimal digits. The example above is represented conceptually as:</p>
          <div class="token-row" aria-label="N 77, X 2, T 4, answer 9">
            <b>N</b><span>7</span><span>7</span><b>X</b><span>2</span><b>T</b><span>4</span><b>ANS</b><span class="answer-token">9</span>
          </div>
          <p><var>N</var> is part of every prompt. Its secret prime factors <var>p</var> and <var>q</var> are never supplied to the submission. The target is the decimal representation of <var>y</var>.</p>
        </div>
      </article>

      <article class="panel problem-copy">
        <div class="panel-head"><h2>How it is scored</h2><span>Exact accuracy</span></div>
        <div class="problem-card-body">
          <p>An example counts as correct only when every target token is correct. A nearly right residue earns no partial credit for that example.</p>
          <p>Hard uses a private hidden evaluator. Easy and Medium equally average test and their merged out-of-distribution split.</p>
        </div>
      </article>
    </section>

    <section class="panel why-panel">
      <div class="panel-head"><h2>Why this tests depth</h2><span>Serial · exact · extrapolative</span></div>
      <div class="why-grid">
        <article><span>01</span><h3>Inherently serial</h3><p>Without the factorization of <var>N</var>, the best known general method carries out the <var>T</var> modular squarings in order. Later states depend on earlier ones.</p></article>
        <article><span>02</span><h3>Fast, exact labels</h3><p>The evaluator generates <var>N = pq</var> and keeps <var>p</var> and <var>q</var> private. It can reduce the exponent with φ(<var>N</var>) and compute exact answers efficiently.</p></article>
        <article><span>03</span><h3>More work for harder inputs</h3><p>Larger <var>T</var> asks for a longer computation. This makes the task a clean test of recurrent depth, adaptive computation, and extrapolation beyond training depths.</p></article>
      </div>
      <div class="trapdoor">
        <div><span>Evaluator-only shortcut</span><code>φ(N) = (p − 1)(q − 1)</code></div>
        <div><span>Reduce the exponent</span><code>e = 2<sup>T</sup> mod φ(N)</code></div>
        <div><span>Produce the label</span><code>y = x<sup>e</sup> mod N</code></div>
      </div>
    </section>

    <section class="problem-cta">
      <div><p class="eyebrow">The design question</p><h2>Can a learned layer keep thinking for longer?</h2></div>
      <a class="button" href="/submit">Build your answer <span>↗</span></a>
    </section>"""
    return _layout("Problem", content)


def leaderboard_page(rows: list[dict]) -> str:
    ranks = {str(row["id"]): rank for rank, row in enumerate(rows, start=1)}
    table_rows = []
    for row in rows:
        rank = ranks.get(str(row["id"]), "—")
        score = "—" if row["score"] is None else f"{100 * row['score']:.2f}%"
        submitter = row["submitter"]
        table_rows.append(
            f"""<tr>
              <td class="rank">{rank}</td>
              <td><a class="entry" href="/submissions/{row["id"]}">{escape(submitter)}</a>
                  <span class="muted block">{escape(row["filename"])}</span></td>
              <td class="score">{score}</td>
              <td class="muted">{escape(str(row.get("dataset_label") or row["manifest_name"]))}</td>
            </tr>"""
        )
    empty = "" if table_rows else '<div class="empty">No submissions yet.</div>'
    rules_link = '<a class="rules-link" href="https://github.com/tilde-research/one-layer-deeper">Detailed rules &amp; criteria on GitHub <span>↗</span></a>'
    content = f"""
    <section class="hero">
      <div><p class="eyebrow">Core Automation × Tilde Research present</p>
      <h1>Make one layer<br><em>think deeper.</em></h1>
      <p class="lede">One file. One H100. One fixed model-state ceiling. Use the training-time budget however you want, then get one final score.</p></div>
      <div class="hero-actions">
        <a class="button hero-button" href="/submit">Submit an entry <span>↗</span></a>
        {rules_link}
      </div>
    </section>
    <section class="panel">
      <div class="panel-head"><h2>Leaderboard</h2><span>{len(rows)} ranked</span></div>
      <div class="notice"><strong>Hard leaderboard:</strong> each participant is ranked once by their best successful Hard score. Easy and Medium are private practice runs; source and run details remain private.</div>
      <div class="table-wrap"><table>
        <thead><tr><th>#</th><th>Participant / file</th><th>Exact accuracy</th><th>Dataset</th></tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table></div>{empty}
    </section>"""
    return _layout("Leaderboard", content)


def submit_page(
    error: str | None = None,
    *,
    selected_tier: str = "easy",
    selected_dataset: str | None = None,
    submissions_are_closed: bool = False,
) -> str:
    if submissions_are_closed and error is None:
        error = SUBMISSIONS_CLOSED_MESSAGE
    error_html = f'<div class="error">{escape(error)}</div>' if error else ""
    tier = TIER_BY_ID.get(selected_tier, TIER_BY_ID["easy"])
    valid_dataset_ids = {dataset.id for dataset in tier.datasets}
    dataset_id = (
        selected_dataset
        if selected_dataset in valid_dataset_ids
        else tier.datasets[0].id
    )
    tier_options = "".join(
        f'<option value="{item.id}"{" selected" if item.id == tier.id else ""}>'
        f'{escape(item.label)} · {item.training_seconds // 60} min · '
        f'{item.daily_attempts}/day</option>'
        for item in TIERS
    )
    dataset_options = "".join(
        f'<option value="{item.id}"{" selected" if item.id == dataset_id else ""}>'
        f'{escape(item.label)}</option>'
        for item in tier.datasets
    )
    dataset_catalog = json.dumps(
        {
            item.id: [
                {"id": dataset.id, "label": dataset.label}
                for dataset in item.datasets
            ]
            for item in TIERS
        }
    )
    dataset_hidden = " hidden" if tier.id == "hard" else ""
    control_disabled = " disabled" if submissions_are_closed else ""
    dataset_disabled = (
        " disabled" if tier.id == "hard" or submissions_are_closed else ""
    )
    upload_card_class = (
        "upload-card submissions-closed"
        if submissions_are_closed
        else "upload-card"
    )
    submit_label = (
        "Submissions closed"
        if submissions_are_closed
        else "Queue H100 evaluation"
    )
    content = f"""
    <section class="submit-grid">
      <div><p class="eyebrow">Single-file contract</p><h1>Ship the idea,<br><em>not the pipeline.</em></h1>
      <p class="lede">Upload one <code>submission.py</code>, then choose a practice dataset or the ranked Hard evaluation.</p>
      <aside class="beta-notice"><strong>Beta:</strong> We're refining the competition and would love your feedback before we finalize the rules in the near future. Share your thoughts in the <a href="https://discord.gg/gpumode">GPU MODE Discord</a> <code>#one-layer-deeper</code> channel.</aside>
      <div class="rules"><span>01</span><p><strong>Artifact:</strong> one UTF-8 file named <code>submission.py</code>, up to 256 KiB. Imports may use the public benchmark API and pinned evaluator dependencies—no extra files, repository implementation modules, or installs.</p>
      <span>02</span><p><strong>Model factory:</strong> receives only tensor shapes, I/O requirements, and the model-state ceiling.</p>
      <span>03</span><p><strong>Training loss:</strong> optionally turn final logits, auxiliary outputs, and current labels into one scalar loss. The evaluator performs one backward pass.</p>
      <span>04</span><p><strong>Optimizer factory:</strong> receives the model, per-seed H100 time allowance, and device type. Include every trainable parameter exactly once; custom optimizers and schedules are welcome.</p>
      <span>05</span><p><strong>Use the whole machine:</strong> optimizer state, activations, memory tokens, and temporary workspace may use available VRAM. An OOM or timeout fails the run; only persistent model state is capped.</p>
      <span>06</span><p><strong>Evaluator boundary:</strong> choose a training and evaluation batch size and an optional lower step limit; the evaluator still fixes data, sampling, one forward, one backward, clipping, optimizer cadence, deadline, final evaluation, and score. Final evaluation has a separate time budget equal to half the training allowance.</p></div>
      <section class="submission-rules" aria-labelledby="submission-rules-heading">
        <h2 id="submission-rules-heading">Rules</h2>
        <div class="rules">
          <span>01</span><p><strong>Maximum 500 million trainable parameters.</strong></p>
          <span>02</span><p><strong>No hard-coded weights.</strong> Trainable weights must use a random initialization and be updated during training. For example, <code>torch.load</code> is not allowed.</p>
          <span>03</span><p><strong>No hard-coded algorithm in the forward pass.</strong> Outputs must be produced by the learned model.</p>
          <span>04</span><p><strong>End-to-end learning only.</strong> Final logits must be produced entirely by the submitted model from its inputs and learned PyTorch state, with all input-dependent computation inside the autograd graph and an unbroken gradient path from the loss to the parameters responsible for the prediction.</p>
          <span>05</span><p><strong>Everything stays on the GPU.</strong> Model state and computation must remain on the GPU throughout training and evaluation; CPU offloading is not allowed.</p>
          <span>06</span><p><strong>Repeated rule-breaking will get you banned.</strong> We still encourage creativity: discussing possible loopholes on Discord or testing one in a submission won't get you banned.</p>
          <span>07</span><p><strong>The metric recorder for a Hard run must not be exploited.</strong> Any attempt to exploit it will result in an immediate ban.</p>
        </div>
      </section></div>
      <form class="{upload_card_class}" action="/submit" method="post" enctype="multipart/form-data">
        <div><span class="file-kicker">PY</span><h2>Upload submission.py</h2><p>One self-contained Python file, up to 256 KiB.</p></div>
        {error_html}
        <label class="text-field"><span>API key</span><input name="api_key" type="password" autocomplete="off" placeholder="old_…" required{control_disabled}></label>
        <label class="text-field"><span>Compute tier</span><select id="tier-select" name="tier" required{control_disabled}>{tier_options}</select></label>
        <label id="dataset-field" class="text-field"{dataset_hidden}><span>Dataset</span><select id="dataset-select" name="dataset"{dataset_disabled}>{dataset_options}</select></label>
        <div class="tier-note"><strong>Daily limits use UTC:</strong> Easy 60 · Medium 6 · Hard 1. Accepted failed runs count; rejected uploads do not.</div>
        <label class="file-field"><span>Choose submission.py</span><input name="file" type="file" accept=".py,text/x-python" required{control_disabled}></label>
        <button class="button" type="submit"{control_disabled}>{submit_label}</button>
        <p class="form-note">No key yet? <a href="/register">Sign in with GitHub</a>. CLI users run <code>one-layer login</code> once; submissions must pass <code>--tier</code> and, for Easy or Medium, <code>--dataset</code>. Only the best successful Hard score per participant is ranked.</p>
      </form>
      <script>
      (() => {{
        const catalog = {dataset_catalog};
        const tier = document.getElementById("tier-select");
        const field = document.getElementById("dataset-field");
        const dataset = document.getElementById("dataset-select");
        tier.addEventListener("change", () => {{
          const isHard = tier.value === "hard";
          field.hidden = isHard;
          dataset.disabled = isHard;
          dataset.required = !isHard;
          dataset.replaceChildren(...catalog[tier.value].map((item) => {{
            const option = document.createElement("option");
            option.value = item.id;
            option.textContent = item.label;
            return option;
          }}));
        }});
        dataset.required = tier.value !== "hard";
      }})();
      </script>
    </section>
    <section class="contract panel"><div class="panel-head"><h2>Contract sketch</h2><a href="/samples/submission.py">Download sample</a></div>
<pre><code>from benchmark import ModelSpec, OptimizerSpec, OptimizerBundle, Submission, assert_model_state

def build_model(spec: ModelSpec):
    model = MyModel(spec)
    assert_model_state(model, spec)  # parameters + persistent buffers
    return model

def build_optimizer(model, spec: OptimizerSpec):
    optimizer = MyOptimizer(model.parameters())
    scheduler = MySchedule(optimizer, spec.training_time_seconds)
    return OptimizerBundle(optimizer, scheduler=scheduler)

def training_loss(logits, labels, auxiliary):
    return my_loss(logits, labels, auxiliary)

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,  # optional
    batch_size=512,  # optional; training
    eval_batch_size=1024,  # optional; evaluation
    max_steps=20_000,  # optional
)</code></pre></section>
    <section class="depth-guide panel">
      <div class="panel-head"><h2>How depth is used</h2><span>Easy 1 min · Medium 10 min · Hard 1 hour</span></div>
      <div class="depth-body">
        <p><strong>You define all of it.</strong> Choose any depth or internal recurrence; return an <code>nn.Module</code> with matching <code>config</code>. The evaluator calls <code>model.train()</code> during optimization and <code>model.eval()</code> for final evaluation; use PyTorch's <code>self.training</code> flag when your forward behavior should differ between them. The evaluator does not supply depth tiers or grade a claimed layer count. Build a fixed-depth network, a recurrent model, a learned halting system, nested refinement, or something we have not named. The benchmark measures the final model it receives.</p>
        <ol class="depth-flow">
          <li><strong>Stay under the model-state ceiling.</strong> The current suite allows at most 500,000,000 parameter and persistent-buffer elements. Shared weights count once.</li>
          <li><strong>Spend the clock.</strong> The deadline includes participant construction and compilation. A cheaper forward gets more optimizer updates; a deeper forward gets fewer.</li>
          <li><strong>Use available VRAM.</strong> Activations, optimizer state, temporary tensors, and workspace do not count as model parameters. If the run OOMs, it fails.</li>
          <li><strong>Evaluate once.</strong> The final checkpoint is measured on the fixed hidden evaluation, producing one leaderboard score.</li>
        </ol>
      </div>
      <div class="depth-question">
        <div><span class="eyebrow">So how is depth graded?</span><h3>It is not graded separately.</h3></div>
        <p>Depth is a means, not the metric. “10,000 layers” earns nothing on its own, so there is no depth-definition reward hack. The competition asks for the best measured accuracy obtainable with the same persistent model-state ceiling and H100 time. The depth-versus-update tradeoff is part of the research.</p>
      </div>
      <div class="example-grid">
        <article class="example"><h3>Go very deep</h3><p>Apply one learned block repeatedly. The weights count once; the runtime does not.</p><pre><code>def refine(self, h):
    for _ in range(self.depth):
        h = self.shared_block(h)
    return h</code></pre></article>
        <article class="example"><h3>Mix shared stages</h3><p>Cycle a fixed bank of blocks, route tokens, or alternate update types.</p><pre><code>def refine(self, h):
    for step in range(self.depth):
        block = self.blocks[step % len(self.blocks)]
        h = block(h)
    return h</code></pre></article>
        <article class="example"><h3>Adapt the compute</h3><p>Use a curriculum, learned halting, or different work per example.</p><pre><code>def refine(self, h):
    for step in range(self.max_depth):
        h = self.update(h)
        if self.should_halt(h).all():
            break
    return h</code></pre></article>
      </div>
      <div class="depth-boundary"><strong>No artificial-depth rule is needed.</strong> Internal sublayers, arbitrary loops, input reinjection, learned memory/state tokens, iterative refinement, routing, reversible updates, cross-step attention, parameter-free work, adaptive halting, custom training losses, and depth curricula are all allowed. Use fewer parameters if useful, but never exceed the published persistent-state ceiling. Data, the evaluator loop, the H100 deadline, final evaluation, and scoring remain fixed.</div>
    </section>"""
    return _layout("Submit", content)


def register_page() -> str:
    content = f"""
    <section class="register-grid">
      <div><p class="eyebrow">Participant access</p><h1>GitHub in.<br><em>Build deeper.</em></h1>
      <p class="lede">GitHub proves who you are. After authorization, the service immediately gives you the API key used by browser and CLI submissions.</p>
      <div class="rules"><span>01</span><p>Your GitHub display name and handle identify leaderboard submissions.</p>
      <span>02</span><p>The full API key is shown once; only its SHA-256 digest and short support prefix are stored.</p>
      <span>03</span><p>Sign in again to rotate a lost key. One queued or running evaluation is allowed per account, and operators can ban abusive GitHub users.</p></div></div>
      <div class="upload-card register-card">
        <div><span class="file-kicker">GH</span><h2>Participant login</h2><p>No competition password, event form, or manually issued token.</p></div>
        <a class="button" href="/auth/github">Continue with GitHub <span>↗</span></a>
        <p class="form-note">Requested GitHub access: public profile and verified email. We do not request repository access.</p>
      </div>
    </section>"""
    return _layout("Sign in", content)


def registered_page(user: dict, api_key: str) -> str:
    content = f"""
    <section class="key-result">
      <p class="eyebrow">GitHub authentication complete</p>
      <h1>Welcome,<br><em>{escape(user['display_name'])}.</em></h1>
      <p class="lede">Authenticated as <strong>@{escape(user['github_login'])}</strong>. Copy this key now. The service cannot recover it because only its digest is stored; signing in again rotates it.</p>
      <div class="key-box"><code>{escape(api_key)}</code></div>
      <pre><code>export ONE_LAYER_API_KEY={escape(api_key)}
one-layer submit submission.py --tier easy --dataset e1 --wait</code></pre>
      <div class="key-actions"><a class="button" href="/submit">Submit in browser</a><a href="/">View leaderboard</a></div>
    </section>"""
    return _layout("API key", content)


def submission_page(row: dict) -> str:
    score = row.get("score")
    score_text = "Pending" if score is None else f"{100 * score:.2f}%"
    submitter = row["submitter"]
    failure = (
        '<div class="error"><strong>Evaluation failed.</strong> The participant can inspect the private error through the authenticated CLI.</div>'
        if row["status"] == "failed"
        else ""
    )
    content = f"""
    <section class="detail-hero"><div><p class="eyebrow">Participant</p><h1>{escape(submitter)}</h1>
    <p class="lede">{escape(row["filename"])}</p></div>
    <div class="score-card"><span>Exact accuracy</span><strong>{score_text}</strong><span class="status {escape(row["status"])}">{escape(row["status"])}</span></div></section>
    {failure}
    <section class="facts panel"><div><span>Participant</span><strong>{escape(submitter)}</strong></div>
    <div><span>Filename</span><strong>{escape(row["filename"])}</strong></div>
    <div><span>Tier</span><strong>{escape(str(row.get("tier") or "Legacy").title())}</strong></div>
    <div><span>Dataset</span><strong>{escape(str(row.get("dataset_label") or row["manifest_name"]))}</strong></div>
    <div><span>Submitted</span><strong>{escape(str(row["created_at"]))}</strong></div></section>
    <section class="contract panel"><div class="panel-head"><h2>Private during competition</h2></div>
    <p>Source and per-seed measurements are visible only to the participant and organizers. Use <code>one-layer status {row["id"]} --json</code> with the saved API key. After a successful run, download the bounded structured log with <code>one-layer metrics {row["id"]}</code>. Raw submission output is never included.</p></section>"""
    return _layout(
        f"{submitter} · {row['filename']}",
        content,
        refresh=row["status"] in {"queued", "running"},
    )
