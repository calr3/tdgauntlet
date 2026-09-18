"use strict";

// The page behind tdg-web. It shows each position the tournament is waiting on and sends back
// the move chosen. Positions come complete from the server; the only thing kept between them is
// the board after each of our moves, per game, to highlight what the opponent changed.

const $ = (id) => document.getElementById(id);

const ui = {
  name: "tdgauntlet",
  pending: [],          // views from the last snapshot, oldest first
  receivedAt: new Map(), // view id -> performance.now() when first seen
  currentId: null,
  selection: [],        // squares clicked so far
  flipped: false,
  busy: false,
  lastPlayed: null,     // {view, answered} for the idle screen; answered is null on resigning
};

const AFTER_KEY = "tdg-web:after-move";
const afterMove = JSON.parse(sessionStorage.getItem(AFTER_KEY) || "{}");

function rememberAfterMove(gameId, state) {
  afterMove[gameId] = state;
  sessionStorage.setItem(AFTER_KEY, JSON.stringify(afterMove));
}

function current() {
  return ui.pending.find((v) => v.id === ui.currentId) || null;
}

// ---------------------------------------------------------------------------------------------
// Games the page can draw. Anything else falls back to the server's text diagram.

const epaminondas = {
  width: 14,
  height: 12,
  cols: "abcdefghijklmn",

  label(s) {
    return this.cols[s % this.width] + (Math.floor(s / this.width) + 1);
  },

  summary(view) {
    const s = view.state;
    const count = (c) => [...s.board].filter((x) => x === c).length;
    return `move ${s.moves + 1} · quiet ${s.quiet_moves}/100 · pieces W ${count("W")} / B ${count("B")}`;
  },

  sideName(side) {
    return side === 0 ? "White" : "Black";
  },

  // Draws `state`. With `play`, the board is interactive for the side to move.
  draw(container, state, { bottom, previous, play }) {
    const { width, height } = this;
    const board = state.board;
    const rows = [...Array(height).keys()];
    const cols = [...Array(width).keys()];
    if (bottom === 0) rows.reverse(); else cols.reverse();

    const grid = document.createElement("div");
    grid.className = "board" + (play ? "" : " waiting-board");
    grid.style.setProperty("--cols", width);
    grid.style.setProperty("--rows", height);
    const label = (text) => {
      const el = document.createElement("div");
      el.className = "label";
      el.textContent = text;
      grid.append(el);
    };
    const letters = () => {
      label("");
      cols.forEach((c) => label(this.cols[c]));
      label("");
    };

    const hints = play ? play.hints() : { selected: [], partners: [], targets: [] };
    letters();
    for (const r of rows) {
      label(r + 1);
      for (const c of cols) {
        const s = r * width + c;
        const sq = document.createElement("button");
        sq.type = "button";
        sq.className = "sq";
        sq.dataset.square = this.label(s);
        if ((r + c) % 2) sq.classList.add("dark");
        if (r === 0) sq.classList.add("home-0");
        if (r === height - 1) sq.classList.add("home-1");
        if (previous && previous[s] !== board[s]) sq.classList.add("changed");
        if (hints.selected.includes(s)) sq.classList.add("selected");
        if (hints.partners.includes(s)) sq.classList.add("partner");
        if (hints.targets.includes(s)) {
          sq.classList.add("target");
          if (board[s] !== ".") sq.classList.add("capture");
        }
        if (board[s] !== ".") {
          const piece = document.createElement("div");
          piece.className = "piece " + board[s];
          sq.append(piece);
        }
        sq.title = this.label(s);
        if (play) sq.addEventListener("click", () => play.click(s));
        else sq.tabIndex = -1;
        grid.append(sq);
      }
      label(r + 1);
    }
    letters();
    container.replaceChildren(grid);
  },

  // Click handling: pick a piece (the rear or the lead of what moves), optionally the other end
  // of its phalanx, then where it goes. Clicking a destination straight away works too when only
  // one move can reach it.
  controller(view) {
    const own = view.context.side === 0 ? "W" : "B";
    const board = view.state.board;
    const moves = view.moves;
    const ends = (m) => [m.move.rear, m.move.lead];
    const involves = (m, s) => ends(m).includes(s);
    const spans = (m, a, b) =>
      a !== b && ((m.move.rear === a && m.move.lead === b) || (m.move.rear === b && m.move.lead === a));

    const candidates = () => {
      const sel = ui.selection;
      if (sel.length === 1) return moves.filter((m) => involves(m, sel[0]));
      if (sel.length === 2) return moves.filter((m) => spans(m, sel[0], sel[1]));
      return [];
    };

    return {
      candidates,
      hints() {
        const sel = ui.selection;
        const partners = sel.length === 1
          ? moves.filter((m) => involves(m, sel[0]) && m.move.rear !== m.move.lead)
            .flatMap(ends).filter((s) => s !== sel[0])
          : [];
        return {
          selected: sel,
          partners: [...new Set(partners)],
          targets: [...new Set(candidates().map((m) => m.move.dest))],
        };
      },
      click(s) {
        hideChooser();
        const sel = ui.selection;
        if (board[s] === own) {
          if (sel.includes(s)) {
            ui.selection = sel.filter((x) => x !== s);
          } else if (sel.length === 1 && moves.some((m) => spans(m, sel[0], s))) {
            ui.selection = [sel[0], s];
          } else if (moves.some((m) => involves(m, s))) {
            ui.selection = [s];
          } else {
            ui.selection = [];
            say(`${epaminondas.label(s)} cannot lead or bring up the rear of any move.`, true);
          }
          render();
          return;
        }
        const pool = sel.length ? candidates() : moves;
        const matches = pool.filter((m) => m.move.dest === s);
        if (matches.length === 1) {
          submit(matches[0]);
        } else if (matches.length > 1) {
          choose(matches);
        } else {
          ui.selection = [];
          if (sel.length) say(`Nothing selected can reach ${epaminondas.label(s)}.`, true);
          render();
        }
      },
    };
  },
};

const games = { "epaminondas/1": epaminondas };

// ---------------------------------------------------------------------------------------------
// Rendering.

function say(text, bad = false) {
  const el = $("message");
  el.textContent = text;
  el.classList.toggle("bad", bad);
}

function hideChooser() {
  $("chooser").hidden = true;
}

function choose(matches) {
  const el = $("chooser");
  el.replaceChildren();
  for (const m of matches) {
    const b = document.createElement("button");
    b.textContent = m.notation;
    b.addEventListener("click", () => submit(m));
    el.append(b);
  }
  const cancel = document.createElement("button");
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => { hideChooser(); ui.selection = []; render(); });
  el.append(cancel);
  el.hidden = false;
  say("More than one move ends there: which?");
}

function formatWait(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function waited(view) {
  const since = ui.receivedAt.get(view.id);
  return since === undefined ? view.waited_ms : performance.now() - since;
}

function renderSummary(view) {
  const el = $("summary");
  const g = games[view.game];
  const side = g ? g.sideName(view.context.side) : `side ${view.context.side}`;
  const budget = view.budget.ms != null ? ` · budget ${(view.budget.ms / 1000).toFixed(1)} s` : "";
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.textContent = `Your move as ${side}`;
  const detail = document.createElement("div");
  detail.className = "detail";
  detail.textContent =
    `${view.context.game_id} (${view.context.match_id}) · ply ${view.context.ply}` +
    (g ? ` · ${g.summary(view)}` : "") + budget;
  const clock = document.createElement("div");
  clock.className = "detail clock";
  clock.id = "clock";
  clock.textContent = `waiting ${formatWait(waited(view))}`;
  el.replaceChildren(turn, detail, clock);
}

function renderIdle() {
  const wrap = $("board-wrap");
  const last = ui.lastPlayed;
  $("summary").replaceChildren();
  if (!last) {
    wrap.replaceChildren();
    say("Waiting for the tournament to ask for a move…");
    return;
  }
  const g = games[last.view.game];
  const bottom = last.view.context.side ^ (ui.flipped ? 1 : 0);
  if (!last.answered) {
    if (g) g.draw(wrap, last.view.state, { bottom });
    else drawText(wrap, last.view.text);
    say(`You resigned ${last.view.context.game_id}. Waiting for the next position…`);
    return;
  }
  if (g) g.draw(wrap, last.answered.state, { bottom, previous: last.view.state.board });
  else drawText(wrap, last.answered.text);
  const o = last.answered.outcome;
  let text = `You played ${last.answered.notation}.`;
  if (o && o.result === "win") {
    const won = o.side === last.view.context.side;
    text += ` The game is over: you ${won ? "won" : "lost"}.`;
  } else if (o) {
    text += " The game is over: a draw.";
  } else {
    text += " Waiting for the next position…";
  }
  say(text);
}

function drawText(container, text) {
  const pre = document.createElement("pre");
  pre.className = "text";
  pre.textContent = text;
  container.replaceChildren(pre);
}

function normalise(s) {
  return s.trim().toLowerCase().replaceAll("x", "-");
}

// Moves matching what has been typed, as the keyboard client reads it: exact, else by prefix.
function typed(view, text) {
  const want = normalise(text);
  if (!want) return null;
  const exact = view.moves.filter((m) => normalise(m.notation) === want);
  if (exact.length === 1) return exact;
  return view.moves.filter((m) => normalise(m.notation).startsWith(want));
}

function renderMoves(view, controller) {
  const list = $("moves");
  list.replaceChildren();
  if (!view) {
    $("move-count").textContent = "";
    return;
  }
  let shown = typed(view, $("notation").value);
  if (shown === null) {
    const picked = controller ? controller.candidates() : [];
    shown = ui.selection.length ? picked : view.moves;
  }
  $("move-count").textContent = `(${shown.length} of ${view.moves.length})`;
  for (const m of shown) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.textContent = m.notation;
    b.addEventListener("click", () => submit(m));
    li.append(b);
    list.append(li);
  }
}

function renderQueue() {
  const q = $("queue");
  q.replaceChildren();
  $("count").textContent = ui.pending.length ? `(${ui.pending.length})` : "";
  if (!ui.pending.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "nothing";
    q.append(li);
  }
  for (const v of ui.pending) {
    const li = document.createElement("li");
    if (v.id === ui.currentId) li.className = "current";
    li.textContent = v.context.game_id;
    const d = document.createElement("span");
    d.className = "detail";
    d.textContent = ` ply ${v.context.ply}`;
    li.append(d);
    li.addEventListener("click", () => {
      ui.currentId = v.id;
      ui.selection = [];
      hideChooser();
      say("");
      render();
    });
    q.append(li);
  }
}

function render() {
  const view = current();
  document.title = ui.pending.length ? `(${ui.pending.length}) Your move · ${ui.name}` : ui.name;
  $("who").textContent = ui.name;
  renderQueue();
  for (const el of $("entry").elements) el.disabled = !view || ui.busy;
  if (!view) {
    renderMoves(null);
    renderIdle();
    return;
  }
  renderSummary(view);
  const g = games[view.game];
  const wrap = $("board-wrap");
  let controller = null;
  if (g) {
    controller = g.controller(view);
    const prev = afterMove[view.context.game_id];
    g.draw(wrap, view.state, {
      bottom: view.context.side ^ (ui.flipped ? 1 : 0),
      previous: prev && prev.moves < view.state.moves ? prev.board : null,
      play: ui.busy ? null : controller,
    });
  } else {
    drawText(wrap, view.text);
  }
  renderMoves(view, controller);
}

// ---------------------------------------------------------------------------------------------
// Talking to tdg-web.

async function submit(m) {
  const view = current();
  if (!view || ui.busy) return;
  ui.busy = true;
  hideChooser();
  say(`Playing ${m.notation}…`);
  render();
  try {
    const r = await fetch("api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: view.id, move: m.move }),
    });
    const body = await r.json().catch(() => ({}));
    if (r.ok) {
      rememberAfterMove(view.context.game_id, body.state);
      ui.lastPlayed = { view, answered: body };
      dropView(view.id);
      say(`You played ${body.notation}.`);
    } else {
      if (r.status === 410) dropView(view.id);
      say(body.error || `The move was refused (${r.status}).`, true);
    }
  } catch (e) {
    say(`Could not reach tdg-web: ${e}`, true);
  } finally {
    ui.busy = false;
    ui.selection = [];
    $("notation").value = "";
    render();
  }
}

async function resign() {
  const view = current();
  if (!view || ui.busy) return;
  if (!confirm(`Resign ${view.context.game_id}?`)) return;
  ui.busy = true;
  try {
    const r = await fetch("api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: view.id, resign: true }),
    });
    dropView(view.id);
    if (r.ok) {
      ui.lastPlayed = { view, answered: null };
      say(`You resigned ${view.context.game_id}.`);
    } else {
      say("That position was no longer waiting.", true);
    }
  } catch (e) {
    say(`Could not reach tdg-web: ${e}`, true);
  } finally {
    ui.busy = false;
    render();
  }
}

function dropView(id) {
  ui.pending = ui.pending.filter((v) => v.id !== id);
  if (ui.currentId === id) ui.currentId = ui.pending.length ? ui.pending[0].id : null;
}

function onSnapshot(snap) {
  ui.name = snap.name;
  const before = ui.currentId;
  ui.pending = snap.pending;
  const now = performance.now();
  for (const v of snap.pending) if (!ui.receivedAt.has(v.id)) ui.receivedAt.set(v.id, now - v.waited_ms);
  for (const id of ui.receivedAt.keys()) if (!snap.pending.some((v) => v.id === id)) ui.receivedAt.delete(id);
  if (!current()) ui.currentId = snap.pending.length ? snap.pending[0].id : null;
  if (ui.currentId !== before) {
    ui.selection = [];
    hideChooser();
    if (ui.currentId !== null) say("");
  }
  render();
}

function connection(ok) {
  const el = $("connection");
  el.textContent = ok ? "connected" : "reconnecting…";
  el.classList.toggle("bad", !ok);
}

async function poll() {
  let since = null;
  for (;;) {
    try {
      const r = await fetch("api/pending" + (since === null ? "" : `?since=${since}`));
      if (!r.ok) throw new Error(r.status);
      const snap = await r.json();
      connection(true);
      if (snap.version !== since) onSnapshot(snap);
      since = snap.version;
    } catch {
      connection(false);
      since = null;
      await new Promise((ok) => setTimeout(ok, 2000));
    }
  }
}

// ---------------------------------------------------------------------------------------------

$("entry").addEventListener("submit", (e) => {
  e.preventDefault();
  const view = current();
  if (!view) return;
  const text = $("notation").value;
  const found = typed(view, text);
  if (!found || !found.length) say(text.trim() ? `${text.trim()} is not a legal move here.` : "Type a move.", true);
  else if (found.length === 1) submit(found[0]);
  else say(`${text.trim()} could be any of: ${found.map((m) => m.notation).join(", ")}`, true);
});

$("notation").addEventListener("input", () => {
  const view = current();
  if (view) renderMoves(view, games[view.game]?.controller(view));
});

$("flip").addEventListener("click", () => {
  ui.flipped = !ui.flipped;
  render();
});

$("resign").addEventListener("click", resign);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    ui.selection = [];
    hideChooser();
    render();
  }
});

setInterval(() => {
  const view = current();
  const clock = $("clock");
  if (view && clock) clock.textContent = `waiting ${formatWait(waited(view))}`;
}, 500);

render();
poll();
