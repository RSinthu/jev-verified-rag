const TAU = parseFloat(document.body.dataset.tau);

const STATUS_LABELS = {
  answered: "Answered",
  not_in_docs: "Not in the docs",
  injection: "Blocked: prompt injection",
  no_documents: "No documents",
  error: "Error",
};

const THINKING_STEPS = [
  "Searching your documents…",
  "Jev is scoring 20 passages…",
  "Checking the gates…",
  "Writing the answer…",
];

const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("file-input");
const askForm = $("ask-form");
const questionBox = $("question");
const askBtn = $("ask-btn");
const results = $("results");

// ---------- Helpers ----------

/** Create an element: el("div", { class: "x", text: "hi" }, [children]) */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

function toast(message, type = "") {
  const node = el("div", { class: `toast ${type}`, text: message });
  $("toasts").append(node);
  setTimeout(() => node.remove(), 5000);
}

function scoreBar(score, good) {
  const fill = el("div", { class: `bar-fill ${good ? "good" : "bad"}` });
  fill.style.width = `${Math.round(score * 100)}%`;
  const tau = el("div", { class: "bar-tau", title: `τ = ${TAU}` });
  tau.style.left = `${TAU * 100}%`;
  return el("div", { class: "bar" }, [fill, tau]);
}

// ---------- Documents ----------

function renderDocuments(documents) {
  const names = Object.keys(documents).sort();
  $("doc-list").replaceChildren(
    ...names.map((name) =>
      el("li", { class: "doc" }, [
        el("span", { class: "doc-type", text: name.split(".").pop().toUpperCase() }),
        el("span", { class: "doc-name", text: name, title: name }),
        el("span", { class: "doc-chunks", text: `${documents[name]} chunks` }),
      ])
    )
  );
  $("doc-empty").hidden = names.length > 0;
  $("clear-btn").hidden = names.length === 0;
  $("doc-count").textContent = `${names.length} document${names.length === 1 ? "" : "s"}`;
}

async function uploadFiles(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const file of files) form.append("files", file);

  dropzone.classList.add("busy");
  try {
    const data = await api("/api/upload", { method: "POST", body: form });
    renderDocuments(data.documents);
    const indexed = Object.keys(data.indexed);
    if (indexed.length) toast(`Indexed ${indexed.join(", ")}`, "success");
    for (const [name, reason] of Object.entries(data.skipped)) toast(`Skipped ${name}: ${reason}`, "error");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    dropzone.classList.remove("busy");
    fileInput.value = "";
  }
}

fileInput.addEventListener("change", () => uploadFiles([...fileInput.files]));

dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

$("clear-btn").addEventListener("click", async () => {
  if (!confirm("Delete all uploaded documents?")) return;
  try {
    renderDocuments((await api("/api/documents", { method: "DELETE" })).documents);
    toast("All documents cleared");
  } catch (err) {
    toast(err.message, "error");
  }
});

// ---------- Answer ----------

const CITATION = /\[(C\d+(?:\s*,\s*C\d+)*)\]/g;
// $$…$$ and \[…\] are display math; \(…\) and $…$ are inline ($5 and $10 is left alone)
const MATH = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)/g;

/** Markdown + LaTeX -> safe HTML. Returns null if the libraries didn't load (e.g. offline). */
function markdownToHtml(text) {
  if (!window.marked || !window.DOMPurify) return null;

  // Take math out first, otherwise Markdown eats the backslashes in \( \) and \[ \]
  const math = [];
  const withoutMath = text.replace(MATH, (match, block1, block2, inline1, inline2) => {
    const tex = block1 ?? block2 ?? inline1 ?? inline2;
    const displayMode = block1 !== undefined || block2 !== undefined;
    math.push(window.katex ? katex.renderToString(tex.trim(), { displayMode, throwOnError: false }) : match);
    return `MATHPLACEHOLDER${math.length - 1}END`;
  });

  const html = DOMPurify.sanitize(marked.parse(withoutMath));
  // KaTeX output is safe: it escapes the TeX it is given
  return html.replace(/MATHPLACEHOLDER(\d+)END/g, (_, i) => math[i]);
}

/** Turn every [C1] / [C1, C3] in the text (outside code and math) into clickable citation chips. */
function linkCitations(root, card) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement.closest("code, pre, .katex") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const node of textNodes) {
    const text = node.textContent;
    const parts = [];
    let last = 0;
    for (const match of text.matchAll(CITATION)) {
      parts.push(text.slice(last, match.index));
      for (const id of match[1].split(/\s*,\s*/)) {
        const chip = el("button", { type: "button", class: "cite", text: id, title: `Show passage ${id}` });
        chip.addEventListener("click", () => showPassage(card, id));
        parts.push(chip);
      }
      last = match.index + match[0].length;
    }
    if (last === 0) continue;
    parts.push(text.slice(last));
    node.replaceWith(...parts);
  }
}

function renderAnswer(data, card) {
  const answer = el("div", { class: `answer ${data.status === "answered" ? "" : data.status === "error" ? "error" : "muted"}` });
  const html = data.status === "answered" ? markdownToHtml(data.answer) : null;
  if (html !== null) {
    answer.classList.add("markdown");
    answer.innerHTML = html;
  } else {
    answer.textContent = data.answer;
  }
  linkCitations(answer, card);
  return answer;
}

function showPassage(card, id) {
  const details = card.querySelector(".passages");
  const row = card.querySelector(`[data-id="${id}"]`);
  if (!details || !row) return;
  details.open = true;
  details.classList.remove("only-kept");
  details.querySelector("input").checked = false;
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

function gate(label, value, pass, note) {
  return el("div", { class: `gate ${pass ? "pass" : "fail"}` }, [
    el("div", { class: "gate-label" }, [el("span", { text: label }), el("span", { text: pass ? "✓ pass" : "✕ fail" })]),
    el("div", { class: "gate-value", text: value }),
    el("div", { class: "gate-note", text: note }),
  ]);
}

function passageRow(passage) {
  const text = el("p", { class: "passage-text", text: passage.text, title: "Click to expand" });
  text.addEventListener("click", () => text.classList.toggle("expanded"));

  return el("li", { class: `passage ${passage.kept ? "kept" : ""}`, "data-id": passage.id }, [
    el("div", { class: "passage-head" }, [
      el("span", { class: "passage-id", text: passage.id }),
      el("span", { class: "passage-source", text: passage.source, title: passage.source }),
      scoreBar(passage.score, passage.kept),
      el("span", { class: "passage-score", text: passage.score.toFixed(2) }),
      el("span", { class: `tag ${passage.kept ? "kept" : "dropped"}`, text: passage.kept ? "kept" : "dropped" }),
    ]),
    text,
  ]);
}

function resultCard(question, data) {
  const card = el("article", { class: "result" });

  card.append(
    el("div", { class: "result-head" }, [
      el("h3", { class: "result-question", text: question }),
      el("span", { class: `badge ${data.status}`, text: STATUS_LABELS[data.status] }),
    ]),
    renderAnswer(data, card)
  );

  if (!data.passages?.length) return card;

  const kept = data.passages.filter((p) => p.kept).length;
  card.append(
    el("div", { class: "gates" }, [
      gate("Answerable", data.answerable.toFixed(2), data.answerable >= TAU, `needs ≥ ${TAU}`),
      gate("Injection", data.injection.toFixed(2), data.injection < TAU, `must be < ${TAU}`),
      gate("Passages kept", `${kept} / ${data.passages.length}`, kept > 0, `scored ≥ ${TAU} by Jev`),
    ])
  );

  const onlyKept = el("input", { type: "checkbox" });
  const details = el("details", { class: "passages" }, [
    el("summary", {}, [el("span", { text: `Jev scores for all ${data.passages.length} passages` })]),
    el("label", { class: "passage-tools" }, [onlyKept, el("span", { text: "Show kept passages only" })]),
    el("ol", { class: "passage-list" }, data.passages.map(passageRow)),
  ]);
  onlyKept.addEventListener("change", () => details.classList.toggle("only-kept", onlyKept.checked));
  card.append(details);
  return card;
}

function thinkingCard(question) {
  const label = el("span", { text: THINKING_STEPS[0] });
  const card = el("article", { class: "result" }, [
    el("div", { class: "result-head" }, [el("h3", { class: "result-question", text: question })]),
    el("div", { class: "thinking" }, [el("span", { class: "spinner" }), label]),
  ]);
  let step = 0;
  card.timer = setInterval(() => {
    step = Math.min(step + 1, THINKING_STEPS.length - 1);
    label.textContent = THINKING_STEPS[step];
  }, 1500);
  return card;
}

async function ask(question) {
  $("welcome").hidden = true;
  askBtn.disabled = true;
  const pending = thinkingCard(question);
  results.prepend(pending);

  let data;
  try {
    data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (err) {
    data = { answer: err.message, status: "error" };
  } finally {
    clearInterval(pending.timer);
    askBtn.disabled = false;
  }
  pending.replaceWith(resultCard(question, data));
}

askForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionBox.value.trim();
  if (!question || askBtn.disabled) return;
  questionBox.value = "";
  questionBox.style.height = "";
  ask(question);
});

questionBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});

// Grow the question box with its content
questionBox.addEventListener("input", () => {
  questionBox.style.height = "auto";
  questionBox.style.height = `${questionBox.scrollHeight}px`;
});

// ---------- Start ----------

api("/api/documents")
  .then((data) => renderDocuments(data.documents))
  .catch((err) => toast(err.message, "error"));
