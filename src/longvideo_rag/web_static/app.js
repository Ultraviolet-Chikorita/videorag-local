const state = {
  sources: [],
  selected: null,
  clipEndSeconds: null,
};

const elements = {
  chatForm: document.getElementById("chatForm"),
  question: document.getElementById("question"),
  sendButton: document.getElementById("sendButton"),
  messages: document.getElementById("messages"),
  sources: document.getElementById("sources"),
  sourceCount: document.getElementById("sourceCount"),
  clipPlayer: document.getElementById("clipPlayer"),
  clipMonitor: document.getElementById("clipMonitor"),
  monitorPlaceholder: document.getElementById("monitorPlaceholder"),
  monitorActive: document.getElementById("monitorActive"),
  playingLabel: document.getElementById("playingLabel"),
  playingTitle: document.getElementById("playingTitle"),
  playingTimestamp: document.getElementById("playingTimestamp"),
  playingTranscript: document.getElementById("playingTranscript"),
  playClip: document.getElementById("playClip"),
  statusLight: document.getElementById("statusLight"),
  statusText: document.getElementById("statusText"),
  statsText: document.getElementById("statsText"),
};

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (!response.ok || !status.ready) {
      throw new Error("Index not ready");
    }
    elements.statusLight.classList.add("online");
    elements.statusText.textContent = `ONLINE // ${status.model}`;
    elements.statsText.textContent = `${status.stats.videos} videos | ${status.stats.segments} clips | ${status.stats.visual_segments || 0} visual`;
  } catch (error) {
    elements.statusLight.classList.add("error");
    elements.statusText.textContent = "OFFLINE";
    elements.statsText.textContent = "Unable to read local index";
  }
}

function addMessage(role, answer, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "speaker";
  label.textContent = role === "user" ? "YOU" : "ASSISTANT";
  article.appendChild(label);
  if (role === "assistant") {
    renderAnswer(article, answer, sources);
  } else {
    const paragraph = document.createElement("p");
    paragraph.textContent = answer;
    article.appendChild(paragraph);
  }
  elements.messages.appendChild(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
}

function addWaitingMessage() {
  const article = addMessage("assistant", "Reviewing local transcript and visual evidence...");
  article.classList.add("waiting");
  return article;
}

function renderAnswer(container, answer, sources) {
  const lines = answer.split(/\r?\n/).filter((line) => line.trim());
  for (const rawLine of lines) {
    const paragraph = document.createElement("p");
    const line = rawLine.replace(/^-\s*/, "");
    if (/^-\s*/.test(rawLine)) {
      paragraph.className = "bullet";
    }
    const citationPattern = /\[S\d+(?:\s*,\s*S\d+)*\]/g;
    let cursor = 0;
    for (const match of line.matchAll(citationPattern)) {
      paragraph.append(document.createTextNode(line.slice(cursor, match.index)));
      const labels = match[0].slice(1, -1).split(",").map((item) => item.trim());
      labels.forEach((sourceLabel, index) => {
        const source = sources.find((item) => item.label === sourceLabel);
        if (source) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "citation-button";
          button.textContent = `[${sourceLabel}]`;
          button.addEventListener("click", () => selectSource(sourceLabel));
          paragraph.appendChild(button);
        } else {
          paragraph.append(document.createTextNode(`[${sourceLabel}]`));
        }
        if (index < labels.length - 1) {
          paragraph.append(document.createTextNode(" "));
        }
      });
      cursor = match.index + match[0].length;
    }
    paragraph.append(document.createTextNode(line.slice(cursor)));
    container.appendChild(paragraph);
  }
}

function renderSources(sources) {
  state.sources = sources;
  elements.sourceCount.textContent = String(sources.length).padStart(2, "0");
  elements.sources.replaceChildren();
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "empty-sources";
    empty.textContent = "No evidence clips matched this question.";
    elements.sources.appendChild(empty);
    clearMonitor();
    return;
  }
  for (const source of sources) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "source-card";
    card.dataset.label = source.label;
    const badge = document.createElement("span");
    badge.className = "source-label";
    badge.textContent = source.label;
    const details = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = source.title;
    const time = document.createElement("time");
    time.textContent = formatRange(source.start_ms, source.end_ms);
    const excerpt = document.createElement("p");
    excerpt.textContent = source.visual_caption || source.transcript;
    details.append(title, time, excerpt);
    card.append(badge, details);
    card.addEventListener("click", () => selectSource(source.label));
    elements.sources.appendChild(card);
  }
  selectSource(sources[0].label);
}

function selectSource(label) {
  const source = state.sources.find((item) => item.label === label);
  if (!source) {
    return;
  }
  state.selected = source;
  state.clipEndSeconds = source.end_ms / 1000;
  elements.monitorPlaceholder.classList.add("hidden");
  elements.monitorActive.classList.remove("hidden");
  elements.playingLabel.textContent = source.label;
  elements.playingTitle.textContent = source.title;
  elements.playingTimestamp.textContent = formatRange(source.start_ms, source.end_ms);
  const evidence = [];
  if (source.visual_caption) {
    evidence.push(`Visual: ${source.visual_caption}`);
  }
  if (source.transcript) {
    evidence.push(`Transcript: ${source.transcript}`);
  }
  elements.playingTranscript.textContent = evidence.join("\n\n");
  const startSeconds = source.start_ms / 1000;
  const endSeconds = source.end_ms / 1000;
  elements.clipPlayer.src = `${source.media_url}#t=${startSeconds},${endSeconds}`;
  elements.clipPlayer.load();
  document.querySelectorAll(".source-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.label === label);
  });
}

function clearMonitor() {
  state.selected = null;
  state.clipEndSeconds = null;
  elements.clipPlayer.removeAttribute("src");
  elements.clipPlayer.load();
  elements.monitorActive.classList.add("hidden");
  elements.monitorPlaceholder.classList.remove("hidden");
}

function formatRange(startMs, endMs) {
  return `${formatTime(startMs)} - ${formatTime(endMs)}`;
}

function formatTime(milliseconds) {
  const seconds = Math.floor(milliseconds / 1000);
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const remainder = String(seconds % 60).padStart(2, "0");
  return `${hours}:${minutes}:${remainder}`;
}

async function submitQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed) {
    return;
  }
  elements.question.value = "";
  addMessage("user", trimmed);
  const waiting = addWaitingMessage();
  elements.sendButton.disabled = true;
  elements.sendButton.textContent = "SCANNING";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: trimmed,
        mode: "answer",
        top_k: 8,
      }),
    });
    const payload = await response.json();
    waiting.remove();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to answer locally.");
    }
    addMessage("assistant", payload.answer, payload.sources);
    renderSources(payload.sources);
  } catch (error) {
    waiting.remove();
    addMessage("assistant", `Local query failed: ${error.message}`);
  } finally {
    elements.sendButton.disabled = false;
    elements.sendButton.textContent = "SEND";
    elements.question.focus();
  }
}

elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(elements.question.value);
});

elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

elements.playClip.addEventListener("click", async () => {
  if (!state.selected) {
    return;
  }
  elements.clipPlayer.currentTime = state.selected.start_ms / 1000;
  await elements.clipPlayer.play();
});

elements.clipPlayer.addEventListener("timeupdate", () => {
  if (state.clipEndSeconds && elements.clipPlayer.currentTime >= state.clipEndSeconds) {
    elements.clipPlayer.pause();
  }
});

loadStatus();
