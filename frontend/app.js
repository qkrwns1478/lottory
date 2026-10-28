/* 외부 스크립트 없이 Streamlit 컴포넌트 통신 규약으로 Python과 연결한다. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = { args: {}, pending: null, active: "chat", evaluationURL: null };
function send(type, extra = {}) {
  parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
function resize() {
  const height = Math.ceil(
    document.querySelector(".workspace").getBoundingClientRect().height,
  );
  if (state.height !== height) {
    state.height = height;
    send("streamlit:setFrameHeight", { height });
  }
}
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function feedback(text, error = false) {
  $("feedback").textContent = text;
  $("feedback").className = text ? "visible" + (error ? " error" : "") : "";
  resize();
}
function switchPanel(name) {
  state.active = name;
  document
    .querySelectorAll(".panel")
    .forEach((p) => (p.hidden = p.id !== "panel-" + name));
  document.querySelectorAll(".nav").forEach((b) => {
    b.classList.toggle("active", b.dataset.panel === name);
    b.setAttribute("aria-selected", String(b.dataset.panel === name));
  });
  resize();
}
function event(action, extra = {}) {
  if (state.pending) return;
  const id = crypto.randomUUID
    ? crypto.randomUUID()
    : String(Date.now()) + Math.random();
  state.pending = id;
  document
    .querySelectorAll("button[type=submit],#run-evaluation")
    .forEach((b) => (b.disabled = true));
  feedback(
    action === "evaluate"
      ? "20문항을 비교 평가하고 있습니다. 완료될 때까지 기다려주세요."
      : "요청을 처리하고 있습니다…",
  );
  send("streamlit:setComponentValue", {
    dataType: "json",
    value: {
      id,
      action,
      selected_round: Number($("round-select").value),
      ai: $("ai-mode").checked,
      search_type: $("search-type").value,
      fetch_k: Number($("fetch-k").value),
      lambda_mult: Number($("lambda").value),
      ...extra,
    },
  });
}
function safeLink(url) {
  try {
    const value = new URL(url);
    return value.protocol === "https:" &&
      [
        "cashlogy.kr",
        "www.law.go.kr",
        "law.go.kr",
        "d.nts.go.kr",
        "www.nts.go.kr",
        "www.dhlottery.co.kr",
      ].includes(value.hostname)
      ? value.href
      : null;
  } catch {
    return null;
  }
}
function sourceName(source) {
  return (
    {
      statute: "법령",
      government_guidance: "국세청 안내",
      structured_summary: "작성 설명",
      round_csv: "회차 CSV",
    }[source.source_type || source.document_type] ||
    (source.document_type === "law_article" ? "법령" : "회차 CSV")
  );
}
function sources(container, values) {
  if (!values?.length) {
    container.append(el("p", "인용 가능한 근거 문서가 없습니다.", "muted"));
    return;
  }
  for (const source of values) {
    const box = el("div", null, "source");
    box.append(el("span", sourceName(source), "source-type"));
    const url = safeLink(source.source_url);
    const link = el(url ? "a" : "span", source.document_title);
    if (url) {
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
    box.append(link);
    box.append(
      el(
        "small",
        [
          source.source_name,
          source.effective_date ? "시행 " + source.effective_date : "",
          source.collected_at ? "수집 " + source.collected_at.slice(0, 10) : "",
        ]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    container.append(box);
  }
}
function table(container, headers, records) {
  const wrap = el("div", null, "table-wrap"),
    t = el("table"),
    head = el("thead"),
    tr = el("tr");
  headers.forEach((h) => tr.append(el("th", h)));
  head.append(tr);
  t.append(head);
  const body = el("tbody");
  records.forEach((row) => {
    const r = el("tr");
    row.forEach((value) => r.append(el("td", value)));
    body.append(r);
  });
  t.append(body);
  wrap.append(t);
  container.append(wrap);
}
function rows(container, values) {
  table(
    container,
    [
      "회차",
      "추첨일",
      "당첨번호",
      "보너스",
      "1게임당 1등 당첨금",
      "당첨 게임 수",
    ],
    values.map((row) => [
      row.round,
      row.draw_date,
      row.numbers.join(", "),
      row.bonus_number,
      Number(row.first_prize_amount).toLocaleString("ko-KR") + "원",
      row.first_prize_winning_games,
    ]),
  );
}
function renderMessages(messages) {
  const signature = JSON.stringify(messages || []);
  if (state.messagesSignature === signature) return;
  state.messagesSignature = signature;
  const box = $("chat-log");
  const openDetails = new Set(
    [...box.querySelectorAll("details[open]")].map(
      (detail) => detail.dataset.disclosureKey,
    ),
  );
  box.replaceChildren();
  if (!messages?.length) {
    box.append(
      el(
        "div",
        "회차를 선택하고 당첨금, 번호, 수령기한 등을 질문하세요.",
        "empty",
      ),
    );
    return;
  }
  for (const [index, item] of messages.entries()) {
    box.append(el("div", item.question, "chat-message user"));
    const answer = el("article", null, "chat-message"),
      r = item.response;
    answer.append(el("div", r.answer, "answer-text"));
    if (r.rows?.length && r.plan?.intent !== "round_lookup")
      rows(answer, r.rows);
    sources(answer, r.sources);
    for (const note of r.limitations || [])
      answer.append(el("p", note, "limitation"));
    if (r.calculation) {
      const details = el("details"),
        summary = el("summary", "계산 조건과 결과");
      details.dataset.disclosureKey = `${index}:${item.question}`;
      details.open = openDetails.has(details.dataset.disclosureKey);
      details.append(
        summary,
        el("pre", JSON.stringify(r.calculation, null, 2)),
      );
      answer.append(details);
    }
    box.append(answer);
  }
  box.scrollTop = box.scrollHeight;
}
function renderComparison(report) {
  const box = $("compare-results");
  box.replaceChildren(el("p", report.note));
  for (const [collection, pair] of Object.entries(report.collections)) {
    box.append(
      el("h3", collection === "lotto_rounds" ? "회차 기록" : "문서 근거"),
    );
    const grid = el("div", null, "comparison-grid");
    for (const method of ["similarity", "mmr"]) {
      const data = pair[method],
        card = el("article", null, "comparison-card");
      card.append(
        el("h3", method.toUpperCase()),
        el(
          "p",
          `다양성 ${Number(data.diversity).toFixed(3)} · 출처 유형 ${data.source_type_count || 0}종`,
        ),
      );
      for (const result of data.results || []) {
        card.append(
          el("p", `${result.rank}. ${result.metadata.document_title}`),
        );
        card.append(
          el("small", `${sourceName(result.metadata)} · ${result.chunk_id}`),
        );
      }
      grid.append(card);
    }
    box.append(grid);
  }
}
function renderEvaluation(report) {
  if (!report) return;
  const signature = JSON.stringify(report);
  if (state.evaluationSignature === signature) return;
  state.evaluationSignature = signature;
  const box = $("evaluation-results");
  const openDetails = new Set(
    [...box.querySelectorAll("details[open]")].map(
      (detail) => detail.dataset.disclosureKey,
    ),
  );
  box.replaceChildren();
  box.append(el("p", report.methodology));
  const grid = el("div", null, "comparison-grid");
  for (const method of ["baseline", "rag"]) {
    const m = report.summary[method],
      card = el("article", null, "comparison-card");
    card.append(
      el(
        "h3",
        method === "baseline" ? "RAG 이전 · 모델만" : "RAG 이후 · 근거와 계산",
      ),
    );
    const metrics = el("div", null, "metrics");
    for (const [label, value] of [
      ["정확성", (m.accuracy * 100).toFixed(1) + "%"],
      ["인용률", (m.citation_rate * 100).toFixed(1) + "%"],
      ["응답시간 중앙값", m.median_seconds.toFixed(2) + "초"],
    ]) {
      const metric = el("div", null, "metric");
      metric.append(el("strong", value), el("small", label));
      metrics.append(metric);
    }
    card.append(
      metrics,
      el(
        "p",
        `근거 부족 처리: ${m.abstention_accuracy == null ? "대상 없음" : (m.abstention_accuracy * 100).toFixed(0) + "%"} · 실행 오류 ${m.errors}건`,
      ),
    );
    grid.append(card);
  }
  box.append(grid);
  table(
    box,
    ["질문", "모델만", "RAG", "모델 시간", "RAG 시간"],
    report.results.map((r) => [
      r.case.question,
      r.baseline.correct ? "정답" : "오답/보류",
      r.rag.correct ? "정답" : "오답/보류",
      r.baseline.elapsed_seconds.toFixed(2) + "초",
      r.rag.elapsed_seconds.toFixed(2) + "초",
    ]),
  );
  for (const result of report.results) {
    const detail = el("details");
    detail.dataset.disclosureKey = result.case.id || result.case.question;
    detail.open = openDetails.has(detail.dataset.disclosureKey);
    detail.append(
      el("summary", result.case.question),
      el("h4", "RAG 이전"),
      el("div", result.baseline.answer, "answer-text"),
      el("h4", "RAG 이후"),
      el("div", result.rag.answer, "answer-text"),
    );
    sources(detail, result.rag.sources);
    box.append(detail);
  }
  if (state.evaluationURL) URL.revokeObjectURL(state.evaluationURL);
  state.evaluationURL = URL.createObjectURL(
    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
  );
  $("download-evaluation").href = state.evaluationURL;
  $("download-evaluation").hidden = false;
}
function renderRecommendation(report) {
  if (!report) return;
  const signature = JSON.stringify(report);
  if (state.recommendationSignature === signature) return;
  state.recommendationSignature = signature;
  const box = $("recommendation-results");
  box.replaceChildren(
    el("h3", `${report.target_date} 추첨일 기준 추천`),
    el(
      "p",
      `${report.from_round}~${report.to_round}회, 총 ${report.row_count}회 분석 (${report.from_date}~${report.to_date})`,
    ),
  );
  if (report.row_count < report.requested_window) {
    box.append(
      el(
        "p",
        `요청한 ${report.requested_window}회 중 데이터가 있는 ${report.row_count}회만 분석했습니다.`,
        "limitation",
      ),
    );
  }
  if (report.warning) box.append(el("p", report.warning, "limitation"));
  report.games.forEach((game, index) => {
    const card = el("article", null, "recommendation-game");
    card.append(el("h3", `추천 ${index + 1}`));
    const balls = el("div", null, "lotto-balls");
    game.numbers.forEach((number) => {
      balls.append(
        el(
          "span",
          number,
          `lotto-ball ball-${Math.min(4, Math.floor((number - 1) / 10))}`,
        ),
      );
    });
    card.append(balls);
    box.append(card);
  });
  const method = el("details");
  method.append(el("summary", "추천 기준 보기"), el("p", report.method));
  box.append(method);
  const details = el("details");
  details.append(el("summary", "번호별 출현 횟수 전체 보기 (보너스 제외)"));
  table(
    details,
    [
      "번호",
      `분석 ${report.row_count}회 출현 횟수`,
      `최근 ${report.recent_count}회 출현 횟수`,
    ],
    report.frequencies.map((item) => [
      item.number,
      item.count,
      item.recent_count,
    ]),
  );
  box.append(details);
}
function render(args) {
  state.args = args;
  const select = $("round-select"),
    selected =
      state.lastServerRound !== args.selected_round
        ? String(args.selected_round)
        : select.value || String(args.selected_round);
  state.lastServerRound = args.selected_round;
  select.replaceChildren(
    ...(args.rounds || []).map((n) => {
      const option = el("option", n + "회");
      option.value = n;
      return option;
    }),
  );
  select.value = selected;
  if (!select.value) select.value = args.selected_round;
  renderMessages(args.messages);
  renderEvaluation(args.evaluation);
  renderRecommendation(args.recommendation);
  const result = args.result;
  if (result) {
    if (result.kind === "filter") {
      const box = $("filter-results");
      box.replaceChildren(el("h3", `조건에 맞는 회차 ${result.rows.length}개`));
      rows(box, result.rows);
    }
    if (result.kind === "comparison") renderComparison(result.comparison);
    if (result.kind === "evaluation") renderEvaluation(result.evaluation);
    if (result.kind === "recommendation")
      renderRecommendation(result.recommendation);
    if (result.kind === "error") feedback(result.message, true);
    if (state.pending === result.id) {
      state.pending = null;
      document
        .querySelectorAll("button[type=submit],#run-evaluation")
        .forEach((b) => (b.disabled = false));
      if (result.kind !== "error") feedback("처리 완료");
    }
  }
  resize();
}
document
  .querySelectorAll(".nav")
  .forEach((button) =>
    button.addEventListener("click", () => switchPanel(button.dataset.panel)),
  );
document.querySelectorAll("[data-question]").forEach((button) =>
  button.addEventListener("click", () => {
    $("question").value = button.dataset.question;
    $("question").focus();
  }),
);
$("lambda").addEventListener("input", () => {
  $("lambda-value").value = $("lambda").value;
});
$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  event("ask", { question: $("question").value });
});
$("recommendation-form").addEventListener("submit", (e) => {
  e.preventDefault();
  event("recommend", {
    recommendation_window: Number($("recommendation-window").value),
    game_count: Number($("game-count").value),
  });
});
$("filter-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const number = (id) => ($(id).value === "" ? null : Number($(id).value));
  const numbers = $("numbers").value.trim()
    ? $("numbers")
        .value.split(",")
        .map((x) => Number(x.trim()))
    : [];
  if (numbers.some((n) => !Number.isInteger(n) || n < 1 || n > 45)) {
    feedback("번호는 1~45 사이의 정수로 입력하세요.", true);
    return;
  }
  event("filter", {
    filters: {
      start_date: $("start-date").value || null,
      end_date: $("end-date").value || null,
      numbers,
      min_prize: number("min-prize"),
      max_prize: number("max-prize"),
      odd_count: number("odd-count"),
      min_consecutive_pairs: number("pairs"),
      min_sum: number("min-sum"),
      max_sum: number("max-sum"),
    },
  });
});
$("compare-form").addEventListener("submit", (e) => {
  e.preventDefault();
  event("compare", { question: $("compare-question").value });
});
$("clear-chat").addEventListener("click", () => event("clear"));
$("round-select").addEventListener("change", () => event("clear"));
$("run-evaluation").addEventListener("click", () => event("evaluate"));
window.addEventListener("message", (e) => {
  if (e.source === parent && e.data?.type === "streamlit:render")
    render(e.data.args);
});
window.addEventListener("resize", resize);
// 접거나 펼친 뒤의 실제 콘텐츠 높이를 부모 화면에 전달한다.
document.addEventListener("toggle", () => requestAnimationFrame(resize), true);
new ResizeObserver(resize).observe(document.body);
send("streamlit:componentReady", { apiVersion: 1 });
resize();
