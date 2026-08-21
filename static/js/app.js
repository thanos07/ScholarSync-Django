(() => {
  const STORAGE_KEY = "scholarsync:library-collapsed";

  function setSidebarState(shell, collapsed) {
    const toggle = shell.querySelector("[data-sidebar-toggle]");
    const label = shell.querySelector("[data-sidebar-toggle-label]");
    const sidebar = shell.querySelector("[data-research-sidebar]");

    shell.classList.toggle("sidebar-collapsed", collapsed);
    shell.classList.toggle("sidebar-open", !collapsed);
    if (toggle) toggle.setAttribute("aria-expanded", String(!collapsed));
    if (sidebar) sidebar.setAttribute("aria-hidden", String(collapsed));
    if (label) {
      const isDemo = document.body.classList.contains("research-page") && shell.classList.contains("demo-shell");
      label.textContent = collapsed
        ? (isDemo ? "Show library" : "Show documents")
        : (isDemo ? "Hide library" : "Hide documents");
    }
  }

  document.querySelectorAll("[data-research-shell]").forEach((shell) => {
    const toggle = shell.querySelector("[data-sidebar-toggle]");
    if (!toggle) return;

    let collapsed = false;
    try {
      collapsed = localStorage.getItem(STORAGE_KEY) === "true";
    } catch (_) {
      collapsed = false;
    }
    if (window.matchMedia("(max-width: 720px)").matches) collapsed = true;
    setSidebarState(shell, collapsed);

    toggle.addEventListener("click", () => {
      const nextCollapsed = !shell.classList.contains("sidebar-collapsed");
      setSidebarState(shell, nextCollapsed);
      try {
        localStorage.setItem(STORAGE_KEY, String(nextCollapsed));
      } catch (_) {
        // Storage can be unavailable in privacy-restricted browsers.
      }
    });
  });

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

function inlineMarkdown(value) {
  const protectedValues = [];

  const protect = (raw, html) => {
    const token = `SCHOLARSYNCPROTECTED${protectedValues.length}TOKEN`;
    protectedValues.push({ token, html });
    return token;
  };

  let source = String(value ?? "");

  // Protect TeX before applying Markdown emphasis.
  // Display delimiters can appear inside a table cell;
  // convert them to inline delimiters there so KaTeX
  // can render the expression without breaking the table layout.
  source = source.replace(/\$\$([\s\S]+?)\$\$/g, (_, formula) => {
    const wrapped = `\\(${formula.trim()}\\)`;
    return protect(
      wrapped,
      `<span class="math-inline">${escapeHtml(wrapped)}</span>`
    );
  });

  source = source.replace(/\\\[([\s\S]+?)\\\]/g, (_, formula) => {
    const wrapped = `\\(${formula.trim()}\\)`;
    return protect(
      wrapped,
      `<span class="math-inline">${escapeHtml(wrapped)}</span>`
    );
  });

  source = source.replace(/\\\((.+?)\\\)/g, (match) =>
    protect(
      match,
      `<span class="math-inline">${escapeHtml(match)}</span>`
    )
  );

  source = source.replace(
    /(^|[^$])\$([^$\n]+?)\$(?!\$)/g,
    (match, prefix, formula) =>
      `${prefix}${protect(
        `$${formula}$`,
        `<span class="math-inline">${escapeHtml(`$${formula}$`)}</span>`
      )}`
  );

  source = source.replace(/`([^`]+)`/g, (_, code) =>
    protect(
      code,
      `<code>${escapeHtml(code)}</code>`
    )
  );

  let text = escapeHtml(source);

  // ---------------------------------------------------------
  // Markdown emphasis
  // ---------------------------------------------------------

  // Normal **bold**
  text = text.replace(
    /\*\*(.+?)\*\*/g,
    "<strong>$1</strong>"
  );

  /*
    IMPORTANT FIX:

    Treat __bold__ as Markdown only when the underscores
    behave like real Markdown delimiters.

    This prevents filenames / identifiers such as:

      44_Songkhla, Thailand_GIS_AHP_2019
      PV_site_suitability_2026

    from losing underscores.
  */
  text = text.replace(
    /(^|[\s([{>])__([^_\n]+?)__(?=$|[\s.,!?;:)\]}>])/g,
    "$1<strong>$2</strong>"
  );

  // Normal *italic*
  text = text.replace(
    /(^|[^*])\*([^*\n]+?)\*(?!\*)/g,
    "$1<em>$2</em>"
  );

  /*
    IMPORTANT FIX:

    Treat _italic_ as Markdown only when the opening
    underscore occurs at a natural text boundary.

    So this still works:

      _important text_

    but this remains untouched:

      44_Songkhla, Thailand_GIS_AHP_2019
  */
  text = text.replace(
    /(^|[\s([{>])_([^_\n]+?)_(?=$|[\s.,!?;:)\]}>])/g,
    "$1<em>$2</em>"
  );

  // Restore protected TeX / code.
  protectedValues.forEach(({ token, html }) => {
    text = text.replaceAll(token, html);
  });

  return text;
}

  function renderMarkdown(value) {
    const lines = String(value ?? "").replace(/\r\n?/g, "\n").split("\n");
    const html = [];
    let listType = null;

    const closeList = () => {
      if (listType) {
        html.push(`</${listType}>`);
        listType = null;
      }
    };

    const tableCells = (line) => line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());

    const isTableDivider = (line) => {
      const cells = tableCells(line);
      return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
    };

    for (let index = 0; index < lines.length; index += 1) {
      const rawLine = lines[index];
      const line = rawLine.trim();
      if (!line) {
        closeList();
        continue;
      }

      // Display TeX: $$ ... $$ or \[ ... \]. Keep the delimiters so KaTeX's
      // auto-render extension can typeset the expression after Markdown.
      if (line.startsWith("$$") || line.startsWith("\\[")) {
        closeList();
        const dollar = line.startsWith("$$");
        const opening = dollar ? "$$" : "\\[";
        const closing = dollar ? "$$" : "\\]";
        let formula = line.slice(opening.length);
        let closed = formula.endsWith(closing) && formula.length > closing.length;
        if (closed) formula = formula.slice(0, -closing.length);
        while (!closed && index + 1 < lines.length) {
          index += 1;
          const next = lines[index];
          if (next.trim().endsWith(closing)) {
            formula += `\n${next.trim().slice(0, -closing.length)}`;
            closed = true;
          } else {
            formula += `\n${next}`;
          }
        }
        const wrapped = dollar ? `$$${formula.trim()}$$` : `\\[${formula.trim()}\\]`;
        html.push(`<div class="math-block">${escapeHtml(wrapped)}</div>`);
        continue;
      }

      const nextLine = lines[index + 1]?.trim() || "";
      if (line.includes("|") && isTableDivider(nextLine)) {
        closeList();
        const headers = tableCells(line);
        const rows = [];
        index += 2;
        while (index < lines.length) {
          const rowLine = lines[index].trim();
          if (!rowLine || !rowLine.includes("|")) break;
          rows.push(tableCells(rowLine));
          index += 1;
        }
        index -= 1;

        const normalizedHeaders = headers.map((cell) => cell.toLowerCase());
        const isFormulaTable = normalizedHeaders.includes("formula") && normalizedHeaders.includes("symbols");
        const tableClass = isFormulaTable ? "message-table formula-table" : "message-table";
        html.push(`<div class="message-table-wrap${isFormulaTable ? " formula-table-wrap" : ""}"><table class="${tableClass}">`);
        if (isFormulaTable && headers.length === 4) {
          html.push('<colgroup><col class="formula-index-col"><col class="formula-expression-col"><col class="formula-symbols-col"><col class="formula-source-col"></colgroup>');
        }
        html.push('<thead><tr>');
        headers.forEach((cell) => html.push(`<th>${inlineMarkdown(cell)}</th>`));
        html.push("</tr></thead><tbody>");
        rows.forEach((row) => {
          html.push("<tr>");
          headers.forEach((_, cellIndex) => {
            html.push(`<td>${inlineMarkdown(row[cellIndex] || "")}</td>`);
          });
          html.push("</tr>");
        });
        html.push("</tbody></table></div>");
        continue;
      }

      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(5, heading[1].length + 2);
        html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }

      const numbered = line.match(/^\d+[.)]\s+(.+)$/);
      if (numbered) {
        if (listType !== "ol") {
          closeList();
          listType = "ol";
          html.push("<ol>");
        }
        html.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
        continue;
      }

      const bullet = line.match(/^(?:[-*•])\s+(.+)$/);
      if (bullet) {
        if (listType !== "ul") {
          closeList();
          listType = "ul";
          html.push("<ul>");
        }
        html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
        continue;
      }

      const quote = line.match(/^>\s*(.+)$/);
      if (quote) {
        closeList();
        html.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
        continue;
      }

      closeList();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
    }

    closeList();
    return html.join("");
  }

  function typesetMath(node, retry = true) {
    if (!node) return;
    if (typeof window.renderMathInElement !== "function") {
      if (retry) setTimeout(() => typesetMath(node, false), 250);
      return;
    }
    try {
      window.renderMathInElement(node, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        strict: "ignore",
      });
    } catch (error) {
      console.warn("ScholarSync could not typeset a formula:", error);
    }
  }

  document.querySelectorAll("[data-markdown-content]").forEach((node) => {
    node.innerHTML = renderMarkdown(node.textContent);
    typesetMath(node);
  });

  const messageStream = document.querySelector("[data-message-stream]");
  const scrollToLatest = (smooth = false) => {
    if (!messageStream) return;
    messageStream.scrollTo({
      top: messageStream.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  };

  if (messageStream && messageStream.querySelector(".chat-message")) {
    requestAnimationFrame(() => scrollToLatest(false));
    setTimeout(() => scrollToLatest(false), 80);
  }

  const questionInput = document.querySelector("[data-question-input]");
  document.querySelectorAll("[data-suggested-question]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!questionInput) return;
      questionInput.value = button.dataset.suggestedQuestion || button.textContent.trim();
      questionInput.dispatchEvent(new Event("input", { bubbles: true }));
      questionInput.focus();
      questionInput.setSelectionRange(questionInput.value.length, questionInput.value.length);
    });
  });

  document.querySelectorAll("textarea").forEach((textarea) => {
    const resize = () => {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    };
    textarea.addEventListener("input", resize);
    resize();
  });

  function addCopyControls(article, content, hidden = false) {
    article._copyText = String(content ?? "");

    const raw = document.createElement("span");
    raw.hidden = true;
    raw.dataset.messageRaw = "true";
    raw.textContent = article._copyText;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "message-copy-button";
    button.dataset.copyMessage = "true";
    button.setAttribute("aria-label", "Copy message");
    button.textContent = "Copy";
    button.hidden = hidden;

    article.append(button, raw);
    return { button, raw };
  }

  function createMessage(role, content, pending = false) {
    const article = document.createElement("article");
    article.className = `chat-message ${role === "USER" ? "user-message" : "assistant-message"}`;
    if (pending) article.dataset.pendingMessage = "true";

    const roleLabel = document.createElement("span");
    roleLabel.className = "message-role";
    roleLabel.textContent = role === "USER" ? "You" : "ScholarSync";
    article.appendChild(roleLabel);

    addCopyControls(article, content, pending);

    const body = document.createElement("div");
    body.className = "message-content";
    if (pending) {
      body.innerHTML = '<span class="generation-status"><span class="status-dot"></span><span>Retrieving evidence and generating an answer…</span></span>';
    } else if (role === "ASSISTANT") {
      body.innerHTML = renderMarkdown(content);
      typesetMath(body);
    } else {
      body.textContent = content;
    }

    article.appendChild(body);
    return article;
  }

  async function copyText(value) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-copy-message]");
    if (!button) return;
    const article = button.closest(".chat-message");
    if (!article) return;
    const raw = article.querySelector("[data-message-raw]");
    const text = article._copyText ?? raw?.textContent ?? article.querySelector(".message-content")?.innerText ?? "";
    if (!text.trim()) return;

    const original = button.textContent;
    try {
      await copyText(text);
      button.textContent = "Copied";
      button.classList.add("is-copied");
    } catch (_) {
      button.textContent = "Copy failed";
    }
    setTimeout(() => {
      button.textContent = original;
      button.classList.remove("is-copied");
    }, 1500);
  });

  function appendCitations(article, citations) {
    if (!citations?.length) return;
    const list = document.createElement("div");
    list.className = "citation-list";

    const heading = document.createElement("b");
    heading.textContent = "Evidence";
    list.appendChild(heading);

    citations.forEach((citation) => {
      const link = document.createElement("a");
      link.href = citation.paper_url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = `[${citation.number}] ${citation.source} · p. ${citation.page}`;
      list.appendChild(link);

      const verification = document.createElement("span");
      verification.className = `citation-verification citation-verification-${String(citation.verification_status || "unchecked").toLowerCase()}`;
      verification.textContent = citation.verification_label || "Not checked";
      list.appendChild(verification);
    });
    article.appendChild(list);
  }

  function updateEvidencePanel(citations) {
    const container = document.querySelector("[data-evidence-list]");
    if (!container) return;
    container.innerHTML = "";

    if (!citations?.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No supporting passage was attached to this response.";
      container.appendChild(empty);
      return;
    }

    citations.forEach((citation) => {
      const card = document.createElement("article");
      card.className = "evidence-card";
      card.innerHTML = `
        <div><span>[${citation.number}]</span><b>Page ${citation.page}</b></div>
        <p>${escapeHtml(citation.excerpt)}</p>
        <a href="${escapeHtml(citation.paper_url)}" target="_blank" rel="noopener">Open page ${citation.page} →</a>
        <small>${escapeHtml(citation.source)} · retrieval score ${escapeHtml(citation.score)}</small>
        <span class="citation-verification citation-verification-${escapeHtml(String(citation.verification_status || "unchecked").toLowerCase())}">
          ${escapeHtml(citation.verification_label || "Not checked")}
        </span>
      `;
      container.appendChild(card);
    });
  }

  const chatForm = document.querySelector("[data-chat-form]");
  const submitButton = document.querySelector("[data-submit-button]");
  let requestInFlight = false;

  if (questionInput && chatForm) {
    questionInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.isComposing) return;
      if (event.shiftKey) return;
      event.preventDefault();
      if (!requestInFlight && questionInput.value.trim()) chatForm.requestSubmit();
    });

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (requestInFlight) return;

      const question = questionInput.value.trim();
      if (!question) return;
      requestInFlight = true;

      document.querySelector("[data-chat-empty]")?.remove();
      const userMessage = createMessage("USER", question);
      const pendingMessage = createMessage("ASSISTANT", "", true);
      messageStream?.append(userMessage, pendingMessage);

      questionInput.value = "";
      questionInput.style.height = "auto";
      questionInput.disabled = true;
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = "Generating…";
      }
      scrollToLatest(true);

      try {
        const formData = new FormData(chatForm);
        formData.set("question", question);
        const response = await fetch(chatForm.action || window.location.href, {
          method: "POST",
          body: formData,
          credentials: "same-origin",
          headers: { "X-Requested-With": "XMLHttpRequest" },
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Request failed.");

        pendingMessage.removeAttribute("data-pending-message");
        pendingMessage._copyText = data.answer;
        const raw = pendingMessage.querySelector("[data-message-raw]");
        if (raw) raw.textContent = data.answer;
        const copyButton = pendingMessage.querySelector("[data-copy-message]");
        if (copyButton) copyButton.hidden = false;

        const body = pendingMessage.querySelector(".message-content");
        body.innerHTML = renderMarkdown(data.answer);
        typesetMath(body);
        appendCitations(pendingMessage, data.citations);
        updateEvidencePanel(data.citations);
        document.querySelector("[data-export-link]")?.removeAttribute("hidden");
        scrollToLatest(true);
      } catch (error) {
        const body = pendingMessage.querySelector(".message-content");
        body.innerHTML = `<p>${escapeHtml(error.message || "ScholarSync could not complete this request.")}</p>`;
        pendingMessage.classList.add("message-error");
        const copyButton = pendingMessage.querySelector("[data-copy-message]");
        if (copyButton) copyButton.hidden = true;
        scrollToLatest(true);
      } finally {
        requestInFlight = false;
        questionInput.disabled = false;
        questionInput.focus();
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.textContent = "Ask ScholarSync";
        }
      }
    });
  }
})();
