// ListSmart frontend. Plain JavaScript, no build step.
// Flow: pick photos -> POST /api/generate -> show listing, price and competitors.
"use strict";

(function () {
  // ---------- Settings ----------
  const MAX_FILES = 5;
  const MAX_BYTES = 8 * 1024 * 1024; // 8 MB per photo (same limit as the backend)
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const TITLE_LIMIT = { meesho: 100, etsy: 140 };
  const TAG_MAX_LEN = 20;
  const TAG_MAX_COUNT = 13;

  // Demo mode: add ?demo=1 to the URL, or open the file straight from disk.
  const params = new URLSearchParams(window.location.search);
  const DEMO = params.get("demo") === "1" || window.location.protocol === "file:";

  // Indian rupee formatting, e.g. ₹1,24,999
  const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
  const formatINR = (n) => (typeof n === "number" && isFinite(n) ? inr.format(n) : "–");

  // Small helper to grab elements by id
  const $ = (id) => document.getElementById(id);

  // ---------- App state ----------
  let files = []; // photos the seller picked: [{ file, url }]
  let tags = []; // current tags (editable)
  let lastResult = null; // last response from the backend

  // ---------- Built-in sample data for demo mode ----------
  const MOCK_HEALTH = { ok: true, writer_backend: "mock", gemma: false, serpapi: false, tinker: false };

  const MOCK_RESULT = {
    attributes: {
      product_type: "kurti", category: "Women Ethnic Wear", colors: ["maroon", "gold"],
      material: "cotton", pattern: "block print", style: "straight", gender: "women",
      key_features: ["3/4 sleeves", "mandarin collar", "side slits"], visible_text: "", condition: "new"
    },
    listing: {
      title: "Women's Maroon Cotton Block Print Straight Kurti with Mandarin Collar & 3/4 Sleeves",
      description:
        "Add a touch of tradition to your everyday look with this maroon straight kurti, made from soft, breathable cotton.\n\n" +
        "The hand block print in gold tones and the neat mandarin collar make it perfect for office, college or festive days.\n\n" +
        "Pair it with leggings, palazzos or jeans for a comfortable, stylish outfit.",
      bullets: [
        "Fabric: 100% breathable cotton, soft on skin",
        "Pattern: traditional hand block print in maroon and gold",
        "Fit: straight cut with side slits for easy movement",
        "Sleeves: 3/4 length with a classic mandarin collar",
        "Care: gentle hand wash in cold water, dry in shade"
      ],
      tags: ["cotton kurti", "block print kurti", "maroon kurti", "straight kurti", "women ethnic wear",
        "office wear kurti", "mandarin collar kurti", "festive kurti for women"],
      search_query: "maroon cotton block print straight kurti women"
    },
    pricing: {
      currency: "INR",
      competitors: [
        { title: "Jaipur Kurti Women Maroon Cotton Printed Straight Kurta", price: 449, source: "Myntra", link: "https://www.myntra.com/", thumbnail: "" },
        { title: "Anouk Women Block Print Pure Cotton Kurta", price: 599, source: "Ajio", link: "https://www.ajio.com/", thumbnail: "" },
        { title: "Maroon Printed Straight Cotton Kurti", price: 329, source: "Meesho", link: "https://www.meesho.com/", thumbnail: "" },
        { title: "Women Cotton Mandarin Collar Kurti - Maroon", price: 499, source: "Flipkart", link: "https://www.flipkart.com/", thumbnail: "" },
        { title: "Hand Block Printed Kurta for Women", price: 899, source: "Amazon.in", link: "https://www.amazon.in/", thumbnail: "" },
        { title: "Straight Fit Cotton Kurti Ethnic Daily Wear", price: 399, source: "Meesho", link: "https://www.meesho.com/", thumbnail: "" },
        { title: "Libas Women Maroon Ethnic Motifs Kurta", price: 549, source: "Libas", link: "https://www.libas.in/", thumbnail: "" },
        { title: "Gold Print Cotton Kurti 3/4 Sleeve", price: 479, source: "Snapdeal", link: "https://www.snapdeal.com/", thumbnail: "" },
        { title: "Biba Women Printed Straight Kurta", price: 699, source: "Biba", link: "https://www.biba.in/", thumbnail: "" },
        { title: "Cotton Block Print Kurti Combo", price: 299, source: "Meesho", link: "https://www.meesho.com/", thumbnail: "" }
      ],
      stats: { count: 10, min: 299, median: 489, max: 899, p25: 399, p75: 599 },
      suggested_price: 459,
      strategy: "Competitors sit between ₹399 and ₹599 for most listings (median ₹489). We undercut the median by about 5% and end the price in 9, which lands at ₹459 — inside the busy middle of the market.",
      margin_pct: null
    },
    meta: { writer_backend: "mock", latency_ms: 1840 }
  };

  const MOCK_EVAL = {
    available: true,
    demo: true,
    base_model: "base model (sample)",
    finetuned_model: "tinker://sample",
    n_test: 120,
    metrics: {
      format_valid: { base: 0.71, finetuned: 0.98 },
      constraint_compliance: { base: 0.52, finetuned: 0.91 },
      attribute_coverage: { base: 0.63, finetuned: 0.81 },
      rouge_l: { base: 0.21, finetuned: 0.34 }
    },
    judge: { finetuned_win: 0.64, base_win: 0.22, tie: 0.14 },
    examples: [{
      attributes: MOCK_RESULT.attributes,
      base: { title: "Beautiful Maroon Kurti For Women Very Nice Quality Best Price Cotton Kurti Block Print Ethnic Wear Party Office Daily Use", bullets: ["Nice kurti", "Good quality"], tags: ["kurti", "women kurti for festive occasions"] },
      finetuned: { title: MOCK_RESULT.listing.title, bullets: MOCK_RESULT.listing.bullets, tags: MOCK_RESULT.listing.tags.slice(0, 5) },
      reference: {}
    }]
  };

  // ---------- Small helpers ----------

  // Wait for a number of milliseconds (used for demo mode and the loading steps)
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Only allow normal web links (blocks things like "javascript:" links)
  function safeUrl(url) {
    try {
      const u = new URL(url, window.location.href);
      return u.protocol === "http:" || u.protocol === "https:" ? u.href : "";
    } catch (e) {
      return "";
    }
  }

  // Create an element with optional class and text in one line
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function showToast(message) {
    const t = $("toast");
    t.textContent = message;
    t.hidden = false;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { t.hidden = true; }, 1800);
  }

  // Copy text to the clipboard, with an old-style fallback for non-secure pages
  async function copyText(text, label) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      const ta = el("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    showToast((label || "Text") + " copied");
  }

  function currentPlatform() {
    return document.querySelector('input[name="platform"]:checked').value;
  }

  // ---------- Health badge ----------
  async function loadHealth() {
    const box = $("health");
    let h;
    try {
      if (DEMO) {
        h = MOCK_HEALTH;
      } else {
        const res = await fetch("/api/health");
        if (!res.ok) throw new Error("HTTP " + res.status);
        h = await res.json();
      }
    } catch (e) {
      box.className = "health down";
      box.querySelector(".health-text").textContent = "Backend offline — try ?demo=1";
      return;
    }

    box.className = "health ok";
    box.textContent = "";
    box.appendChild(el("span", "health-dot"));
    box.appendChild(el("span", "", DEMO ? "Demo" : "Writer: " + (h.writer_backend || "?")));
    // One small pill per service: green = live, amber = mock
    [["Gemma", h.gemma], ["SerpApi", h.serpapi], ["Tinker", h.tinker]].forEach(([name, live]) => {
      const pill = el("span", "svc " + (live ? "live" : "mock"), name + (live ? " live" : " mock"));
      pill.title = live ? name + " is connected" : name + " key missing, using offline mock";
      box.appendChild(pill);
    });
  }

  // ---------- Photo picking ----------
  const dropzone = $("dropzone");
  const fileInput = $("file-input");

  function addFiles(list) {
    const errors = [];
    for (const f of Array.from(list)) {
      if (files.length >= MAX_FILES) {
        errors.push("You can add up to " + MAX_FILES + " photos.");
        break;
      }
      if (!ALLOWED_TYPES.includes(f.type)) {
        errors.push(f.name + ": only JPG, PNG or WEBP images are allowed.");
        continue;
      }
      if (f.size > MAX_BYTES) {
        errors.push(f.name + ": file is larger than 8 MB.");
        continue;
      }
      // Skip the same photo picked twice
      if (files.some((x) => x.file.name === f.name && x.file.size === f.size)) continue;
      files.push({ file: f, url: URL.createObjectURL(f) });
    }
    const errBox = $("file-error");
    errBox.textContent = errors.join(" ");
    errBox.hidden = errors.length === 0;
    renderPreviews();
  }

  function removeFile(index) {
    URL.revokeObjectURL(files[index].url);
    files.splice(index, 1);
    $("file-error").hidden = true;
    renderPreviews();
  }

  function renderPreviews() {
    const ul = $("previews");
    ul.textContent = "";
    files.forEach((item, i) => {
      const li = el("li");
      const img = el("img");
      img.src = item.url;
      img.alt = "Photo " + (i + 1) + ": " + item.file.name;
      const btn = el("button", "remove-btn", "×");
      btn.type = "button";
      btn.setAttribute("aria-label", "Remove photo " + (i + 1));
      btn.addEventListener("click", () => removeFile(i));
      li.append(img, btn);
      ul.appendChild(li);
    });
    // Enable the button only when at least one photo is picked
    $("gen-btn").disabled = files.length === 0;
    $("form-hint").textContent = files.length === 0
      ? "Add at least one photo to start."
      : files.length + " of " + MAX_FILES + " photos added.";
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = ""; // so picking the same file again still fires "change"
  });
  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  // ---------- Loading steps ----------
  // The backend does all three steps in one request, so we move the labels on a timer
  // and mark everything done when the answer arrives.
  let stepTimer = null;
  function startProgress() {
    const box = $("progress");
    const items = box.querySelectorAll("li");
    box.hidden = false;
    let step = 0;
    const paint = () => items.forEach((li, i) => {
      li.className = i < step ? "done" : i === step ? "active" : "";
    });
    paint();
    stepTimer = setInterval(() => {
      if (step < items.length - 1) { step += 1; paint(); }
    }, 3500);
  }
  function stopProgress(success) {
    clearInterval(stepTimer);
    const box = $("progress");
    if (success) {
      box.querySelectorAll("li").forEach((li) => { li.className = "done"; });
      setTimeout(() => { box.hidden = true; }, 600);
    } else {
      box.hidden = true;
    }
  }

  // ---------- Generate ----------
  $("gen-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (files.length === 0) return;

    const btn = $("gen-btn");
    const errBox = $("gen-error");
    errBox.hidden = true;
    btn.disabled = true;
    btn.textContent = "Generating…";
    startProgress();

    try {
      let data;
      if (DEMO) {
        // Pretend to work for a few seconds so the loading steps are visible
        await sleep(3000);
        data = JSON.parse(JSON.stringify(MOCK_RESULT));
        const cost = parseFloat($("cost-price").value);
        if (cost > 0) {
          data.pricing.margin_pct = Math.round(((data.pricing.suggested_price - cost) / cost) * 100);
        }
      } else {
        // Build the multipart form. The field name "images" is repeated once per photo.
        const form = new FormData();
        files.forEach((item) => form.append("images", item.file, item.file.name));
        form.append("platform", currentPlatform());
        const notes = $("notes").value.trim();
        if (notes) form.append("notes", notes);
        const cost = $("cost-price").value.trim();
        if (cost) form.append("cost_price", cost);

        const res = await fetch("/api/generate", { method: "POST", body: form });
        if (!res.ok) {
          // FastAPI errors usually look like {"detail": "..."}
          let msg = "Request failed (HTTP " + res.status + ").";
          try {
            const body = await res.json();
            if (body && body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
          } catch (ignore) { /* not JSON */ }
          throw new Error(msg);
        }
        data = await res.json();
      }
      stopProgress(true);
      lastResult = data;
      renderResults(data);
    } catch (err) {
      stopProgress(false);
      errBox.textContent = "Could not generate the listing: " + err.message;
      errBox.hidden = false;
    } finally {
      btn.disabled = files.length === 0;
      btn.textContent = "Generate listing";
    }
  });

  // When the platform changes, update the title counter limit
  document.querySelectorAll('input[name="platform"]').forEach((r) => r.addEventListener("change", updateTitleCount));

  // ---------- Results ----------
  function renderResults(data) {
    const section = $("results");
    section.hidden = false;

    renderAttributes(data.attributes || {});
    renderListing(data.listing || {});
    renderPricing(data.pricing || {});

    const meta = data.meta || {};
    const parts = [];
    if (meta.writer_backend) parts.push("Writer: " + meta.writer_backend);
    if (typeof meta.latency_ms === "number") parts.push((meta.latency_ms / 1000).toFixed(1) + " s");
    $("meta").textContent = parts.join(" · ");

    section.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderAttributes(attrs) {
    const box = $("attributes");
    box.textContent = "";
    // Turn "product_type" into "product type"
    Object.entries(attrs).forEach(([key, value]) => {
      const values = Array.isArray(value) ? value : [value];
      values.filter((v) => v !== null && v !== undefined && String(v).trim() !== "").forEach((v) => {
        const chip = el("span", "chip");
        chip.appendChild(el("b", "", key.replace(/_/g, " ")));
        chip.appendChild(document.createTextNode(String(v)));
        box.appendChild(chip);
      });
    });
    if (!box.children.length) box.appendChild(el("span", "muted", "No attributes detected."));
  }

  function renderListing(listing) {
    $("out-title").value = listing.title || "";
    $("out-desc").value = listing.description || "";
    updateTitleCount();

    // Always show 5 bullet boxes so the seller can fill any missing ones
    const ol = $("out-bullets");
    ol.textContent = "";
    const bullets = (listing.bullets || []).slice(0, 5);
    while (bullets.length < 5) bullets.push("");
    bullets.forEach((b, i) => {
      const li = el("li");
      const num = el("span", "", String(i + 1));
      num.setAttribute("aria-hidden", "true");
      const ta = el("textarea");
      ta.rows = 1;
      ta.value = b;
      ta.setAttribute("aria-label", "Key point " + (i + 1));
      li.append(num, ta);
      ol.appendChild(li);
    });

    tags = (listing.tags || []).map(String);
    renderTags();
  }

  function updateTitleCount() {
    const limit = TITLE_LIMIT[currentPlatform()];
    const len = $("out-title").value.length;
    const counter = $("title-count");
    counter.textContent = len + " / " + limit;
    counter.classList.toggle("over", len > limit);
  }
  $("out-title").addEventListener("input", updateTitleCount);

  function renderTags() {
    const box = $("out-tags");
    box.textContent = "";
    tags.forEach((t, i) => {
      const tooLong = t.length > TAG_MAX_LEN;
      const chip = el("span", "chip tag" + (tooLong ? " too-long" : ""));
      chip.appendChild(document.createTextNode(t));
      if (tooLong) {
        chip.title = "Too long: " + t.length + " characters (max " + TAG_MAX_LEN + ")";
        chip.appendChild(el("b", "", t.length + "/" + TAG_MAX_LEN));
      }
      const rm = el("button", "", "×");
      rm.type = "button";
      rm.setAttribute("aria-label", "Remove tag " + t);
      rm.addEventListener("click", () => { tags.splice(i, 1); renderTags(); });
      chip.appendChild(rm);
      box.appendChild(chip);
    });
    const longCount = tags.filter((t) => t.length > TAG_MAX_LEN).length;
    const counter = $("tag-count");
    counter.textContent = tags.length + " / " + TAG_MAX_COUNT + (longCount ? " · " + longCount + " too long" : "");
    counter.classList.toggle("over", tags.length > TAG_MAX_COUNT || longCount > 0);
  }

  // Press Enter (or comma) in the tag box to add a tag
  $("tag-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      const value = e.target.value.trim().replace(/,$/, "");
      if (value && !tags.includes(value)) {
        tags.push(value);
        renderTags();
      }
      e.target.value = "";
    }
  });

  // Read the (possibly edited) listing back from the page
  function readListing() {
    return {
      title: $("out-title").value.trim(),
      description: $("out-desc").value.trim(),
      bullets: Array.from($("out-bullets").querySelectorAll("textarea")).map((t) => t.value.trim()).filter(Boolean),
      tags: tags.slice()
    };
  }

  // Per-field copy buttons
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const l = readListing();
      const field = btn.getAttribute("data-copy");
      const text = {
        title: l.title,
        description: l.description,
        bullets: l.bullets.map((b) => "• " + b).join("\n"),
        tags: l.tags.join(", ")
      }[field];
      copyText(text, field.charAt(0).toUpperCase() + field.slice(1));
    });
  });

  $("copy-all").addEventListener("click", () => {
    const l = readListing();
    let text = "TITLE\n" + l.title + "\n\nDESCRIPTION\n" + l.description +
      "\n\nKEY POINTS\n" + l.bullets.map((b) => "• " + b).join("\n") +
      "\n\nTAGS\n" + l.tags.join(", ");
    if (lastResult && lastResult.pricing && typeof lastResult.pricing.suggested_price === "number") {
      text += "\n\nSUGGESTED PRICE\n" + formatINR(lastResult.pricing.suggested_price);
    }
    copyText(text, "Full listing");
  });

  // ---------- Pricing ----------
  function renderPricing(p) {
    $("price-big").textContent = formatINR(p.suggested_price);
    $("margin").textContent = typeof p.margin_pct === "number"
      ? "About " + Math.round(p.margin_pct) + "% margin over your cost price."
      : "Add your cost price to see your margin.";
    $("strategy").textContent = p.strategy || "";
    $("strategy").hidden = !p.strategy;

    // Five summary numbers
    const s = p.stats || {};
    const dl = $("stats");
    dl.textContent = "";
    [["Min", s.min], ["P25", s.p25], ["Median", s.median], ["P75", s.p75], ["Max", s.max]].forEach(([name, v]) => {
      const wrap = el("div");
      wrap.append(el("dt", "", name), el("dd", "", formatINR(v)));
      dl.appendChild(wrap);
    });

    const comps = (p.competitors || []).filter((c) => typeof c.price === "number" && isFinite(c.price));
    renderStrip(comps, s, p.suggested_price);
    renderCompetitors(comps);
  }

  // Draw a simple dot plot: one dot per competitor price, the busy middle (P25-P75)
  // as a shaded band, and our suggested price as a bold line.
  function renderStrip(comps, s, suggested) {
    const box = $("strip");
    const prices = comps.map((c) => c.price);
    if (!prices.length) {
      box.textContent = "";
      box.removeAttribute("aria-label");
      return;
    }
    const all = prices.concat(typeof suggested === "number" ? [suggested] : []);
    let lo = Math.min.apply(null, all);
    let hi = Math.max.apply(null, all);
    if (hi === lo) { lo -= 50; hi += 50; }

    const W = 600, H = 96, PAD = 30, AXIS_Y = 58;
    const x = (v) => PAD + ((v - lo) / (hi - lo)) * (W - PAD * 2);

    let svg = '<svg viewBox="0 0 ' + W + " " + H + '" aria-hidden="true">';
    if (typeof s.p25 === "number" && typeof s.p75 === "number") {
      svg += '<rect class="iqr" x="' + x(s.p25) + '" y="' + (AXIS_Y - 16) + '" width="' +
        Math.max(2, x(s.p75) - x(s.p25)) + '" height="32" rx="6"></rect>';
    }
    svg += '<line class="axis" x1="' + PAD + '" y1="' + AXIS_Y + '" x2="' + (W - PAD) + '" y2="' + AXIS_Y + '"></line>';
    // Small up/down offsets so dots with the same price do not hide each other
    prices.forEach((v, i) => {
      const dy = ((i % 3) - 1) * 7;
      svg += '<circle class="dot" cx="' + x(v).toFixed(1) + '" cy="' + (AXIS_Y + dy) + '" r="5"></circle>';
    });
    if (typeof suggested === "number") {
      const sx = x(suggested).toFixed(1);
      svg += '<line class="sugg" x1="' + sx + '" y1="' + (AXIS_Y - 24) + '" x2="' + sx + '" y2="' + (AXIS_Y + 20) + '"></line>';
      // Keep the label inside the picture near the edges
      const anchor = x(suggested) < 80 ? "start" : x(suggested) > W - 80 ? "end" : "middle";
      svg += '<text class="sugg-label" x="' + sx + '" y="' + (AXIS_Y - 28) + '" text-anchor="' + anchor + '">You: ' + formatINR(suggested) + "</text>";
    }
    svg += '<text class="tick" x="' + PAD + '" y="' + (H - 4) + '" text-anchor="start">' + formatINR(lo) + "</text>";
    svg += '<text class="tick" x="' + (W - PAD) + '" y="' + (H - 4) + '" text-anchor="end">' + formatINR(hi) + "</text>";
    svg += "</svg>";
    box.innerHTML = svg; // safe: only numbers we formatted ourselves go into this string
    box.setAttribute("aria-label",
      prices.length + " competitor prices from " + formatINR(Math.min.apply(null, prices)) + " to " +
      formatINR(Math.max.apply(null, prices)) + ". Suggested price " + formatINR(suggested) + ".");
  }

  function renderCompetitors(comps) {
    const ul = $("competitors");
    ul.textContent = "";
    if (!comps.length) {
      ul.appendChild(el("li", "muted", "No competitor prices found."));
      return;
    }
    // Cheapest first
    comps.slice().sort((a, b) => a.price - b.price).forEach((c) => {
      const li = el("li");
      const thumb = safeUrl(c.thumbnail || "");
      if (thumb) {
        const img = el("img");
        img.src = thumb;
        img.alt = "";
        img.loading = "lazy";
        img.referrerPolicy = "no-referrer";
        img.addEventListener("error", () => img.replaceWith(el("span", "thumb-ph")));
        li.appendChild(img);
      } else {
        li.appendChild(el("span", "thumb-ph"));
      }

      const body = el("div", "comp-body");
      const link = safeUrl(c.link || "");
      const title = el(link ? "a" : "span", "comp-title", c.title || "Untitled product");
      if (link) {
        title.href = link;
        title.target = "_blank";
        title.rel = "noopener noreferrer";
      }
      body.append(title, el("div", "comp-src", c.source || ""));
      li.append(body, el("span", "comp-price", formatINR(c.price)));
      ul.appendChild(li);
    });
  }

  // ---------- Base vs Fine-tuned panel ----------
  async function loadEval() {
    const body = $("eval-body");
    let ev;
    try {
      if (DEMO) {
        ev = MOCK_EVAL;
      } else {
        const res = await fetch("/api/eval");
        if (!res.ok) throw new Error("HTTP " + res.status);
        ev = await res.json();
      }
    } catch (e) {
      ev = { available: false };
    }

    body.textContent = "";
    if (!ev || !ev.available) {
      const p = el("p", "muted");
      p.append("Run ", el("code", "", "finetune/evaluate.py"), " to see measured results.");
      body.appendChild(p);
      return;
    }

    if (ev.demo) body.appendChild(el("p", "demo-banner", "Sample numbers shown in demo mode, not real results."));

    const info = el("p", "small muted",
      "Test items: " + (ev.n_test || "?") + " · Base: " + (ev.base_model || "?") + " · Fine-tuned: " + (ev.finetuned_model || "?"));
    info.style.overflowWrap = "anywhere";
    body.appendChild(info);

    // Metric table
    const metrics = ev.metrics || {};
    const names = Object.keys(metrics).filter((k) => metrics[k] && typeof metrics[k].base === "number");
    if (names.length) {
      const wrap = el("div", "table-wrap");
      const table = el("table");
      const thead = el("thead");
      const hr = el("tr");
      [["Metric", ""], ["Base", "num"], ["Fine-tuned", "num"], ["Change", "num"]].forEach(([t, c]) => {
        const th = el("th", c, t);
        th.scope = "col";
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      const tbody = el("tbody");
      names.forEach((name) => {
        const m = metrics[name];
        const tr = el("tr");
        const isRate = Math.abs(m.base) <= 1 && Math.abs(m.finetuned) <= 1; // 0-1 numbers shown as %
        const fmt = (v) => (typeof v !== "number" ? "–" : isRate ? (v * 100).toFixed(1) + "%" : v.toFixed(2));
        const delta = typeof m.finetuned === "number" ? m.finetuned - m.base : null;
        const dText = delta === null ? "–" : (delta >= 0 ? "+" : "") + (isRate ? (delta * 100).toFixed(1) + " pts" : delta.toFixed(2));
        const th = el("th", "", prettyName(name));
        th.scope = "row";
        tr.append(th, el("td", "num", fmt(m.base)), el("td", "num", fmt(m.finetuned)),
          el("td", "num " + (delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : ""), dText));
        tbody.appendChild(tr);
      });
      table.append(thead, tbody);
      wrap.appendChild(table);
      body.appendChild(wrap);
    }

    // Judge win-rate bar
    const j = ev.judge;
    if (j && typeof j.finetuned_win === "number") {
      body.appendChild(el("h3", "sub-title", "Blind LLM-judge: which listing is better?"));
      const bar = el("div", "winbar");
      bar.setAttribute("role", "img");
      const pct = (v) => Math.round((v || 0) * 100);
      bar.setAttribute("aria-label", "Fine-tuned wins " + pct(j.finetuned_win) + "%, ties " + pct(j.tie) + "%, base wins " + pct(j.base_win) + "%");
      [["w-ft", j.finetuned_win], ["w-tie", j.tie], ["w-base", j.base_win]].forEach(([cls, v]) => {
        if (!v) return;
        const seg = el("div", cls, pct(v) >= 8 ? pct(v) + "%" : "");
        seg.style.flex = String(v);
        bar.appendChild(seg);
      });
      body.appendChild(bar);
      const legend = el("div", "legend");
      [["Fine-tuned wins", "var(--accent)", j.finetuned_win], ["Tie", "var(--surface-2)", j.tie], ["Base wins", "var(--muted)", j.base_win]]
        .forEach(([t, color, v]) => {
          const s = el("span");
          const sw = el("i");
          sw.style.background = color;
          s.append(sw, t + " " + pct(v) + "%");
          legend.appendChild(s);
        });
      body.appendChild(legend);
    }

    // One example, side by side
    const ex = (ev.examples || [])[0];
    if (ex) {
      body.appendChild(el("h3", "sub-title", "Example from the test set"));
      const grid = el("div", "side-by-side");
      grid.append(exampleBox("Base model", ex.base, ""), exampleBox("Fine-tuned", ex.finetuned, "ft"));
      body.appendChild(grid);
    }
  }

  // "format_valid" -> "Format valid"
  function prettyName(key) {
    const nice = { rouge_l: "ROUGE-L", format_valid: "Format valid", constraint_compliance: "Constraint compliance", attribute_coverage: "Attribute coverage" };
    if (nice[key]) return nice[key];
    const s = key.replace(/_/g, " ");
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // Show one model's listing inside the example comparison
  function exampleBox(heading, listing, cls) {
    const box = el("div", cls);
    box.appendChild(el("h4", "", heading));
    if (!listing || typeof listing !== "object") {
      // The model may have returned raw text that could not be parsed
      box.appendChild(el("p", "", listing ? String(listing) : "(no output)"));
      return box;
    }
    if (listing.title) {
      box.appendChild(el("p", "ex-title", listing.title));
      box.appendChild(el("p", "small muted", listing.title.length + " characters"));
    }
    if (Array.isArray(listing.bullets) && listing.bullets.length) {
      const ul = el("ul");
      listing.bullets.forEach((b) => ul.appendChild(el("li", "", b)));
      box.appendChild(ul);
    }
    if (Array.isArray(listing.tags) && listing.tags.length) {
      const chips = el("div", "chips");
      listing.tags.forEach((t) => chips.appendChild(el("span", "chip tag" + (String(t).length > TAG_MAX_LEN ? " too-long" : ""), t)));
      box.appendChild(chips);
    }
    return box;
  }

  // ---------- Start ----------
  if (DEMO) $("demo-banner").hidden = false;
  loadHealth();
  loadEval();
})();
