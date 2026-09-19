/* Standalone slide-deck runtime. No dependencies; load with defer or at body end. */
(() => {
  "use strict";

  function start() {
    const slides = Array.from(document.querySelectorAll(".slide"));
    const stage = document.getElementById("stage");
    const viewport = document.getElementById("viewport");
    if (!slides.length || !stage || !viewport) return;

    const byId = (id) => document.getElementById(id);
    const controls = {
      prev: byId("prev"), next: byId("next"), counter: byId("counter"),
      title: byId("slide-title"), progress: byId("progress"),
      fullscreen: byId("fullscreen"), print: byId("print")
    };
    const overlays = ["notes", "overview", "help"].map((name) => ({
      name, button: byId(`${name}-toggle`), panel: byId(`${name}-panel`)
    })).filter(({ button, panel }) => button && panel);
    const notesContent = byId("notes-content");
    const overviewPanel = byId("overview-panel");
    let index = 0;
    let openedOverlay = null;
    let overlayReturnFocus = null;
    let printing = false;
    let resizeFrame = 0;

    const titleOf = (slide, i) => slide.dataset.title ||
      slide.querySelector("h1, h2, h3")?.textContent.trim() || `Slide ${i + 1}`;

    slides.forEach((slide, i) => {
      if (!slide.id) slide.id = `slide-${i + 1}`;
      slide.setAttribute("role", "group");
      slide.setAttribute("aria-roledescription", "slide");
      slide.setAttribute("aria-label", `${i + 1} of ${slides.length}: ${titleOf(slide, i)}`);
      slide.tabIndex = -1;
      // Notes are source material, never part of the projected slide.
      slide.querySelectorAll("aside.speaker-notes").forEach((notes) => {
        notes.hidden = true;
        notes.setAttribute("aria-hidden", "true");
        notes.inert = true;
      });
    });

    function scaleStage() {
      resizeFrame = 0;
      const bounds = viewport.getBoundingClientRect();
      const scale = Math.max(0.01, Math.min(bounds.width / 1600, bounds.height / 900));
      viewport.style.setProperty("--deck-scale", String(scale));
      stage.style.width = "1600px";
      stage.style.height = "900px";
      stage.style.position = "absolute";
      stage.style.left = "50%";
      stage.style.top = "50%";
      stage.style.transformOrigin = "center center";
      stage.style.transform = `translate(-50%, -50%) scale(${scale})`;
    }

    function scheduleScale() {
      if (!resizeFrame) resizeFrame = requestAnimationFrame(scaleStage);
    }

    function updateNotes() {
      if (!notesContent) return;
      const source = slides[index].querySelector("aside.speaker-notes");
      notesContent.replaceChildren();
      const title = document.createElement("h2");
      title.className = "notes-current-title";
      title.textContent = `${index + 1}. ${titleOf(slides[index], index)}`;
      notesContent.append(title);
      if (source) {
        Array.from(source.childNodes).forEach((node) => {
          const copy = node.cloneNode(true);
          // The source remains in the DOM. Avoid duplicate IDs in copied notes.
          if (copy.nodeType === Node.ELEMENT_NODE) {
            copy.removeAttribute("id");
            copy.querySelectorAll("[id]").forEach((element) => element.removeAttribute("id"));
          }
          notesContent.append(copy);
        });
      } else {
        const empty = document.createElement("p");
        empty.textContent = "No speaker notes for this slide.";
        notesContent.append(empty);
      }
      notesContent.scrollTop = 0;
    }

    function updateOverview() {
      if (!overviewPanel) return;
      overviewPanel.querySelectorAll(".overview-item[data-slide-index]").forEach((button) => {
        const active = Number(button.dataset.slideIndex) === index;
        button.classList.toggle("is-current", active);
        if (active) button.setAttribute("aria-current", "true");
        else button.removeAttribute("aria-current");
      });
    }

    function render() {
      const focusedSlide = document.activeElement?.closest?.(".slide");
      slides.forEach((slide, i) => {
        const active = i === index;
        slide.classList.toggle("active", active);
        slide.hidden = printing ? false : !active;
        slide.inert = !active;
        slide.setAttribute("aria-hidden", String(!active));
      });
      if (focusedSlide && focusedSlide !== slides[index]) {
        slides[index].focus({ preventScroll: true });
      }
      if (controls.prev) controls.prev.disabled = index === 0;
      if (controls.next) controls.next.disabled = index === slides.length - 1;
      if (controls.counter) controls.counter.textContent = `${index + 1} / ${slides.length}`;
      if (controls.title) controls.title.textContent = titleOf(slides[index], index);
      if (controls.progress) {
        const percent = ((index + 1) / slides.length) * 100;
        controls.progress.style.setProperty("--progress", `${percent}%`);
        if (controls.progress.tagName === "PROGRESS") {
          controls.progress.max = slides.length;
          controls.progress.value = index + 1;
        } else {
          controls.progress.style.width = `${percent}%`;
          controls.progress.setAttribute("role", "progressbar");
          controls.progress.setAttribute("aria-valuemin", "1");
          controls.progress.setAttribute("aria-valuemax", String(slides.length));
          controls.progress.setAttribute("aria-valuenow", String(index + 1));
        }
        controls.progress.setAttribute("aria-label", "Slide progress");
      }
      updateNotes();
      updateOverview();
    }

    function goTo(nextIndex, options = {}) {
      if (!Number.isFinite(Number(nextIndex))) return;
      const previousIndex = index;
      index = Math.max(0, Math.min(slides.length - 1, Math.trunc(Number(nextIndex))));
      render();
      if (options.updateHash !== false) {
        const hash = `#slide-${index + 1}`;
        if (location.hash !== hash) {
          // Replacement avoids a browser-history entry for every slide advance.
          try { history.replaceState(null, "", hash); }
          catch (_) { location.hash = hash; }
        }
      }
      if (options.focus) slides[index].focus({ preventScroll: true });
      if (previousIndex !== index || options.initial) {
        window.dispatchEvent(new CustomEvent("slidechange", {
          detail: { index, previousIndex, slide: slides[index], total: slides.length }
        }));
      }
    }

    function indexFromHash() {
      const match = /^#slide-(\d+)$/.exec(location.hash);
      return match ? Math.max(0, Math.min(slides.length - 1, Number(match[1]) - 1)) : 0;
    }

    function closeOverlays(restoreFocus = false) {
      overlays.forEach(({ button, panel }) => {
        panel.hidden = true;
        panel.inert = true;
        button.setAttribute("aria-expanded", "false");
      });
      openedOverlay = null;
      if (restoreFocus && overlayReturnFocus?.isConnected && !overlayReturnFocus.disabled) {
        overlayReturnFocus.focus({ preventScroll: true });
      }
      overlayReturnFocus = null;
    }

    function toggleOverlay(name) {
      if (openedOverlay === name) {
        closeOverlays(true);
        return;
      }
      const overlay = overlays.find((item) => item.name === name);
      if (!overlay) return;
      closeOverlays(false);
      overlayReturnFocus = overlay.button;
      openedOverlay = name;
      overlay.panel.hidden = false;
      overlay.panel.inert = false;
      overlay.button.setAttribute("aria-expanded", "true");
      if (name === "notes") updateNotes();
      const focusTarget = name === "overview"
        ? overlay.panel.querySelector(".is-current")
        : overlay.panel;
      (focusTarget || overlay.panel).focus({ preventScroll: true });
    }

    overlays.forEach(({ name, button, panel }) => {
      panel.hidden = true;
      panel.inert = true;
      panel.tabIndex = -1;
      if (!panel.hasAttribute("role")) panel.setAttribute("role", "region");
      if (!panel.hasAttribute("aria-label")) {
        panel.setAttribute("aria-label", name === "notes" ? "Speaker notes" :
          name === "overview" ? "Slide overview" : "Keyboard shortcuts");
      }
      button.setAttribute("aria-controls", panel.id);
      button.setAttribute("aria-expanded", "false");
      button.addEventListener("click", () => toggleOverlay(name));
      panel.querySelectorAll("[data-close-overlay]").forEach((close) => {
        close.addEventListener("click", () => closeOverlays(true));
      });
    });

    if (overviewPanel) {
      // Supply #overview-list to preserve a custom panel header/close button.
      const list = byId("overview-list") || overviewPanel;
      const generated = document.createElement("div");
      generated.className = "overview-grid";
      slides.forEach((slide, i) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "overview-item";
        button.dataset.slideIndex = String(i);
        if (slide.dataset.kind) button.dataset.kind = slide.dataset.kind;
        const pieces = [
          ["overview-number", String(i + 1).padStart(2, "0")],
          ["overview-title", titleOf(slide, i)],
          ["overview-section", slide.dataset.section || ""],
          ["overview-time", slide.dataset.minutes ? `${slide.dataset.minutes} min` : ""]
        ];
        pieces.forEach(([className, text]) => {
          if (!text) return;
          const span = document.createElement("span");
          span.className = className;
          span.textContent = text;
          button.append(span);
        });
        button.addEventListener("click", () => {
          closeOverlays(false);
          goTo(i, { focus: true });
        });
        generated.append(button);
      });
      list.append(generated);
    }

    controls.prev?.addEventListener("click", () => goTo(index - 1));
    controls.next?.addEventListener("click", () => goTo(index + 1));
    controls.print?.addEventListener("click", () => {
      closeOverlays(false);
      window.print();
    });

    if (controls.fullscreen) {
      if (!document.documentElement.requestFullscreen) {
        controls.fullscreen.disabled = true;
        controls.fullscreen.title = "Fullscreen is unavailable in this browser. Use the browser's fullscreen control.";
      }
      controls.fullscreen.setAttribute("aria-pressed", "false");
      controls.fullscreen.addEventListener("click", async () => {
        try {
          if (document.fullscreenElement) await document.exitFullscreen();
          else await document.documentElement.requestFullscreen();
        } catch (_) {
          controls.fullscreen.title = "Fullscreen could not be opened. Use the browser's fullscreen control.";
        }
      });
      document.addEventListener("fullscreenchange", () => {
        controls.fullscreen.setAttribute("aria-pressed", String(Boolean(document.fullscreenElement)));
        scheduleScale();
      });
    }

    const interactiveSelector = [
      "input", "textarea", "select", "button", "a[href]", "summary", "audio", "video",
      '[contenteditable]:not([contenteditable="false"])', '[role="slider"]',
      '[role="spinbutton"]', '[role="combobox"]', '[role="textbox"]', "[data-deck-ignore-keys]"
    ].join(",");

    document.addEventListener("keydown", (event) => {
      if (event.defaultPrevented || event.isComposing) return;
      if (event.key === "Escape") {
        if (openedOverlay) {
          event.preventDefault();
          closeOverlays(true);
        }
        return;
      }
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      // Chrome buttons keep native Space/Enter activation, but navigation keys
      // still work after a presenter clicks Next or closes a panel.
      const chromeNavigationKey = Boolean(event.target.closest?.(".deck-chrome button")) &&
        ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "PageDown", "PageUp", "Home", "End"].includes(event.key);
      if (event.target.closest?.(interactiveSelector) && !chromeNavigationKey) return;
      // Open panels own their scrolling keys; do not advance behind an overlay.
      if (openedOverlay) return;
      switch (event.key) {
        case "ArrowRight": case "ArrowDown": case "PageDown": case " ":
          event.preventDefault(); goTo(index + 1); break;
        case "ArrowLeft": case "ArrowUp": case "PageUp":
          event.preventDefault(); goTo(index - 1); break;
        case "Home": event.preventDefault(); goTo(0); break;
        case "End": event.preventDefault(); goTo(slides.length - 1); break;
        default: break;
      }
    });

    window.addEventListener("hashchange", () => goTo(indexFromHash(), { updateHash: false }));
    window.addEventListener("resize", scheduleScale, { passive: true });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(scheduleScale).observe(viewport);
    window.addEventListener("beforeprint", () => {
      printing = true;
      document.body.classList.add("printing");
      slides.forEach((slide) => { slide.hidden = false; });
    });
    window.addEventListener("afterprint", () => {
      printing = false;
      document.body.classList.remove("printing");
      render();
      scheduleScale();
    });

    window.deck = Object.freeze({
      goTo,
      next: () => goTo(index + 1),
      previous: () => goTo(index - 1),
      get index() { return index; },
      get total() { return slides.length; }
    });
    goTo(indexFromHash(), { initial: true });
    scaleStage();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
