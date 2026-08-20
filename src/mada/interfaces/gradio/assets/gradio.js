(() => {
  const storageKey = "mada.gradio.chatbotHeightPx";
  const minHeightPx = 240;

  function initResizableChat(remainingAttempts = 50) {
    const chatbot = document.getElementById("mada-chatbot");
    if (!chatbot) {
      if (remainingAttempts > 0) {
        window.setTimeout(() => initResizableChat(remainingAttempts - 1), 200);
      }
      return;
    }
    let handle = document.getElementById("mada-chat-resize-handle");
    if (!handle) {
      handle = document.createElement("div");
      handle.id = "mada-chat-resize-handle";
      handle.setAttribute("aria-label", "Resize chat");
      handle.title = "Drag to resize chat";
      chatbot.appendChild(handle);
    }
    if (handle.dataset.madaResizeInit === "1") return;
    handle.dataset.madaResizeInit = "1";

    try {
      const saved = Number.parseInt(window.localStorage.getItem(storageKey) || "", 10);
      if (Number.isFinite(saved) && saved >= minHeightPx) {
        chatbot.style.height = `${saved}px`;
      }
    } catch (e) {
      // localStorage unavailable (private browsing, etc.) - skip restoration
    }

    let dragging = false;
    let startY = 0;
    let startHeight = 0;

    const onMouseMove = (event) => {
      if (!dragging) return;
      const deltaY = event.clientY - startY;
      const next = Math.max(minHeightPx, startHeight + deltaY);
      chatbot.style.height = `${Math.round(next)}px`;
    };

    const stopDragging = () => {
      if (!dragging) return;
      dragging = false;
      document.body.classList.remove("mada-resize-dragging");
      const current = Number.parseInt(chatbot.style.height || "", 10);
      if (Number.isFinite(current) && current >= minHeightPx) {
        try {
          window.localStorage.setItem(storageKey, String(current));
        } catch (e) {
          // localStorage unavailable - resize still works, just won't persist
        }
      }
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", stopDragging);
    };

    handle.addEventListener("mousedown", (event) => {
      event.preventDefault();
      dragging = true;
      startY = event.clientY;
      startHeight = chatbot.getBoundingClientRect().height;
      document.body.classList.add("mada-resize-dragging");
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", stopDragging);
    });
  }

  function initSmartAutoscroll(remainingAttempts = 50) {
    const chatbot = document.getElementById("mada-chatbot");
    if (!chatbot) {
      if (remainingAttempts > 0) {
        window.setTimeout(() => initSmartAutoscroll(remainingAttempts - 1), 200);
      }
      return;
    }

    // Find the scrollable container within the chatbot
    const scrollContainer = chatbot.querySelector('.scroll-hide, .overflow-y-auto, [class*="overflow"]');
    if (!scrollContainer) {
      if (remainingAttempts > 0) {
        window.setTimeout(() => initSmartAutoscroll(remainingAttempts - 1), 200);
      }
      return;
    }

    if (scrollContainer.dataset.madaScrollInit === "1") return;
    scrollContainer.dataset.madaScrollInit = "1";

    let userHasScrolledUp = false;
    let lastScrollTop = scrollContainer.scrollTop;
    let lastScrollHeight = scrollContainer.scrollHeight;

    // Detect when user manually scrolls
    scrollContainer.addEventListener('scroll', () => {
      const scrollTop = scrollContainer.scrollTop;
      const scrollHeight = scrollContainer.scrollHeight;
      const clientHeight = scrollContainer.clientHeight;
      const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 5;

      // If user scrolled up manually (not due to content change)
      if (scrollTop < lastScrollTop && scrollHeight === lastScrollHeight) {
        userHasScrolledUp = true;
      }

      // If user scrolled back to bottom, re-enable auto-scroll
      if (isAtBottom) {
        userHasScrolledUp = false;
      }

      lastScrollTop = scrollTop;
      lastScrollHeight = scrollHeight;
    }, { passive: true });

    // Observe content changes and auto-scroll only if user hasn't scrolled up
    const observer = new MutationObserver(() => {
      if (!userHasScrolledUp) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    });

    observer.observe(scrollContainer, {
      childList: true,
      subtree: true,
      characterData: true
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      initResizableChat();
      initSmartAutoscroll();
    }, { once: true });
  } else {
    initResizableChat();
    initSmartAutoscroll();
  }
})();
