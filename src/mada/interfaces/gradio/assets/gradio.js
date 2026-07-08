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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initResizableChat, { once: true });
  } else {
    initResizableChat();
  }
})();
