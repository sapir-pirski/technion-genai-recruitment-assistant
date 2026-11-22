// No messages or documents leave the browser. Hooks are scoped to this application's keys.
export default function ({ parentElement, data = {} }) {
  const doc = parentElement.ownerDocument;
  const app = doc.querySelector('.stApp');
  if (!app) return;
  const view = doc.defaultView;
  const main = app.querySelector('[data-testid="stMain"]') || doc.scrollingElement;
  const latest = parentElement.querySelector('.latest');
  const uploadSelector = '.st-key-introduction_file [data-testid="stFileUploaderDropzone"] > span > button';
  const listeners = [];
  const timers = new Set();
  const enhanced = new Map();
  let disposed = false;
  let tooltipButton = null;
  let dismissed = null;
  let openTimer;
  let closeTimer;

  const state = app.recruitmentScrollState?.conversation === data.conversation_id
    ? app.recruitmentScrollState
    : { conversation: data.conversation_id, request: null, nearBottom: true, follow: true, interaction: 0 };
  app.recruitmentScrollState = state;

  const tooltip = doc.createElement('span');
  tooltip.className = 'candidate-upload-tooltip';
  tooltip.id = 'candidate-upload-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.textContent = 'Upload CV or profile';
  tooltip.hidden = true;
  app.append(tooltip);

  function listen(target, event, handler, options) {
    target.addEventListener(event, handler, options);
    listeners.push(() => target.removeEventListener(event, handler, options));
  }
  function later(handler, delay) {
    const timer = view.setTimeout(() => { timers.delete(timer); if (!disposed) handler(); }, delay);
    timers.add(timer);
    return timer;
  }
  function hideTooltip() {
    view.clearTimeout(openTimer);
    view.clearTimeout(closeTimer);
    tooltip.hidden = true;
    tooltipButton = null;
  }
  function positionTooltip() {
    if (!tooltipButton?.isConnected) return hideTooltip();
    const box = tooltipButton.getBoundingClientRect();
    const width = tooltip.offsetWidth;
    tooltip.style.left = `${Math.max(8, Math.min(box.left, view.innerWidth - width - 8))}px`;
    const bottom = view.visualViewport?.height ?? view.innerHeight;
    tooltip.style.top = `${box.bottom + tooltip.offsetHeight + 12 < bottom
      ? box.bottom + 8 : Math.max(8, box.top - tooltip.offsetHeight - 8)}px`;
  }
  function showTooltip(button) {
    view.clearTimeout(closeTimer);
    view.clearTimeout(openTimer);
    if (button.disabled || dismissed === button) return;
    tooltipButton = button;
    openTimer = later(() => { tooltip.hidden = false; positionTooltip(); }, 200);
  }
  function leaveTooltip(event) {
    if (tooltip.contains(event.relatedTarget) || tooltipButton?.contains(event.relatedTarget)) return;
    closeTimer = later(hideTooltip, 120);
  }
  function enhanceUpload() {
    app.querySelectorAll(uploadSelector).forEach(button => {
      if (enhanced.has(button)) return;
      enhanced.set(button, [button.getAttribute('aria-label'), button.getAttribute('aria-describedby')]);
      button.setAttribute('aria-label', 'Upload CV or profile');
      button.setAttribute('aria-describedby', [button.getAttribute('aria-describedby'), tooltip.id].filter(Boolean).join(' '));
      listen(button, 'pointerenter', () => { dismissed = null; showTooltip(button); });
      listen(button, 'pointerleave', leaveTooltip);
      listen(button, 'focus', () => { dismissed = null; showTooltip(button); });
      listen(button, 'blur', hideTooltip);
    });
  }
  listen(tooltip, 'pointerenter', () => view.clearTimeout(closeTimer));
  listen(tooltip, 'pointerleave', leaveTooltip);
  listen(app, 'keydown', event => {
    if (event.key === 'Escape' && tooltipButton) {
      dismissed = tooltipButton;
      hideTooltip();
    }
  });

  function answerTarget() {
    return [...app.querySelectorAll('[data-testid="stChatMessageContent"][aria-label="Chat message from assistant"]')]
      .at(-1)?.closest('[data-testid="stChatMessage"]');
  }
  function jump() {
    const target = answerTarget();
    if (!target) return;
    const top = main.scrollTop + target.getBoundingClientRect().top - main.getBoundingClientRect().top - 16;
    main.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
    state.follow = true;
    latest.hidden = true;
  }
  function latestVisible() {
    const box = answerTarget()?.getBoundingClientRect();
    const footer = Number.parseFloat(app.style.getPropertyValue('--ui-composer-space')) || 0;
    const visible = box && box.bottom > main.getBoundingClientRect().top
      && box.top < (view.visualViewport?.height ?? view.innerHeight) - footer;
    return Boolean(visible);
  }
  function updateLatest() {
    latest.hidden = !data.latest_request_id || state.nearBottom || latestVisible();
  }
  listen(latest, 'click', jump);
  listen(main, 'scroll', () => {
    state.nearBottom = main.scrollHeight - main.clientHeight - main.scrollTop < 96;
    state.follow = state.nearBottom || latestVisible();
    updateLatest();
    if (!tooltip.hidden) positionTooltip();
  }, { passive: true });
  for (const event of ['wheel', 'touchstart', 'pointerdown', 'keydown']) {
    listen(main, event, () => { state.interaction += 1; }, { passive: true });
  }

  let watchedFooter = null;
  let sizes;
  function viewport() {
    const composer = app.querySelector('[data-testid="stBottom"] .st-key-composer');
    const currentFooter = composer?.closest('[data-testid="stBottom"]');
    if (currentFooter && sizes && currentFooter !== watchedFooter) {
      if (watchedFooter) sizes.unobserve(watchedFooter);
      watchedFooter = currentFooter;
      sizes.observe(currentFooter);
    }
    const height = composer?.getBoundingClientRect().height ?? 0;
    app.style.setProperty('--ui-composer-space', `${height ? height + 24 : 0}px`);
    const vv = view.visualViewport;
    const typing = doc.activeElement?.matches('.st-key-composer textarea');
    const inset = typing && vv && vv.scale === 1
      ? Math.max(0, view.innerHeight - vv.height - vv.offsetTop) : 0;
    app.style.setProperty('--ui-keyboard-inset', `${inset}px`);
    if (!tooltip.hidden) positionTooltip();
  }
  listen(view, 'resize', viewport);
  if (view.visualViewport) {
    listen(view.visualViewport, 'resize', viewport);
    listen(view.visualViewport, 'scroll', viewport);
  }
  listen(app, 'focusin', event => {
    viewport();
    const target = event.target;
    if (!target?.getBoundingClientRect || target.closest('[data-testid="stBottom"]')) return;
    const box = target.getBoundingClientRect();
    const footer = Number.parseFloat(app.style.getPropertyValue('--ui-composer-space')) || 0;
    const available = (view.visualViewport?.height ?? view.innerHeight) - footer - 8;
    if (box.bottom > available) main.scrollBy({ top: box.bottom - available, behavior: 'auto' });
  });
  const observer = new view.MutationObserver(() => { enhanceUpload(); viewport(); });
  observer.observe(app, { childList: true, subtree: true });
  sizes = new view.ResizeObserver(viewport);
  enhanceUpload();
  viewport();
  updateLatest();
  if (data.latest_request_id && data.latest_request_id !== state.request) {
    const follow = state.follow ?? state.nearBottom;
    const interaction = state.interaction;
    state.request = data.latest_request_id;
    later(() => { if (follow && interaction === state.interaction) jump(); }, 150);
  }

  return () => {
    disposed = true;
    observer.disconnect();
    sizes.disconnect();
    listeners.forEach(remove => remove());
    timers.forEach(timer => view.clearTimeout(timer));
    tooltip.remove();
    enhanced.forEach(([label, description], button) => {
      if (label === null) button.removeAttribute('aria-label'); else button.setAttribute('aria-label', label);
      if (description === null) button.removeAttribute('aria-describedby'); else button.setAttribute('aria-describedby', description);
    });
  };
}
