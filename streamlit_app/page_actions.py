"""PDF download, print and screen-recording controls for the candidate header."""

import html
from base64 import b64encode
from pathlib import Path

import streamlit as st

HTML = """
<div class="page-actions" role="group" aria-label="Page actions">
  <div class="pdf-actions"></div>
  <div class="action-control">
  <button type="button" data-action="print" aria-label="Print conversation"
          aria-describedby="print-tooltip">
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 8V3h12v5M6 17H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
      <path d="M6 14h12v7H6zM18 11h.01"/>
    </svg>
  </button>
  <span class="tooltip" id="print-tooltip" role="tooltip">Print conversation</span>
  </div>
  <div class="action-control">
  <button type="button" data-action="record" aria-label="Record screen"
          aria-describedby="record-tooltip" aria-pressed="false">
    <svg class="record-icon" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="2" y="3" width="20" height="14" rx="2"/>
      <path d="M8 21h8M12 17v4"/><circle cx="12" cy="10" r="3"/>
    </svg>
    <svg class="stop-icon" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2"/>
    </svg>
  </button>
  <span class="tooltip" id="record-tooltip" role="tooltip" data-tooltip="record">Record screen</span>
  </div>
  <span class="feedback" role="status" aria-live="polite" aria-atomic="true"></span>
  <a class="download" hidden>Download recording</a>
</div>
"""

CSS = """
.page-actions { position: relative; display: flex; gap: 8px; justify-content: flex-end; }
.pdf-actions { display: flex; gap: 8px; }
.action-control { position: relative; display: flex; }
button, .icon-button {
  display: inline-flex; align-items: center; justify-content: center;
  width: 44px; height: 44px; box-sizing: border-box;
  padding: 10px; border: 1px solid var(--ui-input-border); border-radius: 9px;
  background: var(--ui-surface); color: var(--ui-muted); cursor: pointer; text-decoration: none;
}
button:enabled:hover, .icon-button:hover {
  background: var(--ui-tint); color: var(--ui-accent); border-color: var(--ui-accent);
}
button:focus-visible, a:focus-visible { outline: 3px solid var(--ui-accent); outline-offset: 3px; }
button:disabled { opacity: .5; cursor: not-allowed; }
svg { width: 22px; height: 22px; fill: none; stroke: currentColor;
      stroke-width: 1.75; stroke-linecap: round; stroke-linejoin: round; }
.stop-icon { display: none; fill: currentColor; stroke: none; }
button[aria-pressed="true"], button[aria-pressed="true"]:enabled:hover {
  color: var(--ui-error); background: var(--ui-error-bg); border-color: var(--ui-error);
}
button[aria-pressed="true"] .record-icon { display: none; }
button[aria-pressed="true"] .stop-icon { display: block; }
.tooltip {
  position: absolute; z-index: 20; top: calc(100% + 8px); right: 0;
  width: max-content; max-width: min(220px, calc(100vw - 48px));
  box-sizing: border-box; padding: 6px 10px; border-radius: 6px;
  background: var(--ui-text); color: var(--ui-inverse); font: 12px/1.5 sans-serif;
  white-space: normal; box-shadow: 0 3px 10px var(--ui-shadow);
  visibility: hidden; opacity: 0; pointer-events: none;
  transition: opacity .1s ease, visibility 0s;
}
.tooltip::before {
  content: ""; position: absolute; left: 0; right: 0; top: -8px; height: 8px;
}
.tooltip::after {
  content: ""; position: absolute; top: -4px; right: 18px;
  width: 8px; height: 8px; background: var(--ui-text); transform: rotate(45deg);
}
.action-control:not([data-tooltip-dismissed="true"]):hover .tooltip,
.action-control:not([data-tooltip-dismissed="true"]):has(:focus-visible) .tooltip {
  visibility: visible; opacity: 1; pointer-events: auto; transition-delay: .2s;
}
@media (prefers-reduced-motion: reduce) {
  .tooltip { transition-duration: 0s; }
}
.feedback {
  position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip-path: inset(50%); white-space: nowrap;
}
.feedback[data-visible="true"], .download:not([hidden]) {
  position: absolute; z-index: 10; top: 50px; right: 0;
  width: max-content; max-width: min(260px, calc(100vw - 48px)); height: auto;
  box-sizing: border-box; padding: 8px 12px; border: 1px solid var(--ui-border);
  border-radius: 8px; background: var(--ui-surface); color: var(--ui-text);
  font: 13px/1.5 sans-serif; white-space: normal; clip-path: none;
  box-shadow: 0 4px 16px var(--ui-shadow);
}
.download:not([hidden]) { color: var(--ui-accent); }
@media print { .page-actions { display: none; } }
"""


def pdf_controls(downloads):
    controls = []
    for index, document in enumerate(downloads or [{"data": None}]):
        label = "Download job description (PDF)"
        if len(downloads) > 1:
            label += ": " + document["filename"]
        tooltip = label
        if document["data"] is not None:
            data = b64encode(document["data"]).decode("ascii")
            filename = html.escape(document["filename"], quote=True)
            opening = (
                f'<a class="icon-button" data-action="pdf" '
                f'href="data:application/pdf;base64,{data}" download="{filename}" '
            )
            closing = "</a>"
        else:
            opening = '<button type="button" disabled '
            closing = "</button>"
            tooltip = "The job description PDF is temporarily unavailable."
        controls.append(
            f'<div class="action-control">{opening}'
            f'aria-label="{html.escape(label, quote=True)}" aria-describedby="pdf-tooltip-{index}">'
            '<svg viewBox="0 0 24 24" aria-hidden="true">'
            '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>'
            f'</svg>{closing}<span class="tooltip" id="pdf-tooltip-{index}" role="tooltip">'
            f"{html.escape(tooltip)}</span></div>"
        )
    return "".join(controls)


def render_page_actions(downloads):
    component = st.components.v2.component(
        "recruitment_page_actions",
        html=HTML,
        css=CSS,
        js=Path(__file__).with_name("page_actions.js").read_text(),
    )
    width = 52 * (max(1, len(downloads)) + 2) - 8
    with st.container(width=width, key="page_actions"):
        component(
            key="page_actions_controls", data={"pdf_controls": pdf_controls(downloads)}, width=width
        )
