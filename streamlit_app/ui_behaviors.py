"""Small frontend enhancements for native upload controls and conversation scrolling."""

from pathlib import Path

import streamlit as st


def render_ui_behaviors(conversation_id, latest_request_id=None):
    component = st.components.v2.component(
        "recruitment_ui_behaviors",
        html='<button class="latest" type="button" hidden>Latest message ↓</button>',
        css="""
        .latest { position: fixed; z-index: 50; right: max(16px, calc((100vw - 960px) / 2));
          bottom: calc(var(--ui-composer-space, 100px) + var(--ui-keyboard-inset, 0px) + 8px);
          min-height: 44px; padding: 8px 14px; border: 1px solid var(--ui-input-border);
          border-radius: 24px; background: var(--ui-surface); color: var(--ui-accent);
          font: 14px/1.5 sans-serif; cursor: pointer; box-shadow: 0 2px 8px var(--ui-shadow); }
        .latest:hover { background: var(--ui-tint); }
        .latest:focus-visible { outline: 3px solid var(--ui-accent); outline-offset: 3px; }
        .latest[hidden] { display: none; }
        @media print { .latest { display: none; } }
        """,
        js=Path(__file__).with_name("ui_behaviors.js").read_text(),
    )
    component(
        key="ui_behaviors",
        data={"conversation_id": conversation_id, "latest_request_id": latest_request_id},
        height=0,
    )
