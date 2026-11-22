"""Background inputs are captured with the next real conversation message."""

import streamlit as st

from app.profiles import ProfileReadError, content_hash, linkedin_reference, safe_filename


def matching_errors(profile, kind, value, filename=None):
    if not value:
        return []
    sha = content_hash(value)
    return [
        source
        for source in profile.get("sources", [])
        if (
            source["source_kind"] == kind
            and source["status"] == "failed"
            and source.get("content_sha256") == sha
            and (filename is None or source.get("filename") == filename)
        )
    ]


def source_errors(profile, kind, value, filename=None):
    for source in matching_errors(profile, kind, value, filename):
        st.error(source["error_message"])


def clear_introduction_inputs():
    for key in list(st.session_state):
        if key.startswith("introduction_"):
            st.session_state.pop(key, None)


def context_snapshot():
    upload = st.session_state.get("introduction_file")
    return {
        "introduction": st.session_state.get("introduction_text", ""),
        "linkedin_url": st.session_state.get("introduction_linkedin_url", ""),
        "file_name": upload.name if upload is not None else None,
        "file_data": upload.getvalue() if upload is not None else None,
    }


def render_introduction(busy, profile=None):
    profile = profile or {}
    values = context_snapshot()
    failures = (
        matching_errors(profile, "introduction", values["introduction"].strip())
        or matching_errors(profile, "linkedin_reference", values["linkedin_url"].strip())
        or matching_errors(
            profile, "file", values["file_data"], safe_filename((values["file_name"] or "").strip())
        )
    )
    with st.expander(
        "Introduce yourself", expanded=bool(failures), icon=":material/person_outline:"
    ):
        introduction, linkedin, upload = st.tabs(
            ["Introduction", "LinkedIn", "Upload file"], key="introduction_tabs"
        )
        with introduction:
            st.text_area(
                "Tell us about yourself",
                key="introduction_text",
                max_chars=4000,
                placeholder="Share your Python experience, relevant skills and projects.",
                height=140,
                disabled=busy,
            )
            source_errors(
                profile, "introduction", st.session_state.get("introduction_text", "").strip()
            )
        with linkedin:
            st.text_input(
                "LinkedIn reference URL",
                key="introduction_linkedin_url",
                max_chars=500,
                placeholder="https://www.linkedin.com/in/your-name/",
                help="Saved as a reference. The assistant does not fetch or read the profile page. Add background in Introduction or Upload file.",
                disabled=busy,
            )
            value = st.session_state.get("introduction_linkedin_url", "").strip()
            try:
                linkedin_reference(value)
            except ProfileReadError as error:
                st.error(str(error))
            else:
                source_errors(profile, "linkedin_reference", value)
        with upload:
            st.file_uploader(
                "Upload your CV or profile",
                type=["pdf", "doc", "docx", "txt"],
                key="introduction_file",
                max_upload_size=5,
                disabled=busy,
                help="Up to 5 MB. PDFs: at most 10 pages; all documents: at most 24,000 characters. Scans require OCR.",
            )
            if upload := st.session_state.get("introduction_file"):
                source_errors(
                    profile, "file", upload.getvalue(), safe_filename(upload.name.strip())
                )


def render_profile_status(state):
    profile = state.get("candidate_profile", {})
    sources = profile.get("sources", [])
    if not sources:
        return
    labels = {
        "introduction": "Introduction",
        "linkedin_reference": "LinkedIn link",
        "file": "Uploaded document",
    }
    used = [
        source.get("filename") or labels.get(source["source_kind"], "Profile text")
        for source in sources
        if source["status"] == "ready"
    ]
    if used:
        st.caption("Background used: " + " · ".join(used))
    if any(source["status"] == "reference" for source in sources):
        st.caption("LinkedIn URL saved as a reference; its page has not been fetched.")
    if profile.get("conflicts"):
        st.caption(
            "Some supplied profile details differ. The assistant can help clarify them in chat."
        )
