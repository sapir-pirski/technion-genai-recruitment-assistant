# Data handling

The application is an assignment proof of concept. Its Streamlit interface has no
application-level user authentication; run the demo with fictitious candidate
messages and sample availability unless access and retention are established for
real use.

Credentials belong in the ignored `.env` or `.streamlit/secrets.toml` locally and
Streamlit's secrets manager in Community Cloud. Hosted PostgreSQL connections
require TLS. Use a model-repository read token for deployment.

## Benchmark and model artifacts

`data/sms_conversations.json` contains private dialogue and may include candidate
and recruiter contact metadata. Keep it, generated splits, backups and training
artifacts outside version control. Evaluation and training read local JSON, not
runtime database tables. Contact columns and gold labels are excluded from model
inputs; phone/email and credential-shaped patterns are redacted. Remaining dialogue
can still be private, so API evaluation requires authorization to transmit it.

The fine-tuned ONNX weights derive from that benchmark. Private model publication
includes only weights, tokenizer and checksum manifest; it excludes raw dialogue,
contacts and credentials. Hub downloads are pinned to an immutable commit and
validated before inference. Candidate text stays in the application for Exit
inference; Main/Info/Scheduling and embeddings use the configured OpenAI API.

`data/evaluation/` contains shareable metrics, per-example labels, numeric example
IDs and training provenance. It excludes dialogue, generated replies and contact
values. The notebook displays these snapshots without network requests.

## Runtime storage

The isolated `recruitment` PostgreSQL schema holds the assignment PDF/index,
approved appointments, bookings, conversation messages and LangGraph checkpoints.
The previous `public` tables remain intact. Optional legacy copying must be invoked
explicitly and requires an empty target; it preserves source rows.

Conversation records, checkpoints and `artifacts/conversations.json` may contain
private messages. The export is Git-ignored and written with owner-only permissions.
Restrict access to database credentials, backups and local artifacts, and set an
appropriate retention policy before handling real candidates.

Technical logs and request-local traces omit raw prompts, message contents, secret
values and raw exception text. They retain identifiers, timing, decisions, usage
and sanitized error codes. PDF ingestion accepts only the assignment PDF, checks
size and extracted page content, and publishes a complete index before replacing
the active version.
