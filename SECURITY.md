# Security policy

## Project status

This project is a recruitment chatbot proof of concept. It has no authentication
or cross-device conversation recovery and is not intended to handle real candidate
data on a public deployment without additional access controls and a security review.
Security fixes target the current development version; there is no supported
release or guaranteed response timeline yet.

## Reporting a vulnerability

Use the repository's private vulnerability reporting feature if it is enabled.
Otherwise, contact the maintainer through an established private channel and ask
for a secure way to share the report. A dedicated security contact has not yet
been published. Do not post exploit details, credentials, or candidate data in
public issues or pull requests.

Include the affected version or commit, the impact, steps to reproduce, and a
minimal example using synthetic data. Remove secrets and personal information
from logs and screenshots. Coordinate public disclosure with the maintainer
after a fix is available.

## Secrets and candidate data

- Keep API keys and database credentials in local environment variables, `.env`,
  or deployment secrets. Never commit them or `.streamlit/secrets.toml`.
- Keep chat histories, databases, generated datasets, and evaluation artifacts
  out of source control. Use synthetic data for isolated testing.
- Rotate exposed credentials immediately and remove them from repository history
  and shared artifacts where possible.
- Model and embedding requests send relevant content to OpenAI.
  Review what data may be shared before using real candidate information.
- PostgreSQL conversation tables and `artifacts/conversations.json` contain private
  message text. The export is ignored by Git and written with owner-only permissions.
  Restrict database/filesystem access and remove records according to your retention policy.
- Benchmark tables preserve the supplied conversation text and source contact metadata.
  Keep their backups private. Data preparation excludes contact columns and redacts
  phone/email patterns from model inputs; benchmark labels stay separate from runtime predictions.
- Technical logs and trace attributes contain identifiers, timing, usage, and
  sanitized error codes. Raw prompts, exception text, message contents, and secrets
  are excluded from technical diagnostics. Stored model actions are predictions;
  generated conversation labels remain NULL until separately verified.

## Deployment and dependencies

- Restrict access to the application and databases; use encrypted connections
  when accessing remote services.
- Use a database account with only the permissions the application needs.
- Review dependency updates and run the documented checks before deploying.
- Treat retrieved documents and candidate messages as untrusted input. Preserve
  validated scheduling tools and explicit booking confirmation checks.

## External job documents

The ingestion worker fetches only explicitly reviewed employer/recruiter URLs
and configured public APIs with pinned IDs. Automatic job discovery is disabled. Job content is
untrusted data, not agent instructions. Generated PDFs escape source markup and
use bundled licensed fonts; unsupported glyphs or unreadable pages fail explicitly.
PDF size/page-stream limits reduce accidental resource exhaustion but do not
constitute a hostile-document sandbox. Review source terms before redistribution.

Keep the worker and frontend on the same durable PostgreSQL database with pgvector.
The catalog publication filter prevents partial versions and unrelated jobs from
entering a role's retrieval results. Imported jobs are reference-only and cannot
use the assignment's sample interview slots. Scheduler reports/logs can contain
public job titles and source URLs; keep them separate from candidate data.
