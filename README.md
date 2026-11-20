# Python Developer recruitment assistant

A Streamlit proof of concept for the single Python Developer assignment in
[TASK.md](TASK.md). Candidates provide background in chat, ask questions about the
provided PDF, arrange an interview, or opt out.

A LangGraph Main Agent selects `continue`, `schedule` or `end` and consults three
advisors: a fine-tuned ModernBERT Exit classifier, an Info Advisor grounded in
PostgreSQL/pgvector, and a Scheduling Advisor using SQL tools. Main, Info and
Scheduling use LangChain with OpenAI. Hugging Face is permitted for Exit, as
confirmed by the project owner.

## Install and run

Use Python 3.11+ (verified with 3.13), PostgreSQL with pgvector, an OpenAI API key
and the trained Exit runtime files. Install runtime dependencies explicitly:

```bash
git clone https://github.com/sapir-pirski/technion-genai-recruitment-assistant.git
cd technion-genai-recruitment-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` and `DATABASE_URL` in `.env`. Hosted PostgreSQL requires TLS
(`sslmode=require` or `verify-full`). Behavior defaults live in
[app/config.py](app/config.py); `.env` and Streamlit secrets can override them.

The default Exit provider is `huggingface`. Obtain the owner-supplied runtime
files `model.onnx`, `tokenizer.json` and `exit_classifier.json` and place them in
`artifacts/hf-exit/runtime/`, or configure the pinned Hub source below. The model
and benchmark are private inputs and are not included in a Git checkout.

```bash
python -m app.main doctor --verify-exit
# Offline preparation, separate from chat: extract and embed the assignment PDF.
python -m app.main ingest
python -m app.main index-status
python -m app.main calendar status --timezone Asia/Jerusalem
./run-full-project.sh
```

The launcher runs Streamlit in the foreground on port 8502. Stop with Ctrl+C.
Use `./run-full-project.sh --server.port 8503` for another port. On Windows, activate
`.venv\Scripts\activate` and run:

```bash
python -m streamlit run streamlit_app/streamlit_main.py --server.port 8502
```

Startup verifies the fine-tuned artifact before opening runtime storage. It does
not install packages, train models, embed PDFs or add interview availability.
PyTorch and Transformers are unnecessary for inference. A configured Hub source
may download the three pinned files before CPU inference starts.

## Use the app

Ask “Which skills are required?” to see a cited answer. The job description and
PDF download are available from the header. Supply experience directly in chat.
Ask to arrange an interview to see up to three available SQL appointments in the
browser's timezone. Select an appointment, review it and explicitly confirm it.
A confirmed booking is retained across retries. Opting out closes the conversation.
“New conversation” starts another thread while preserving recorded messages and
confirmed bookings.

Recruiter availability is an explicit operator input. Inspect it with `calendar
status`; use an approved future date range with `calendar add`. Preview before
writing, for example with your own approved values:

```bash
python -m app.main calendar add --from-date YYYY-MM-DD --to-date YYYY-MM-DD \
  --weekdays mon tue wed thu fri --start-time 09:00 --end-time 12:00 \
  --timezone Asia/Jerusalem --duration-minutes 30 --dry-run
```

Remove `--dry-run` only when those dates and hours are approved. Expired calendars
are not extended automatically. Synthetic test appointments stay in disposable
SQL databases; they do not establish real recruiter availability.

## Project structure

```text
app/
  main.py              CLI entry point
  config.py            Settings and model routing
  dataset.py           JSON benchmark validation and frozen splits
  agents/              Main, Exit, Info and Scheduling advisors; model backends
  workflow/            LangGraph orchestration and conversation service
  scheduling/          SQL calendar, date interpretation and confirmation
  rag/                 Assignment PDF ingestion, publication and retrieval
  storage/             Database setup, messages, exports and legacy copying
  fine_tuning/         ModernBERT training and ONNX export
  evaluation/          Action metrics, controlled replay and optional quality judge
  observability/       Sanitized logs and request-local usage traces
streamlit_app/         Candidate interface
config/               Generated assignment index summary
data/jobs/             Assignment PDF
data/evaluation/       Shareable metric and training snapshots
tests/                 Unit/integration tests and test_evals.ipynb
```

Runtime data lives in the isolated PostgreSQL `recruitment` schema: seven
application tables and four LangGraph checkpoint tables. The original `public`
tables remain intact. The benchmark loads from JSON, never from runtime SQL.
PDF replacement publishes a complete new index before retiring the prior version.
Only the assignment document is visible to retrieval and the candidate interface.

An explicit `copy-legacy --source-schema public --dry-run` command can preview
copying assignment conversations, bookings and checkpoints into an empty target.
It is optional and has not been run against the hosted database. It does not
copy the benchmark or other jobs. See [SECURITY.md](SECURITY.md) for data handling.

## Benchmark and evaluation

Place the authorized assignment JSON at `data/sms_conversations.json`. It contains
15 conversations, 103 messages and 59 labeled recruiter turns. Conversation IDs
remain in the original partitions:

| Split | Conversation IDs | Labeled turns |
|---|---|---:|
| Train | 1, 2, 3, 4, 5, 6, 8, 10, 14 | 36 |
| Validation | 7, 11, 13 | 12 |
| Test | 9, 12, 15 | 11 |

`DATASET_PATH` or `--dataset` selects another local file. Loading validates the
assignment split, identities, labels and timestamp order. Inputs contain only the
history before the labeled recruiter response. Contact columns, target labels and
future replies are excluded; phone/email and credential-shaped text are redacted.

The required [evaluation notebook](tests/test_evals.ipynb) reads committed,
text-free snapshots in [data/evaluation](data/evaluation/manifest.json), verifies
their hashes and recomputes metrics from per-example labels. It makes no API calls
or database connections and needs no weights. Viewing saved outputs needs no
private inputs; executing it requires the authorized JSON and development packages:

```bash
python -m pip install -r requirements-dev.txt
OPENAI_API_KEY= DATABASE_URL= HF_TOKEN= python -m app.main prepare-data --output artifacts/examiner/splits
python -m ipykernel install --sys-prefix --name recruitment --display-name "Recruitment (.venv)"
jupyter execute tests/test_evals.ipynb --kernel_name recruitment --timeout 120 --output ../artifacts/examiner/test_evals.ipynb
```

| Measurement | Test result | Evidence |
|---|---:|---|
| Historical Main, prompted Exit | 6/11 all turns; 3/8 candidate-response turns | 8 November saved run; previous workflow |
| Historical prompted Exit | 10/11 | 8 November saved run |
| Fine-tuned Exit | 9/11 test; 11/12 validation | Unchanged ONNX reproduced locally on 18 November |

The fine-tuned classifier did not improve on the historical prompted Exit score.
Its test confusion matrix is `[[6, 2], [0, 3]]` in `do_not_end`, `end` order: two
false `end` recommendations. Three test conversations are a small exploratory
sample. Openings and candidate-response decisions are reported separately, and
failed calls receive no accuracy credit.

The simplified Main workflow has not received a new held-out API measurement.
That run is prepared but needs explicit authorization to transmit the redacted
private dialogue to OpenAI. Historical metrics do not measure current Main
accuracy. Historical transcripts also lack a verified calendar and slot IDs, so
label agreement cannot establish booking correctness.

When authorized, run current measurements to separate output files:

```bash
python -m app.main evaluate --task actions --split test --exit-provider huggingface --output artifacts/examiner/actions_test_current.json
# Independent ONNX verification uses local JSON and weights, without API calls.
OPENAI_API_KEY= DATABASE_URL= HF_TOKEN= python -m app.main evaluate-exit-hf --split test --output artifacts/examiner/exit_test_current.json
OPENAI_API_KEY= DATABASE_URL= HF_TOKEN= python -m app.main evaluate-exit-hf --split validation --output artifacts/examiner/exit_validation_current.json
```

Controlled replay uses the fictitious [workflow cases](tests/fixtures/workflow_cases.json),
a disposable SQL calendar and the real model workflow. It checks citations,
relative dates, offers, explicit booking, retries and closure separately from
historical accuracy. The current run on **20 November 2026** passed **15/15 turns
across six cases**, using real OpenAI calls and the fine-tuned local Exit model:

```bash
python -m app.main evaluate --task workflow --exit-provider huggingface --input tests/fixtures/workflow_cases.json --output artifacts/workflow_evaluation.json
```

The optional quality judge accepts explicit evidence packets through
`evaluate --task quality --input tests/fixtures/quality_cases.json`. Model calls
and embeddings use the configured OpenAI account and incur API usage charges.

## Fine-tuning

ModernBERT-base was trained locally with seed 42, five epochs, learning rate
`2e-5`, batch size 4 and class weights derived from training labels. Validation
macro F1 selected epoch 4; test conversations were excluded from training,
checkpoint selection and quantization selection. Training provenance is in
[exit_training_summary.json](data/evaluation/exit_training_summary.json).

Training and inference share the same role/content input contract: at most 12
recent messages, 16,000 history characters and 2,048 tokens with left truncation.
The quantized ONNX export matches the selected checkpoint's validation predictions.
Loading checks the model/tokenizer hashes, label order and input contract.

```bash
python -m pip install -r requirements-training.txt
# Use a separate output so the preserved working model stays available.
python -m app.main fine-tune-hf --output artifacts/examiner/retrained-exit
```

`EXIT_PROVIDER=openai` explicitly selects the prompted baseline. It is not the
assignment's fine-tuned Exit implementation. To select newly trained weights,
set `HF_EXIT_LOCAL_PATH` and their model SHA-256 in `HF_EXIT_REVISION` and restart.

## Tests

The suite contains 135 tests, including native Streamlit behavior, SQL booking
concurrency, retries, schema isolation, PDF index repair and model integrity.
All 135 passed from a fresh source copy and newly installed environment on
20 November. The notebook executed all 13 code cells without errors; fresh
ONNX inference reproduced 9/11 test predictions, and native Streamlit startup
selected the fine-tuned provider without model API calls.
Model calls in automated tests are mocked; test pass counts do not establish
model accuracy. Use a disposable PostgreSQL/pgvector server whose role can create
temporary test databases:

```bash
python -m pip install -r requirements-dev.txt
TEST_DATABASE_URL=postgresql://USER:PASSWORD@HOST/TEST_DB python -m pytest -q tests
python -m ruff check app streamlit_app tests
python -m ruff format --check app streamlit_app tests
```

## Community Cloud release

[TASK.md](TASK.md) requires Streamlit Community Cloud and a fine-tuned Exit
Advisor. The existing [hosted app](https://technion-genai-recruitment-assistant.streamlit.app)
is connected to GitHub `main`. This simplified release has not been deployed or
verified there. Its fine-tuned artifact publication remains pending; the existing
prompted deployment is not evidence of the required fine-tuned runtime.

Publish only the three verified runtime files to the private model repository,
then configure Streamlit **Settings → Secrets** with these fields plus the existing
OpenAI and database credentials:

```toml
EXIT_PROVIDER = "huggingface"
HF_EXIT_REPO_ID = "sapir-pirski/technion-recruitment-exit-8bfb0c30"
HF_EXIT_REPO_REVISION = "<immutable 40-character model repository commit>"
HF_EXIT_REVISION = "8bfb0c30560f6069412ff19210678ce1ca6f5c8b08b5db3fb92375c9ff631582"
HF_TOKEN = "<read token for this private repository>"
DATABASE_SCHEMA = "recruitment"
```

The [pinned download](https://huggingface.co/docs/huggingface_hub/guides/download)
fetches only weights, tokenizer and checksum manifest into the ignored cache.
Inference runs inside the app on CPU; candidate messages are not sent to Hugging
Face. Missing or corrupt files cause an error rather than a prompted fallback.

Use `streamlit_app/streamlit_main.py`, Python 3.13 and `requirements.txt`. Prepare
the hosted PDF index explicitly using the same database settings before release;
add only approved future availability. After deploying the source and secrets,
verify the fine-tuned provider/artifact and exercise sourced answers, opt-out,
three available appointments, explicit confirmation and conversation recovery.
Do not publish private benchmark JSON, model weights, credentials or live exports
to the source repository. Cloud secrets belong in its
[secrets manager](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Assignment status

The Main/three-advisor architecture, fine-tuned local Exit, offline PDF embeddings
in PostgreSQL/pgvector, SQL scheduling, Streamlit UI and executed evaluation notebook
are implemented. The preserved working checkpoint is tagged
`baseline-single-job-json`, with a verified private backup under
`artifacts/baselines/2026-11-17-baseline-single-job-json/`.

On 20 November, the hosted `recruitment` schema was initialized with all 11 tables
and four assignment PDF chunks. All 22 legacy `public` tables retained their
contents. No candidate histories or bookings were copied. The approved extension adds 480
appointments from 22 November through 31 December 2026: Sunday–Thursday,
09:00–17:00 Asia/Jerusalem, in 30-minute slots. These are in the new schema and
will become available to the new Cloud release when deployed.
The owner approved extending the existing availability pattern through year-end:
the new calendar covers 22 November–31 December, Sunday–Thursday, 09:00–17:00
Asia/Jerusalem, with 30-minute appointments. The currently deployed older app
continues using its legacy schema until this release is deployed.

Remaining release evidence: publish the private model artifact, measure current
Main on the authorized benchmark, and configure and verify the fine-tuned Cloud
runtime. These are recorded
as pending rather than inferred from older deployment or model scores.
