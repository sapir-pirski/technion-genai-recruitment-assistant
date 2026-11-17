from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.ingestion.files import ORIGINAL_PDF_NAME, resolve_pdf
from app.ingestion.models import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID
from app.main import main
from app.rag.ingest import ingest_pdf
from app.rag.retriever import JobRetriever


def test_only_named_assignment_pdf_is_accepted(tmp_path):
    settings = SimpleNamespace(job_pdf_path=tmp_path)
    assignment = tmp_path / ORIGINAL_PDF_NAME
    assignment.write_bytes(b"assignment")
    assert resolve_pdf(settings, assignment) == assignment
    other = tmp_path / "another-role.pdf"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="Only data/jobs/"):
        ingest_pdf(other, settings)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="Only data/jobs/"):
        resolve_pdf(settings, outside / ORIGINAL_PDF_NAME)
    assignment.unlink()
    assignment.symlink_to(other)
    with pytest.raises(ValueError, match="Only data/jobs/"):
        resolve_pdf(settings, assignment)


@pytest.mark.parametrize("changes", [{"job_id": "reference"}, {"document_id": "another-pdf"}])
def test_other_job_identities_are_rejected_before_database_or_api_access(changes):
    with pytest.raises(ValueError, match="Only the assignment"):
        ingest_pdf("unused.pdf", SimpleNamespace(), **changes)


@pytest.mark.parametrize(
    "arguments",
    [["sync-jobs"], ["export-corpus"], ["chat", "--job-id", "reference"]],
)
def test_removed_collection_commands_and_other_roles_are_rejected(arguments, monkeypatch):
    monkeypatch.setattr("sys.argv", ["recruitment", *arguments])
    with patch("app.main.Settings", side_effect=AssertionError("No credentials needed")):
        with pytest.raises(SystemExit) as result:
            main()
    assert result.value.code == 2


def test_saved_citations_exclude_other_documents_attached_to_the_assignment_role():
    retriever = JobRetriever.__new__(JobRetriever)
    retriever.store = Mock()
    retriever.store.get.return_value = {
        "ids": ["assignment-source", "old-extra-pdf"],
        "metadatas": [
            {"document_id": ASSIGNMENT_DOCUMENT_ID, "page": 1},
            {"document_id": "extra-document", "page": 2},
        ],
    }
    references = retriever.references(["assignment-source", "old-extra-pdf"], DEFAULT_JOB_ID)
    assert [item["id"] for item in references] == ["assignment-source"]
    assert retriever.references(["old-extra-pdf"], "reference") == []
    assert retriever.store.get.call_count == 1
