from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Action = Literal["continue", "schedule", "end"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Plan(StrictModel):
    next_advisor: Literal["exit", "scheduling", "info", "respond"]
    stop_requested: bool
    closure_reason: Literal["opt_out", "scheduled"] | None
    declined_slot_ids: list[int]
    info_question: str | None
    scheduling_needed: bool
    next_question: str


class InfoDecision(StrictModel):
    decision: Literal["info_needed", "info_not_needed"]
    question: str | None

    @model_validator(mode="after")
    def validate_question(self):
        if self.decision == "info_needed" and (not self.question or not self.question.strip()):
            raise ValueError("Information retrieval needs a question")
        if self.decision == "info_not_needed" and self.question is not None:
            raise ValueError("An unnecessary information request must not contain a question")
        return self


class ScheduleRequest(StrictModel):
    """Extract a request; availability and dates are validated by application code."""

    date_text: str | None = Field(description="Exact date phrase, e.g. next Friday or 2026-11-06")
    time_text: str | None = Field(description="Explicit time including AM/PM or 24-hour time")
    period: Literal["morning", "afternoon"] | None
    selected_slot_id: int | None = Field(
        description="Only an ID in offered_slots explicitly accepted"
    )
    accepts_slot: bool = Field(
        description="True only for an unambiguous acceptance, never a question"
    )


class InfoAnswer(StrictModel):
    answer: str
    source_ids: list[str]
    supported: bool


QualityDimension = Literal["grounding", "completeness", "relevance_clarity", "followup_quality"]
QualityScore = Literal[1, 2, 3, 4, 5]


class JudgeRating(StrictModel):
    reason: str
    evidence_refs: list[str]
    status: Literal["scored", "not_assessable"]
    score: QualityScore | None

    @field_validator("score", mode="before")
    @classmethod
    def integer_score(cls, value):
        if value is not None and type(value) is not int:
            raise ValueError("Scores must be integers or null")
        return value

    @model_validator(mode="after")
    def consistent_rating(self):
        if (self.status == "scored") != (self.score is not None):
            raise ValueError("Scored ratings need a score; unassessable ratings need null")
        if not self.reason.strip() or len(self.reason) > 600:
            raise ValueError("A rating needs a brief reason")
        if len(self.evidence_refs) > 20:
            raise ValueError("Too many evidence references")
        return self


class JudgeIssue(StrictModel):
    dimension: QualityDimension
    description: str
    evidence_refs: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def brief_issue(self):
        if (
            not self.description.strip()
            or len(self.description) > 600
            or not 1 <= len(self.evidence_refs) <= 20
        ):
            raise ValueError("An issue needs a brief description and bounded references")
        return self


class JudgeResult(StrictModel):
    grounding: JudgeRating
    completeness: JudgeRating
    relevance_clarity: JudgeRating
    followup_quality: JudgeRating
    issues: list[JudgeIssue]

    @field_validator("issues")
    @classmethod
    def bounded_issues(cls, value):
        if len(value) > 8:
            raise ValueError("At most eight issues may be reported")
        return value


class TurnResult(BaseModel):
    action: Action
    response: str
    conversation_status: Literal["active", "closed"] = "active"
    closure_reason: Literal["opt_out", "scheduled"] | None = None
    booking: dict | None = None
    offered_slots: list[dict] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    advisors: list[str] = Field(default_factory=list)
    error: str | None = None
