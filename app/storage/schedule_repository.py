from bisect import bisect_left
from datetime import UTC, datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import Time, and_, cast, extract, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ingestion.models import DEFAULT_JOB_ID, Job
from app.observability.logging import log_event
from app.observability.tracing import span
from app.storage.models import Base, Booking, Slot


class BookingConflict(ValueError):
    pass


def utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("An explicit timezone is required.")
    return value.astimezone(UTC).replace(tzinfo=None)


def slot_data(slot: Slot, timezone: str) -> dict:
    instant = slot.starts_at.replace(tzinfo=UTC)
    local = instant.astimezone(ZoneInfo(timezone))
    result = {
        "id": slot.id,
        "job_id": slot.job_id,
        "starts_at": instant.isoformat(),
        "local_time": local.isoformat(),
        "label": local.strftime("%A, %d %B %Y at %H:%M %Z"),
        "position": slot.position,
    }
    if slot.duration_minutes is not None:
        result.update(
            duration_minutes=slot.duration_minutes,
            ends_at=(instant + timedelta(minutes=slot.duration_minutes)).isoformat(),
        )
        result["label"] += f" ({slot.duration_minutes} minutes)"
    return result


class ScheduleRepository:
    def __init__(self, engine):
        self.engine = engine

    def initialize(self):
        Base.metadata.create_all(self.engine)

    def calendar_status(self, job_id, timezone, now):
        ZoneInfo(timezone)
        instant = utc_naive(now)
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError("Unknown job; inspect list-jobs")
            total, future, available, expired = session.execute(
                select(
                    func.count(Slot.id),
                    func.count(Slot.id).filter(Slot.starts_at > instant),
                    func.count(Slot.id).filter(Slot.starts_at > instant, Slot.available.is_(True)),
                    func.count(Slot.id).filter(Slot.starts_at <= instant, Slot.available.is_(True)),
                ).where(Slot.job_id == job_id, Slot.position == "Python Dev")
            ).one()
            latest = session.scalar(
                select(Slot)
                .where(Slot.job_id == job_id, Slot.position == "Python Dev")
                .order_by(Slot.starts_at.desc(), Slot.id.desc())
                .limit(1)
            )
            last = slot_data(latest, timezone) if latest else None
            last_date = datetime.fromisoformat(last["local_time"]).date() if last else None
            status = (
                "disabled"
                if not job.scheduling_enabled or job.status != "active"
                else "empty"
                if not total
                else "expired"
                if not future
                else "fully_booked"
                if not available
                else "available"
            )
            result = {
                "job_id": job_id,
                "timezone": timezone,
                "checked_at": now.isoformat(),
                "status": status,
                "slots_total": total,
                "future_available": available,
                "past_unbooked": expired,
                "bookings_total": session.scalar(
                    select(func.count()).select_from(Booking).where(Booking.job_id == job_id)
                ),
                "last_configured_date": last_date.isoformat() if last_date else None,
                "days_until_last_date": (last_date - now.astimezone(ZoneInfo(timezone)).date()).days
                if last_date
                else None,
                "last_appointment": last,
                "notice": "Extend availability only with recruiter-approved dates and hours."
                if last
                else "No availability is configured for this job.",
            }
        result["next_slots"] = (
            self.available(now + timedelta(microseconds=1), None, timezone, job_id=job_id)
            if status == "available"
            else []
        )
        return result

    def add_availability(self, starts, duration_minutes, timezone, *, job_id, now, dry_run=False):
        ZoneInfo(timezone)
        instants = sorted(utc_naive(start) for start in starts)
        if not instants or len(instants) > 10000:
            raise ValueError("Provide between 1 and 10000 appointments")
        if not 1 <= duration_minutes <= 480:
            raise ValueError("Appointment duration must be between 1 and 480 minutes")
        duration = timedelta(minutes=duration_minutes)
        if instants[0] <= utc_naive(now):
            raise ValueError("Every appointment must start in the future")
        if any(
            right < left + duration for left, right in zip(instants, instants[1:], strict=False)
        ):
            raise ValueError("Requested appointments duplicate or overlap each other")
        with span(
            "calendar",
            "add_availability",
            attributes={
                "job_id": job_id,
                "timezone": timezone,
                "duration_minutes": duration_minutes,
                "requested_slot_count": len(instants),
                "dry_run": dry_run,
                "first_start": instants[0].replace(tzinfo=UTC).isoformat(),
                "last_start": instants[-1].replace(tzinfo=UTC).isoformat(),
            },
        ) as item:
            with Session(self.engine) as session, session.begin():
                # Serialize calendar writers for this job, including duplicate checks.
                job = session.get(Job, job_id, with_for_update=True)
                if job is None or job.status != "active" or not job.scheduling_enabled:
                    raise ValueError("Choose an active job with scheduling enabled")
                existing = list(
                    session.scalars(
                        select(Slot).where(
                            Slot.job_id == job_id,
                            Slot.position == "Python Dev",
                            Slot.starts_at < instants[-1] + duration,
                            or_(
                                Slot.starts_at >= instants[0],
                                Slot.starts_at + Slot.duration_minutes * text("INTERVAL '1 minute'")
                                > instants[0],
                                and_(
                                    Slot.duration_minutes.is_(None), Slot.starts_at > utc_naive(now)
                                ),
                            ),
                        )
                    )
                )
                for slot in existing:
                    if slot.duration_minutes is None:
                        raise ValueError(
                            "An existing future appointment has unknown duration; resolve it before adding availability"
                        )
                    end = slot.starts_at + timedelta(minutes=slot.duration_minutes)
                    index = bisect_left(instants, end) - 1
                    if index >= 0 and slot.starts_at < instants[index] + duration:
                        raise ValueError(
                            "Requested availability duplicates or overlaps existing appointments"
                        )
                records = [
                    Slot(
                        job_id=job_id,
                        position="Python Dev",
                        starts_at=start,
                        duration_minutes=duration_minutes,
                        available=True,
                    )
                    for start in instants
                ]
                if not dry_run:
                    session.add_all(records)
                    session.flush()
                    item.attributes["added_slot_ids"] = [record.id for record in records]
            outcome = "validated" if dry_run else "added"
            item.attributes.update(outcome=outcome, slot_count=0 if dry_run else len(instants))
            log_event(
                "calendar_availability_changed",
                component="app.calendar",
                job_id=job_id,
                timezone=timezone,
                calendar_action=outcome,
                slot_count=item.attributes["slot_count"],
            )
            return {
                "job_id": job_id,
                "status": outcome,
                "slots_planned": len(instants),
                "slots_added": item.attributes["slot_count"],
                "timezone": timezone,
                "first_start": item.attributes["first_start"],
                "last_start": item.attributes["last_start"],
                "duration_minutes": duration_minutes,
            }

    def available(
        self,
        start: datetime,
        end: datetime | None,
        timezone: str,
        *,
        job_id: str = DEFAULT_JOB_ID,
        exact_time: time | None = None,
        period: str | None = None,
        limit: int = 3,
        exclude_ids: list[int] | None = None,
    ) -> list[dict]:
        ZoneInfo(timezone)
        query = self._available_query(start, job_id, exclude_ids)
        if end is not None:
            query = query.where(Slot.starts_at < utc_naive(end))
        local = func.timezone(timezone, func.timezone("UTC", Slot.starts_at))
        if exact_time is not None:
            query = query.where(cast(local, Time) == exact_time)
        if period not in (None, "morning", "afternoon"):
            raise ValueError("Choose morning or afternoon")
        if period == "morning":
            query = query.where(extract("hour", local) < 12)
        elif period == "afternoon":
            query = query.where(extract("hour", local) >= 12)
        return self._search(
            query.order_by(Slot.starts_at, Slot.id),
            timezone,
            limit,
            job_id=job_id,
            search_kind="matching",
            starts_at=start.isoformat(),
            ends_at=end.isoformat() if end else None,
            requested_time=exact_time.isoformat() if exact_time else None,
            period=period,
            excluded_slot_ids=exclude_ids or [],
        )

    @staticmethod
    def _available_query(start, job_id, exclude_ids):
        return select(Slot).where(
            Slot.position == "Python Dev",
            Slot.job_id == job_id,
            Slot.available.is_(True),
            Slot.starts_at >= utc_naive(start),
            Slot.id.not_in(exclude_ids or []),
        )

    def nearest(self, start, timezone, *, anchor, job_id=DEFAULT_JOB_ID, exclude_ids=None):
        """Select three future alternatives nearest the requested instant, then display by date."""
        ZoneInfo(timezone)
        distance = func.abs(extract("epoch", Slot.starts_at - utc_naive(anchor)))
        nearest_ids = (
            self._available_query(start, job_id, exclude_ids)
            .with_only_columns(Slot.id)
            .order_by(distance, Slot.starts_at, Slot.id)
            .limit(3)
        )
        query = select(Slot).where(Slot.id.in_(nearest_ids)).order_by(Slot.starts_at, Slot.id)
        return self._search(
            query,
            timezone,
            3,
            job_id=job_id,
            search_kind="alternatives",
            starts_at=start.isoformat(),
            requested_instant=anchor.isoformat(),
            excluded_slot_ids=exclude_ids or [],
        )

    def _search(self, query, timezone, limit, **attributes):
        if not 1 <= limit <= 100:
            raise ValueError("Slot limit must be between 1 and 100")
        with span("calendar", "search", attributes={"timezone": timezone, **attributes}) as item:
            with Session(self.engine) as session:
                results = [
                    slot_data(slot, timezone) for slot in session.scalars(query.limit(limit))
                ]
            item.attributes.update(
                outcome="offered" if results else "unavailable",
                offered_slot_ids=[slot["id"] for slot in results],
                slot_count=len(results),
            )
            log_event(
                "calendar_searched",
                component="app.calendar",
                job_id=attributes["job_id"],
                timezone=timezone,
                search_kind=attributes["search_kind"],
                slot_count=len(results),
                outcome=item.attributes["outcome"],
            )
            return results

    def get_booking(
        self, conversation_id: str, timezone: str, job_id=DEFAULT_JOB_ID
    ) -> dict | None:
        with Session(self.engine) as session:
            booking = session.scalar(
                select(Booking).where(
                    Booking.conversation_id == conversation_id, Booking.job_id == job_id
                )
            )
            if booking is None:
                return None
            return {
                "id": booking.id,
                "slot": slot_data(session.get(Slot, booking.slot_id), timezone),
            }

    def book(
        self,
        slot_id: int,
        conversation_id: str,
        request_id: str,
        now: datetime,
        timezone: str,
        *,
        job_id=DEFAULT_JOB_ID,
    ) -> dict:
        with span(
            "calendar",
            "book",
            attributes={"job_id": job_id, "slot_id": slot_id, "timezone": timezone},
        ) as item:
            try:
                result = self._book(slot_id, conversation_id, request_id, now, timezone, job_id)
            except BookingConflict:
                item.attributes["outcome"] = "conflict"
                log_event(
                    "calendar_booking_conflict",
                    component="app.calendar",
                    job_id=job_id,
                    slot_id=slot_id,
                )
                raise
            item.attributes.update(outcome="booked", booking_id=result["id"])
            log_event("calendar_booked", component="app.calendar", job_id=job_id, slot_id=slot_id)
            return result

    def _book(self, slot_id, conversation_id, request_id, now, timezone, job_id):
        existing = self.get_booking(conversation_id, timezone, job_id)
        if existing:
            if existing["slot"]["id"] == slot_id:
                return existing
            raise BookingConflict(
                "This conversation already has an interview. Rescheduling requires a recruiter."
            )
        try:
            with Session(self.engine) as session, session.begin():
                changed = session.execute(
                    update(Slot)
                    .where(
                        Slot.id == slot_id,
                        Slot.position == "Python Dev",
                        Slot.job_id == job_id,
                        Slot.available.is_(True),
                        Slot.starts_at > utc_naive(now),
                    )
                    .values(available=False)
                )
                if changed.rowcount != 1:
                    raise BookingConflict(
                        "That slot is no longer available. Please choose another time."
                    )
                session.add(
                    Booking(
                        id=str(uuid4()),
                        slot_id=slot_id,
                        job_id=job_id,
                        conversation_id=conversation_id,
                        request_id=request_id,
                        created_at=utc_naive(now),
                    )
                )
        except IntegrityError as exc:
            existing = self.get_booking(conversation_id, timezone, job_id)
            if existing and existing["slot"]["id"] == slot_id:
                return existing
            raise BookingConflict(
                "The booking changed while this request was processed. Please try again."
            ) from exc
        return self.get_booking(conversation_id, timezone, job_id)
