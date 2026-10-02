import re
from datetime import date, timedelta
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

LeaveType = Literal["annual", "sick", "maternity", "paternity", "unpaid", "other"]
WEEKEND = {5, 6}  # Python counts Monday as 0, so Saturday = 5 and Sunday = 6
MISSING_WORDS = {"leave_type": "what type of leave you need", "dates": "which dates you need"}


class LeaveExtraction(BaseModel):
    """What the model must return."""
    leave_type: LeaveType | None = None
    start_date: date | None = None
    end_date: date | None = None
    days_requested: int | None = Field(default=None, ge=1, le=60)
    reason: str | None = Field(default=None, max_length=200)
    missing: list[Literal["leave_type", "dates"]] = []


class LeaveRequest(BaseModel):
    """The final, checked request. Weekends and working days are handled by code."""
    leave_type: LeaveType
    start_date: date
    end_date: date
    working_days: int
    reason: str | None = None
    notes: list[str] = []


class LeaveResult(BaseModel):
    status: Literal["ok", "needs_clarification", "failed"]
    request: LeaveRequest | None = None
    message: str | None = None
    problems: list[str] = []
    attempts: int
    first_try_valid: bool


def working_days(start: date, end: date) -> int:
    count = 0
    day = start
    while day <= end:
        if day.weekday() not in WEEKEND:
            count += 1
        day += timedelta(days=1)
    return count


def trim_weekends(start: date, end: date) -> tuple[date, date]:
    while start <= end and start.weekday() in WEEKEND:
        start += timedelta(days=1)
    while end >= start and end.weekday() in WEEKEND:
        end -= timedelta(days=1)
    return start, end


def reason_is_grounded(reason: str, message: str) -> bool:
    words = [w for w in re.findall(r"[a-z]+", reason.lower()) if len(w) > 2]
    text = message.lower()
    return all(w in text for w in words)


def find_problems(ex: LeaveExtraction, message: str) -> list[str]:
    problems = []
    if ex.leave_type is None:
        problems.append('leave_type is null but "missing" does not include "leave_type"')
    if ex.start_date is None or ex.end_date is None:
        problems.append('start_date or end_date is null but "missing" does not include "dates"')
    elif ex.end_date < ex.start_date:
        problems.append("end_date is before start_date")
    else:
        counted = working_days(ex.start_date, ex.end_date)
        if ex.days_requested is not None and counted > 0 and counted != ex.days_requested:
            problems.append(
                f"the employee asked for {ex.days_requested} days, but "
                f"{ex.start_date.isoformat()} to {ex.end_date.isoformat()} contains "
                f"{counted} working days (Saturdays and Sundays are not counted)"
            )
    if ex.reason and not reason_is_grounded(ex.reason, message):
        problems.append(f'reason "{ex.reason}" does not appear in the employee\'s message; use null if no reason is given')
    return problems


def build_system_prompt(today: date) -> str:
    return (
        "You turn employee leave requests into JSON.\n"
        f"Today is {today:%A} {today.isoformat()}.\n"
        "Reply with ONLY a JSON object with these fields:\n"
        '- "leave_type": one of "annual", "sick", "maternity", "paternity", "unpaid", "other", or null if not stated\n'
        '- "start_date": first day of leave as YYYY-MM-DD, or null if not stated\n'
        '- "end_date": last day of leave as YYYY-MM-DD, or null if not stated\n'
        '- "days_requested": the number of days if the employee states one (for example 3 for "3 days"), otherwise null\n'
        '- "reason": the reason in the employee\'s own words, or null if none is given\n'
        '- "missing": a list of what is missing: "leave_type" and/or "dates". Use [] if nothing is missing\n'
        "Never guess or invent information that is not in the message.\n"
        "No explanations and no extra text."
    )


def call_model(model: str, messages: list, temperature: float) -> str:
    r = ollama.chat(
        model=model,
        messages=messages,
        format="json",
        options={"temperature": temperature, "num_predict": 200},
    )
    return r["message"]["content"]


def finish(ex: LeaveExtraction, attempt: int) -> LeaveResult:
    assert ex.leave_type and ex.start_date and ex.end_date
    counted = working_days(ex.start_date, ex.end_date)
    if counted == 0:
        if ex.start_date == ex.end_date:
            text = f"{ex.start_date:%A %d %B %Y} is a weekend day."
        else:
            text = f"{ex.start_date:%d %B} to {ex.end_date:%d %B %Y} has no working days."
        return LeaveResult(status="needs_clarification",
                           message=text + " Please choose working days (Monday to Friday).",
                           attempts=attempt, first_try_valid=(attempt == 1))

    start, end = trim_weekends(ex.start_date, ex.end_date)
    notes = []
    if start != ex.start_date:
        notes.append(f"{ex.start_date:%A %d %B} is a weekend day, so the leave starts on {start:%A %d %B}.")
    if end != ex.end_date:
        notes.append(f"{ex.end_date:%A %d %B} is a weekend day, so the leave ends on {end:%A %d %B}.")

    request = LeaveRequest(leave_type=ex.leave_type, start_date=start, end_date=end,
                           working_days=counted, reason=ex.reason, notes=notes)
    return LeaveResult(status="ok", request=request,
                       attempts=attempt, first_try_valid=(attempt == 1))


def extract(message: str, model: str, temperature: float = 0, today: date | None = None) -> LeaveResult:
    today = today or date.today()
    messages = [
        {"role": "system", "content": build_system_prompt(today)},
        {"role": "user", "content": message},
    ]
    problems: list[str] = []

    for attempt in (1, 2):
        raw = call_model(model, messages, temperature)
        try:
            ex = LeaveExtraction.model_validate_json(raw)
        except ValidationError as e:
            ex = None
            problems = [
                f"{'.'.join(str(p) for p in err['loc']) or 'response'}: {err['msg']}"
                for err in e.errors()
            ]

        if ex is not None:
            if ex.missing:
                needed = " and ".join(MISSING_WORDS[m] for m in ex.missing)
                return LeaveResult(status="needs_clarification", message=f"Please tell me {needed}.",
                                   attempts=attempt, first_try_valid=(attempt == 1))
            problems = find_problems(ex, message)
            if not problems:
                return finish(ex, attempt)

        if attempt == 1:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your JSON has these problems:\n- " + "\n- ".join(problems) + "\n"
                    "Change only what is needed to fix them, using only the employee's message. "
                    "Return the corrected JSON only."
                )},
            ]

    return LeaveResult(status="failed", problems=problems, attempts=2, first_try_valid=False)