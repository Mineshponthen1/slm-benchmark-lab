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
    reason: str | None = Field(default=None, max_length=200)
    missing: list[Literal["leave_type", "dates"]] = []


class LeaveRequest(BaseModel):
    """The final, checked request. Working days are counted by code, not the model."""
    leave_type: LeaveType
    start_date: date
    end_date: date
    working_days: int
    reason: str | None = None


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
    if ex.reason and not reason_is_grounded(ex.reason, message):
        problems.append(f'reason "{ex.reason}" does not appear in the employee\'s message; use null if no reason is given')
    return problems


def weekend_problems(ex: LeaveExtraction) -> list[str]:
    out = []
    for name, d in (("start_date", ex.start_date), ("end_date", ex.end_date)):
        if d is not None and d.weekday() in WEEKEND:
            out.append(f"{name} {d.isoformat()} is a {d:%A}, which is a weekend day.")
    return out


def build_system_prompt(today: date) -> str:
    return (
        "You turn employee leave requests into JSON.\n"
        f"Today is {today:%A} {today.isoformat()}.\n"
        "Reply with ONLY a JSON object with these fields:\n"
        '- "leave_type": one of "annual", "sick", "maternity", "paternity", "unpaid", "other", or null if not stated\n'
        '- "start_date": first day of leave as YYYY-MM-DD, or null if not stated\n'
        '- "end_date": last day of leave as YYYY-MM-DD, or null if not stated\n'
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


def extract(message: str, model: str, temperature: float = 0, today: date | None = None) -> LeaveResult:
    today = today or date.today()
    messages = [
        {"role": "system", "content": build_system_prompt(today)},
        {"role": "user", "content": message},
    ]
    ex: LeaveExtraction | None = None
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
            problems = find_problems(ex, message) + weekend_problems(ex)
            if not problems:
                assert ex.leave_type and ex.start_date and ex.end_date
                request = LeaveRequest(
                    leave_type=ex.leave_type,
                    start_date=ex.start_date,
                    end_date=ex.end_date,
                    working_days=working_days(ex.start_date, ex.end_date),
                    reason=ex.reason,
                )
                return LeaveResult(status="ok", request=request,
                                   attempts=attempt, first_try_valid=(attempt == 1))

        if attempt == 1:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your JSON has these problems:\n- " + "\n- ".join(problems) + "\n"
                    "Fix them using only the employee's message. If the employee really asked for "
                    "these dates, keep them. Return the corrected JSON only."
                )},
            ]

    if ex is not None and not find_problems(ex, message) and weekend_problems(ex):
        return LeaveResult(status="needs_clarification",
                           message=" ".join(weekend_problems(ex)) + " Please choose working days (Monday to Friday).",
                           problems=problems, attempts=2, first_try_valid=False)
    return LeaveResult(status="failed", problems=problems, attempts=2, first_try_valid=False)