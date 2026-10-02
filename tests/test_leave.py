import json
from datetime import date

import leave

TODAY = date(2026, 10, 2)  # a Friday, so "next Monday" is 2026-10-05


def scripted(monkeypatch, replies):
    calls = []
    answers = iter(replies)

    def fake_call_model(model, messages, temperature):
        calls.append(list(messages))
        return next(answers)

    monkeypatch.setattr(leave, "call_model", fake_call_model)
    return calls


def as_json(**fields):
    form = {"leave_type": None, "start_date": None, "end_date": None, "reason": None, "missing": []}
    form.update(fields)
    return json.dumps(form)


def test_valid_first_try(monkeypatch):
    scripted(monkeypatch, [as_json(leave_type="sick", start_date="2026-10-05",
                                   end_date="2026-10-07", reason="flu")])
    r = leave.extract("3 days of sick leave from next Monday because of the flu", "m", today=TODAY)
    assert r.status == "ok"
    assert r.attempts == 1
    assert r.first_try_valid
    assert r.request is not None
    assert r.request.working_days == 3


def test_sunday_is_fixed_on_retry(monkeypatch):
    calls = scripted(monkeypatch, [
        as_json(leave_type="sick", start_date="2026-10-04", end_date="2026-10-07", reason="flu"),
        as_json(leave_type="sick", start_date="2026-10-05", end_date="2026-10-07", reason="flu"),
    ])
    r = leave.extract("3 days of sick leave from next Monday because of the flu", "m", today=TODAY)
    assert r.status == "ok"
    assert r.attempts == 2
    assert not r.first_try_valid
    assert "Sunday" in calls[1][-1]["content"]


def test_missing_info_asks_for_clarification(monkeypatch):
    calls = scripted(monkeypatch, [as_json(missing=["leave_type", "dates"])])
    r = leave.extract("Can I take some leave soon?", "m", today=TODAY)
    assert r.status == "needs_clarification"
    assert r.attempts == 1
    assert len(calls) == 1


def test_invented_reason_fails_after_one_retry(monkeypatch):
    invented = as_json(leave_type="sick", start_date="2026-10-05", end_date="2026-10-06", reason="Flu")
    calls = scripted(monkeypatch, [invented, invented])
    r = leave.extract("I need two days off from Monday", "m", today=TODAY)
    assert r.status == "failed"
    assert r.attempts == 2
    assert len(calls) == 2
    assert any("Flu" in p for p in r.problems)


def test_genuine_weekend_request_asks_the_employee(monkeypatch):
    saturday = as_json(leave_type="unpaid", start_date="2026-10-03",
                       end_date="2026-10-03", reason="personal reasons")
    scripted(monkeypatch, [saturday, saturday])
    r = leave.extract("I need tomorrow off, unpaid, for personal reasons.", "m", today=TODAY)
    assert r.status == "needs_clarification"
    assert "Saturday" in (r.message or "")


def test_broken_json_is_retried(monkeypatch):
    scripted(monkeypatch, [
        "this is not json",
        as_json(leave_type="annual", start_date="2026-10-20", end_date="2026-10-23"),
    ])
    r = leave.extract("Annual leave from 20 to 23 October please", "m", today=TODAY)
    assert r.status == "ok"
    assert r.attempts == 2


def test_working_days_skip_weekends():
    assert leave.working_days(date(2026, 10, 20), date(2026, 10, 24)) == 4