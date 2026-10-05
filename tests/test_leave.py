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
    form = {"leave_type": None, "start_date": None, "end_date": None,
            "days_requested": None, "reason": None, "missing": []}
    form.update(fields)
    return json.dumps(form)


def test_valid_first_try(monkeypatch):
    scripted(monkeypatch, [as_json(leave_type="sick", start_date="2026-10-05",
                                   end_date="2026-10-07", days_requested=3, reason="flu")])
    r = leave.extract("3 days of sick leave from next Monday because of the flu", "m", today=TODAY)
    assert r.status == "ok"
    assert r.attempts == 1
    assert r.first_try_valid
    assert r.request is not None
    assert r.request.working_days == 3
    assert r.request.notes == []


def test_weekend_start_is_moved_without_retry(monkeypatch):
    calls = scripted(monkeypatch, [as_json(leave_type="sick", start_date="2026-10-04",
                                           end_date="2026-10-07", days_requested=3, reason="flu")])
    r = leave.extract("3 days of sick leave from next Monday because of the flu", "m", today=TODAY)
    assert r.status == "ok"
    assert len(calls) == 1
    assert r.request is not None
    assert r.request.start_date == date(2026, 10, 5)
    assert r.request.working_days == 3
    assert "Sunday" in r.request.notes[0]


def test_days_mismatch_is_retried(monkeypatch):
    calls = scripted(monkeypatch, [
        as_json(leave_type="sick", start_date="2026-10-05", end_date="2026-10-08",
                days_requested=3, reason="flu"),
        as_json(leave_type="sick", start_date="2026-10-05", end_date="2026-10-07",
                days_requested=3, reason="flu"),
    ])
    r = leave.extract("3 days of sick leave from next Monday because of the flu", "m", today=TODAY)
    assert r.status == "ok"
    assert r.attempts == 2
    assert not r.first_try_valid
    assert "4 working days" in calls[1][-1]["content"]
    assert r.request is not None
    assert r.request.end_date == date(2026, 10, 7)


def test_unmentioned_days_number_is_ignored(monkeypatch):
    calls = scripted(monkeypatch, [as_json(leave_type="annual", start_date="2026-10-20",
                                           end_date="2026-10-24", days_requested=5,
                                           reason="family trip")])
    r = leave.extract("Please book my annual leave from 20 to 24 October for a family trip.",
                      "m", today=TODAY)
    assert r.status == "ok"
    assert len(calls) == 1
    assert r.request is not None
    assert r.request.working_days == 4


def test_range_ending_on_saturday_is_accepted(monkeypatch):
    scripted(monkeypatch, [as_json(leave_type="annual", start_date="2026-10-20",
                                   end_date="2026-10-24", reason="family trip")])
    r = leave.extract("Please book my annual leave from 20 to 24 October for a family trip.",
                      "m", today=TODAY)
    assert r.status == "ok"
    assert r.request is not None
    assert r.request.end_date == date(2026, 10, 23)
    assert r.request.working_days == 4
    assert "Saturday" in r.request.notes[0]


def test_saturday_only_request_asks_the_employee(monkeypatch):
    calls = scripted(monkeypatch, [as_json(leave_type="unpaid", start_date="2026-10-03",
                                           end_date="2026-10-03", days_requested=1,
                                           reason="personal reasons")])
    r = leave.extract("I need tomorrow off, unpaid, for personal reasons.", "m", today=TODAY)
    assert r.status == "needs_clarification"
    assert len(calls) == 1
    assert "Saturday" in (r.message or "")


def test_missing_info_asks_for_clarification(monkeypatch):
    calls = scripted(monkeypatch, [as_json(missing=["leave_type", "dates"])])
    r = leave.extract("Can I take some leave soon?", "m", today=TODAY)
    assert r.status == "needs_clarification"
    assert r.attempts == 1
    assert len(calls) == 1


def test_unstated_leave_type_asks_the_employee(monkeypatch):
    calls = scripted(monkeypatch, [as_json(leave_type="unpaid", start_date="2026-10-19",
                                           end_date="2026-10-20")])
    r = leave.extract("Can I have leave on 19 and 20 October?", "m", today=TODAY)
    assert r.status == "needs_clarification"
    assert len(calls) == 1
    assert "type of leave" in (r.message or "")


def test_ill_does_not_match_inside_will():
    assert not leave.leave_type_is_grounded("sick", "I will be away on Monday")
    assert leave.leave_type_is_grounded("sick", "I am ill today")


def test_invented_reason_fails_after_one_retry(monkeypatch):
    invented = as_json(leave_type="sick", start_date="2026-10-05", end_date="2026-10-06", reason="Flu")
    calls = scripted(monkeypatch, [invented, invented])
    r = leave.extract("I need two sick days off from Monday", "m", today=TODAY)
    assert r.status == "failed"
    assert r.attempts == 2
    assert len(calls) == 2
    assert any("Flu" in p for p in r.problems)


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