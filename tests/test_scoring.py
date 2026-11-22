import pytest

from vwatch.evaluation import scoring as S
from vwatch.schemas import (
    CausalLink, CausalReport, CausalTruth, DefectReport, DefectTruth, EventReport, EventTruth,
    Observation, QAAnswer, QAItem, QAReport, QATruth, ReportedDefect, ReportedEvent, RunRecord,
    Task, TruthDefect, TruthEvent,
)


def dtruth(*defects):
    return DefectTruth(video_id="v", source="s", duration=30,
                       defects=tuple(TruthDefect(type=t, start=s, end=e) for t, s, e in defects))


def drep(*items):
    return DefectReport(defects=tuple(ReportedDefect(type=t, start=s, end=e) for t, s, e in items))


# ---------------------------------------------------------------- T1


def test_defect_basic_match_and_fields():
    r = S.score_defects(dtruth(("flash_frame", 5, 5.1)), drep(("Flash Frame", 5.3, None)))
    assert (r.tp, r.fp, r.fn) == (1, 0, 0)
    row = r.rows[0]
    assert row.found and row.start_error == pytest.approx(0.3) and row.loc_hit


def test_defect_type_mismatch():
    r = S.score_defects(dtruth(("flash_frame", 5, 5.1)), drep(("frozen_frame", 5, None)))
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)
    assert r.rows[0].start_error is None and not r.rows[0].loc_hit


def test_defect_tolerance_boundary_and_loc():
    t = dtruth(("jump_cut", 10, 10.2))
    assert S.score_defects(t, drep(("jump_cut", 11.5, None))).tp == 1
    assert S.score_defects(t, drep(("jump_cut", 11.6, None))).tp == 0
    r = S.score_defects(t, drep(("jump_cut", 10.6, None)))
    assert r.tp == 1 and not r.rows[0].loc_hit
    assert S.score_defects(t, drep(("jump_cut", 10.5, None))).rows[0].loc_hit


def test_defect_overlap_rule():
    t = dtruth(("frozen_frame", 10, 16))
    # start is 4 s off, but reported interval overlaps truth interval
    assert S.score_defects(t, drep(("frozen_frame", 6, 10.4))).tp == 1
    assert S.score_defects(t, drep(("frozen_frame", 6, 9.0))).tp == 1  # padded intervals touch
    assert S.score_defects(t, drep(("frozen_frame", 6, 8.9))).tp == 0
    # reported end before start is treated as a point
    assert S.score_defects(t, drep(("frozen_frame", 6, 5))).tp == 0


def test_defect_one_to_one_duplicates_and_prefers_closest():
    t = dtruth(("flash_frame", 5, 5.1), ("flash_frame", 6, 6.1))
    r = S.score_defects(t, drep(("flash_frame", 5.9, None)))
    assert r.tp == 1 and r.fn == 1 and r.rows[1].found and not r.rows[0].found
    r = S.score_defects(dtruth(("flash_frame", 5, 5.1)), drep(("flash_frame", 5, None), ("flash_frame", 5.2, None)))
    assert (r.tp, r.fp) == (1, 1)


def test_defect_false_alarm_unknown_and_none():
    clean = dtruth()
    r = S.score_defects(clean, drep(("flash_frame", 1, None), ("mystery", 2, None)))
    assert r.false_alarm and r.fp == 2 and r.unknown_types == 1
    assert not S.score_defects(clean, drep()).false_alarm
    none = S.score_defects(dtruth(("flash_frame", 1, 1.1)), None)
    assert (none.tp, none.fp, none.fn) == (0, 0, 1) and not none.false_alarm
    assert not S.score_defects(dtruth(("flash_frame", 1, 1.1)), drep(("mystery", 1, None))).false_alarm
    with pytest.raises(ValueError):
        S.score_defects(clean, None, match_tol=-1)


# ---------------------------------------------------------------- T2


def etruth(*evs):
    return EventTruth(video_id="v", duration=20, events=tuple(
        TruthEvent(id=f"e{i}", start=s, end=e, kind="appear", subject="x", keywords=kw)
        for i, (s, e, kw) in enumerate(evs)))


def erep(*items):
    return EventReport(events=tuple(ReportedEvent(start=s, end=e, description=d) for s, e, d in items))


def test_buckets():
    assert [S.duration_bucket(d) for d in (0, 0.49, 0.5, 0.99, 1.0, 3.0, 3.01)] == [
        "<0.5s", "<0.5s", "0.5-1s", "0.5-1s", "1-3s", "1-3s", ">3s"]


def test_event_match_keywords_plural_hyphen_case():
    t = etruth((2, 2.3, ("green", "star")))
    assert S.score_events(t, erep((2.5, None, "A GREEN star appears"))).tp == 1
    assert S.score_events(t, erep((2.5, None, "green stars appear"))).tp == 1
    assert S.score_events(t, erep((2.5, None, "green-star pops up"))).tp == 1
    assert S.score_events(t, erep((2.5, None, "green circle"))).tp == 0
    assert S.score_events(t, erep((2.5, None, "greenish starfish"))).tp == 0


def test_keyword_forms():
    assert S.keyword_in("many boxes", "box") and S.keyword_in("a box", "boxes")
    assert S.keyword_in("two cherries", "cherry") and S.keyword_in("a cherry", "cherries")
    assert S.keyword_in("red-ball", "red ball")
    assert not S.keyword_in("anything", "")


def test_event_time_tolerance_overlap_and_oneto_one():
    t = etruth((10, 10.4, ("star",)), (10.2, 10.6, ("star",)))
    assert S.score_events(t, erep((11.0, None, "star"))).tp >= 1
    assert S.score_events(t, erep((12.0, None, "star"))).tp == 0
    assert S.score_events(t, erep((8, 8.9, "star"))).tp == 0
    assert S.score_events(t, erep((8, 9.0, "star"))).tp == 1
    t1 = etruth((10, 12, ("star",)))
    assert S.score_events(t1, erep((7, 8.9, "star"))).tp == 0
    assert S.score_events(t1, erep((7, 9.0, "star"))).tp == 1
    r = S.score_events(t, erep((10.1, None, "star"), (10.2, None, "star"), (10.3, None, "star")))
    assert (r.tp, r.fp, r.fn) == (2, 1, 0)
    with pytest.raises(ValueError):
        S.score_events(t, None, tol=-1)


def test_event_rows_buckets_none_report():
    t = etruth((1, 1.2, ("a",)), (5, 9, ("b",)))
    r = S.score_events(t, None)
    assert [x.bucket for x in r.rows] == ["<0.5s", ">3s"] and r.fn == 2 and r.fp == 0


# ---------------------------------------------------------------- T4


def qtruth():
    return QATruth(video_id="v", duration=10, items=(
        QAItem(id="q1", question="?", answer="80", answer_type="number"),
        QAItem(id="q2", question="?", answer="Green Star", answer_type="text"),
        QAItem(id="q3", question="?", answer="two", answer_type="number"),
    ))


def test_extract_number():
    assert S.extract_number("₹80") == 80
    assert S.extract_number("It is 80 rupees") == 80
    assert S.extract_number("Rs.1,200 total") == 1200
    assert S.extract_number("about 3.5 units") == 3.5
    assert S.extract_number("eighteen and 4") == 18
    assert S.extract_number("Twenty") == 20
    assert S.extract_number("someone said none") is None
    assert S.extract_number("-5 degrees") == -5


def test_score_qa():
    rep = QAReport(answers=(QAAnswer(id="q1", answer="₹80."), QAAnswer(id="q2", answer="the GREEN star!"),
                            QAAnswer(id="q3", answer="three")))
    r = S.score_qa(qtruth(), rep)
    assert (r.correct, r.total) == (2, 3) and r.rows[2].answered and not r.rows[2].correct
    r = S.score_qa(qtruth(), QAReport(answers=(QAAnswer(id="q1", answer="85"), QAAnswer(id="q2", answer="star green"))))
    assert r.correct == 0
    r = S.score_qa(qtruth(), None)
    assert r.correct == 0 and not any(x.answered for x in r.rows)
    r = S.score_qa(qtruth(), QAReport(answers=(QAAnswer(id="q1", answer="none"), QAAnswer(id="q1", answer="80"))))
    assert r.rows[0].answered and not r.rows[0].correct  # first answer wins


def test_qa_bad_truth_number():
    t = QATruth(video_id="v", duration=1, items=(QAItem(id="q", question="?", answer="lots", answer_type="number"),))
    with pytest.raises(ValueError):
        S.score_qa(t, QAReport(answers=(QAAnswer(id="q", answer="5"),)))


# ---------------------------------------------------------------- T3


def ctruth():
    return CausalTruth(video_id="v", duration=10, distractors=("a bird flies by",), links=(
        CausalLink(id="L1", cause="ball hits glass", effect="glass breaks", time=1),
        CausalLink(id="L2", cause="alarm rings", effect="man wakes", time=4),
    ))


def test_prompt_contents():
    p = S.build_causal_prompt(ctruth(), CausalReport(retell=("The ball broke the glass because it hit it.",)))
    for s in ("L1: ball hits glass -> glass breaks", "L2:", "D1: a bird flies by", "1. The ball broke", "strict JSON", "invented"):
        assert s in p


def test_score_causal_fake_judge():
    seen = []

    def judge(prompt):
        seen.append(prompt)
        return 'Sure! ```json\n{"matched": ["L2", "L2"], "invented": 1}\n``` done'

    r = S.score_causal(ctruth(), CausalReport(retell=("x because y",)), judge)
    assert r.matched == ("L2",) and r.recall == 0.5 and r.invented == 1 and len(seen) == 1


def test_score_causal_empty_skips_judge():
    def boom(_):
        raise AssertionError("must not be called")

    for rep in (None, CausalReport(retell=()), CausalReport(retell=("  ",))):
        r = S.score_causal(ctruth(), rep, boom)
        assert r.recall == 0.0 and r.invented == 0 and r.n_links == 2


@pytest.mark.parametrize("out", [
    "no json at all", '{"matched": "L1", "invented": 0}', '{"matched": [], "invented": -1}',
    '{"matched": ["L9"], "invented": 0}', '{"matched": [', "{not json} [1,2]",
])
def test_score_causal_malformed(out):
    with pytest.raises(S.JudgeError):
        S.score_causal(ctruth(), CausalReport(retell=("s",)), lambda _: out)


def test_judge_skips_leading_junk_and_nested():
    v = S.parse_judge_output('{oops} then {"matched": ["L1"], "invented": 0, "why": {"a": 1}}', ["L1"])
    assert v.matched == ["L1"]
    assert S.parse_judge_output('[1] {"matched": [], "invented": 2}', ["L1"]).invented == 2


# ---------------------------------------------------------------- unseen claims


def rec(task, parsed, obs=()):
    return RunRecord(run_id="r", arm="W", budget_tokens=1, repeat=0, video_id="v", task=task, model="m",
                     parsed=parsed, observations=tuple(Observation(kind="frame", t0=a, t1=b) for a, b in obs))


def test_unseen_claims():
    p = {"defects": [{"type": "flash_frame", "start": 5.0}, {"type": "flash_frame", "start": 20.0}]}
    assert S.unseen_claims(rec(Task.DEFECTS, p, [(4.5, 5.5)])) == 1
    assert S.unseen_claims(rec(Task.DEFECTS, p, [(4.5, 5.5), (19.6, 19.6)])) == 0
    assert S.unseen_claims(rec(Task.DEFECTS, p, [(4.5, 5.5), (18.0, 18.5)]), tol=1.0) == 1
    assert S.unseen_claims(rec(Task.DEFECTS, p)) == 2
    assert S.unseen_claims(rec(Task.DEFECTS, None)) == 0
    assert S.unseen_claims(rec(Task.DEFECTS, {"defects": "bad"})) == 0
    ev = {"events": [{"start": 3.0, "description": "x"}]}
    assert S.unseen_claims(rec(Task.EVENTS, ev, [(0, 1)])) == 1
    assert S.unseen_claims(rec(Task.QA, {"answers": []})) == 0


# ---------------------------------------------------------------- validity fixes


def test_overlap_branch_capped_by_truth_length():
    t = dtruth(("flash_frame", 10, 10.1))
    # whole-video interval starts far away: overlap would match, but length cap rejects it
    assert S.score_defects(t, drep(("flash_frame", 8.0, 30.0))).tp == 0
    # start 1.8 s away (outside +-1.5) but short interval reaching the truth: cap = 2*0.1+1.0 = 1.2 s
    assert S.score_defects(t, drep(("flash_frame", 8.2, 9.4))).tp == 1
    assert S.score_defects(t, drep(("flash_frame", 8.2, 9.5))).tp == 0  # 1.3 s long > cap
    # start-time branch is unaffected by length
    assert S.score_defects(t, drep(("flash_frame", 9.0, 40.0))).tp == 1


def test_failed_defect_score():
    clean = S.failed_defect_score(dtruth())
    assert clean.false_alarm and (clean.tp, clean.fp, clean.fn) == (0, 1, 0)
    bad = S.failed_defect_score(dtruth(("jump_cut", 3, 3.2)))
    assert (bad.tp, bad.fp, bad.fn) == (0, 0, 1) and not bad.false_alarm


def test_qa_hedged_answers_rejected():
    def run(items, answers):
        t = QATruth(video_id="v", duration=1, items=tuple(items))
        return S.score_qa(t, QAReport(answers=tuple(QAAnswer(id=i, answer=a) for i, a in answers)))
    txt = QAItem(id="t", question="?", answer="Green Star", answer_type="text")
    assert run([txt], [("t", "green star")]).correct == 1
    assert run([txt], [("t", "The green star")]).correct == 1  # 3 tokens, contains truth
    assert run([txt], [("t", "It is the green star")]).correct == 0  # too long
    assert run([txt], [("t", "green star or red star")]).correct == 0
    assert run([txt], [("t", "green star/ball")]).correct == 0
    assert run([txt], [("t", "red star")]).correct == 0
    num = QAItem(id="n", question="?", answer="80", answer_type="number")
    assert run([num], [("n", "80 or 90")]).correct == 0
    assert run([num], [("n", "eighty is wrong, it's 80")]).correct == 1  # 'eighty' is not a known number word
    assert run([num], [("n", "80 and 80")]).correct == 1  # one distinct number
    assert run([num], [("n", "between 70 and 80")]).correct == 0
    assert run([num], [("n", "none")]).correct == 0
    assert S.extract_numbers("two or 3, 1,000") == [2, 3, 1000]


def test_unseen_only_counts_visual_audio_and_tight_tolerance():
    p = {"defects": [{"type": "flash_frame", "start": 5.0}]}
    def with_kinds(*kinds, t0=4.0, t1=6.0):
        return RunRecord(run_id="r", arm="W", budget_tokens=1, repeat=0, video_id="v", task=Task.DEFECTS,
                         model="m", parsed=p,
                         observations=tuple(Observation(kind=k, t0=t0, t1=t1) for k in kinds))
    assert S.unseen_claims(with_kinds("signal")) == 1
    assert S.unseen_claims(with_kinds("transcript", "signal")) == 1
    for k in ("frame", "crop", "audio"):
        assert S.unseen_claims(with_kinds(k)) == 0
    assert S.unseen_claims(with_kinds("frame", t0=5.6, t1=6)) == 1  # 0.6 s away > 0.5
    assert S.unseen_claims(with_kinds("frame", t0=5.4, t1=6)) == 0


def test_parse_judge_output_ids_are_case_insensitive():
    from vwatch.evaluation.scoring import parse_judge_output

    v = parse_judge_output('{"matched": ["L1", "l2", "L1"], "invented": 0}', ["l1", "l2", "l3"])
    assert v.matched == ["l1", "l2"]
