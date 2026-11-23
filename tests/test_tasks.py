import pytest

from vwatch import tasks
from vwatch.schemas import DEFECT_DEFINITIONS, QAItem, QATruth, Task


@pytest.fixture
def qa_truth():
    return QATruth(video_id="v", duration=10, items=(
        QAItem(id="q1", question="What number is on the sign?", answer="7391", answer_type="number"),
        QAItem(id="q2", question="What word is written?", answer="zebrafish", answer_type="text")))


class TestTaskPrompt:
    def test_t1_lists_every_defect_type(self):
        p = tasks.task_prompt(Task.DEFECTS)
        for t in DEFECT_DEFINITIONS:
            assert t.value in p
        assert '"defects"' in p and "empty list" in p

    def test_t2(self):
        p = tasks.task_prompt(Task.EVENTS)
        assert '"events"' in p and "flash" in p

    def test_t3(self):
        p = tasks.task_prompt(Task.CAUSAL)
        assert '"retell"' in p and "because" in p

    def test_t4_requires_qatruth(self):
        with pytest.raises(ValueError, match="QATruth"):
            tasks.task_prompt(Task.QA)
        with pytest.raises(ValueError, match="QATruth"):
            tasks.task_prompt(Task.QA, truth={"items": []})

    def test_t4_lists_questions_never_answers(self, qa_truth):
        p = tasks.task_prompt(Task.QA, qa_truth)
        assert "q1: What number is on the sign?" in p and "q2: What word is written?" in p
        assert "7391" not in p and "zebrafish" not in p
        assert '"answers"' in p

    def test_non_qa_ignore_truth(self, qa_truth):
        assert tasks.task_prompt(Task.DEFECTS, qa_truth) == tasks.task_prompt(Task.DEFECTS)

    def test_free_task_not_a_scored_prompt(self):
        with pytest.raises(ValueError, match="unknown task"):
            tasks.task_prompt(Task.FREE)

    def test_free_prompt(self):
        p = tasks.free_prompt("  what colour is the ball?  ")
        assert p.startswith("TASK: what colour is the ball?\n") and '"moments"' in p

    @pytest.mark.parametrize("q", ["", "   \n"])
    def test_free_prompt_empty(self, q):
        with pytest.raises(ValueError, match="empty"):
            tasks.free_prompt(q)

    def test_repair_prompt_format(self):
        assert "(boom)" in tasks.REPAIR_PROMPT.format(error="boom")


class TestParseReport:
    def test_bare_defects(self):
        parsed, err = tasks.parse_report(
            Task.DEFECTS, '{"defects": [{"type": "Flash Frame", "start": 1.0, "end": 1.1, "evidence": "e"}]}')
        assert err is None
        assert parsed["defects"][0]["type"] == "flash_frame" and parsed["defects"][0]["start"] == 1.0

    def test_fenced_with_prose(self):
        parsed, err = tasks.parse_report(Task.EVENTS, 'ok\n```json\n{"events": [{"start": 1, "description": "red dot"}]}\n```')
        assert err is None and parsed["events"][0]["end"] is None

    def test_final_wrapper_unwrapped(self):
        parsed, err = tasks.parse_report(Task.CAUSAL, '{"final": {"retell": ["a because b"]}}')
        assert err is None and parsed == {"retell": ["a because b"]}

    def test_final_wrapper_qa(self):
        parsed, err = tasks.parse_report(Task.QA, '{"final": {"answers": [{"id": "q1", "answer": "80"}]}}')
        assert err is None and parsed["answers"] == [{"id": "q1", "answer": "80"}]

    def test_extra_fields_ignored(self):
        parsed, err = tasks.parse_report(Task.CAUSAL, '{"retell": [], "thoughts": "hm"}')
        assert err is None and parsed == {"retell": []}

    def test_empty_defects_ok(self):
        assert tasks.parse_report(Task.DEFECTS, '{"defects": []}') == ({"defects": []}, None)

    def test_schema_mismatch(self):
        parsed, err = tasks.parse_report(Task.DEFECTS, '{"defects": [{"type": "flash_frame"}]}')
        assert parsed is None and err.startswith("schema mismatch")

    def test_schema_mismatch_inside_final(self):
        parsed, err = tasks.parse_report(Task.FREE, '{"final": {"moments": []}}')
        assert parsed is None and "schema mismatch" in err

    def test_no_json(self):
        parsed, err = tasks.parse_report(Task.DEFECTS, "I see nothing")
        assert parsed is None and "no JSON object" in err

    def test_free_report(self):
        parsed, err = tasks.parse_report(Task.FREE, '{"answer": "red", "moments": [{"t": 2, "what": "x"}]}')
        assert err is None and parsed["answer"] == "red" and parsed["moments"][0]["t"] == 2

    def test_output_is_json_serialisable(self):
        import json

        parsed, _ = tasks.parse_report(Task.DEFECTS, '{"defects": [{"type": "jump_cut", "start": 1}]}')
        json.dumps(parsed)


class TestFreshPrompt:
    def test_fresh_items_get_the_generic_prompt(self):
        from vwatch.schemas import DefectTruth
        t = DefectTruth(video_id="fxs_abc_clean", source="abc", duration=50.0)
        p = tasks.task_prompt(Task.DEFECTS, t)
        assert "Hindi" not in p and "caption_blocked" not in p and "jump_cut" in p and "audio_dropout" in p

    def test_old_items_keep_their_prompt(self):
        from vwatch.schemas import DefectTruth
        t = DefectTruth(video_id="t1x_sem_clean_0", source="sem", duration=50.0)
        assert tasks.task_prompt(Task.DEFECTS, t) == tasks.task_prompt(Task.DEFECTS)
