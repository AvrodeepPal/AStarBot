from rag.followup import is_follow_up, last_user_turn, standalone_query

PREV = [
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "explain his credit risk eda project"},
    {"role": "assistant", "content": "He ranked features with ANOVA F-tests."},
]


def test_referent_words_trigger_rewrite():
    q = "how did he do it, what data used and algos used, how were the results and score"
    assert is_follow_up(q)
    assert standalone_query(q, PREV) == "explain his credit risk eda project " + q


def test_continuation_openers_trigger_rewrite():
    for q in ("and the results?", "what about the dataset", "tell me more", "in detail please", "why"):
        assert is_follow_up(q), q
        assert standalone_query(q, PREV).startswith("explain his credit risk eda project ")


def test_self_contained_questions_are_left_alone():
    for q in ("any hobbies", "research interests", "does he watch anime", "where does he work",
              "what is his favourite language", "is he gate qualified"):
        assert not is_follow_up(q), q
        assert standalone_query(q, PREV) == q


def test_no_previous_user_turn_means_no_rewrite():
    assert standalone_query("tell me more about it", []) == "tell me more about it"
    only_bot = [{"role": "assistant", "content": "hi"}]
    assert standalone_query("tell me more about it", only_bot) == "tell me more about it"
    assert last_user_turn([{"role": "assistant", "content": "hi"}]) is None


def test_repeated_question_is_not_doubled():
    q = "tell me more about it"
    assert standalone_query(q, [{"role": "user", "content": q}]) == q
