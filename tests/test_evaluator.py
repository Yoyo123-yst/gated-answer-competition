from sequence_slot_attack.evaluator import AMBIGUOUS, CORRECT, INVALID_OUTPUT, WRONG, classify_output


def test_evaluator_accepts_normalized_exact_alias():
    assert classify_output("The Coca Cola Company!", ["the coca cola company"]).status == CORRECT


def test_evaluator_marks_partial_anls_as_ambiguous():
    assert classify_output("abcde", ["abcdef"]).status == AMBIGUOUS


def test_evaluator_marks_low_overlap_as_wrong():
    assert classify_output("Beverage Volume", ["The Coca-Cola Company"]).status == WRONG


def test_evaluator_rejects_unterminated_or_empty_output():
    assert classify_output("answer", ["answer"], terminated_eos=False).status == INVALID_OUTPUT
    assert classify_output("...", ["answer"]).status == INVALID_OUTPUT
