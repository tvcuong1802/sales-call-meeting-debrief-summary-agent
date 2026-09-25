"""A value the fold produced must survive the composition boundary.

MainNode returns an explicit key-set. Anything a sub-node produced and that list omits is
dropped there, silently -- no error, no warning, nothing in the trace. That is what
happened to the resolved language: StructuredExtractNode returned "ja", the renderer
received "en", and a Japanese debrief kept English headings.

It cost two wrong guesses first -- at the model's wording, then at the language detector
-- because both of those are plausible explanations for "the reply is mixed". The test
that would have found it in one step asserts the value at BOTH ends of the boundary.
"""

from src.nodes.main_node import MainNode


class _Extract:
    """Stands in for StructuredExtractNode: produces a summary and a resolved language."""

    def __init__(self, language: str = "ja") -> None:
        self._language = language

    def execute(self, state, config=None):
        return {
            "summary": "要約",
            "crm_fields": "{}",
            "cannot_summarize": False,
            "output_language": self._language,
        }


class _Validate:
    def execute(self, state, config=None):
        return {}


def _fold(language: str = "ja"):
    node = MainNode(structured_extract_node=_Extract(language), crm_schema_validate_node=_Validate())
    return node.execute({"user_input": "x"})


def test_the_resolved_language_crosses_the_boundary():
    assert _fold("ja")["output_language"] == "ja"


def test_english_crosses_it_too():
    assert _fold("en")["output_language"] == "en"


def test_an_unresolved_language_crosses_as_empty_not_as_a_default():
    # "" means no opinion, and the renderer prints both notices on that branch. Turning
    # it into "en" here would silently pick a language nobody determined.
    assert _fold("")["output_language"] == ""


def test_the_summary_still_crosses_so_this_test_is_not_measuring_an_empty_fold():
    assert _fold("ja")["summary"] == "要約"


# --- a message with no letters is settled where it is still the user's ----------------
def test_a_message_with_no_letters_is_settled_in_pre_process():
    """Where this check lives is the whole point.

    Downstream, the nuance pass has stamped labels onto the transcript, so a message of
    pure punctuation arrives carrying letters that were never typed. Three attempts at
    the check there could not tell the two apart; a predicate needing a third fix is in
    the wrong place, not missing a fourth patch.
    """
    from src.nodes.pre_process_node import PreProcessNode

    result = PreProcessNode().execute({"user_input": "???  ..."})
    assert result["cannot_summarize"] is True
    assert result["output_language"] == "", "no language was determined, so neither is claimed"


def test_a_blank_message_is_left_to_the_existing_rejection():
    # S1_INPUT_REJECTED is a contract callers depend on. The no-letters branch must not
    # take it over.
    from src.nodes.pre_process_node import PreProcessNode

    result = PreProcessNode().execute({"user_input": "   "})
    assert "S1_INPUT_REJECTED" in str(result.get("error", ""))


def test_a_real_transcript_is_not_mistaken_for_an_empty_one():
    # The transcript arrives as raw_transcript from the state factory and as user_input
    # from invoke. Reading only one made a normal transcript short-circuit.
    from src.nodes.pre_process_node import PreProcessNode

    for key in ("user_input", "raw_transcript"):
        result = PreProcessNode().execute({key: "Sales call with Acme. Demo next week."})
        assert not result.get("cannot_summarize"), f"a transcript in {key} was treated as empty"
