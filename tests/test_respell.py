from wiseguy.engine import sentences
from wiseguy.respell import respell


def test_off_is_untouched():
    t = "I'm going to talk to the boss about the car."
    assert respell(t, 0) == t


def test_slang():
    assert respell("I'm going to get you", 1) == "I'm gonna get you"
    assert respell("We're going to the store", 1) == "We're going to the store"
    assert respell("Forget about it!", 1) == "Forget about it!"
    assert respell("What are you doing?", 1) == "Whaddaya doing?"


def test_heavy():
    out = respell("The brother thinks there is nothing better, and I'm talking.", 2)
    assert out == "Duh brudduh tinks there is nuttin better, and I'm talkin'."


def test_short_words_and_keeps_case():
    assert respell("THE thing", 2) == "DUH ting"
    assert respell("her", 2) == "her"
    assert respell("king", 2) == "king"


def test_old_school():
    assert respell("the car and the coffee, you know", 3) == "duh car and duh cawfee, ya know"


def test_sentences_split_long_first():
    parts = sentences("Listen to me, I been running this whole block since nineteen seventy eight and nobody tells me nothing, capisce? Good.")
    assert parts[0] == "Listen to me," and parts[-1] == "Good."


def test_loud_reads_statements_as_exclamations():
    from wiseguy.engine import Wiseguy

    g = Wiseguy.__new__(Wiseguy)
    g.strength, g.loud = 0, True
    assert g.text("Get out of here. Now. What?") == "Get out of here! Now! What?"
    assert g.text("Wait... what.") == "Wait... what!"
