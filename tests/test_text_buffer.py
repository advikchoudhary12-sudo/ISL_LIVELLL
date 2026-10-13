import pytest

from isl_recognition.text_buffer import TextBuffer


def test_confirmed_letters_and_controls():
    buffer = TextBuffer()
    buffer.append_letter("h")
    buffer.append_letter("I")
    buffer.append_space()
    buffer.append_space()
    assert buffer.text == "HI "
    buffer.backspace()
    assert buffer.text == "HI"
    buffer.clear()
    assert buffer.text == ""


def test_text_buffer_rejects_non_letters():
    buffer = TextBuffer()
    with pytest.raises(ValueError):
        buffer.append_letter("AA")
    with pytest.raises(ValueError):
        buffer.append_letter(" ")
