class TextBuffer:
    def __init__(self) -> None:
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    def append_letter(self, letter: str) -> None:
        normalized = letter.upper()
        if len(normalized) != 1 or normalized not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise ValueError("letter must be one English alphabet character")
        self._text += normalized

    def append_space(self) -> None:
        if self._text and not self._text.endswith(" "):
            self._text += " "

    def backspace(self) -> None:
        self._text = self._text[:-1]

    def clear(self) -> None:
        self._text = ""

