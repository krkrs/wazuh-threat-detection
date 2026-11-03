"""Walidacja Markdown sprawozdania: grafiki, tabele, naglowki.

Sprawdza:
 - sciezki grafiki istniejaja wzgledem katalogu report/,
 - kazdy naglowek tabeli ma wiersz oddzielajacy o tej samej liczbie kolumn,
 - tabele nie sa "osierocone" (wiersz tabeli po pustej linii nie jest sygnalem bledu).
"""

import re
from pathlib import Path

REPORT = Path("report/sprawozdanie.md")
BASE = Path("report")
SEPARATOR = re.compile(r"^\|[\s:|-]+\|$", re.MULTILINE)


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def column_count(line: str) -> int:
    return line.strip().count("|") - 1


def main() -> None:
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    index = 0
    tables = 0
    while index < len(lines):
        if not is_table_row(lines[index]):
            index += 1
            continue
        header = lines[index].strip()
        if index + 1 >= len(lines) or not SEPARATOR.match(lines[index + 1].strip()):
            problems.append(f"naglowek tabeli bez wiersza oddzielajacego: linia {index + 1}")
            index += 1
            continue
        tables += 1
        expected = column_count(header)
        if column_count(lines[index + 1].strip()) != expected:
            problems.append(
                f"liczba kolumn oddzielenia ({column_count(lines[index + 1].strip())}) "
                f"nie zgadza sie z naglowkiem ({expected}) w linii {index + 2}"
            )
        index += 2
        while index < len(lines) and is_table_row(lines[index]):
            if column_count(lines[index]) != expected:
                problems.append(f"wiersz o innej liczbie kolumn ({column_count(lines[index])}) w linii {index + 1}")
            index += 1

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("!["):
            target = stripped.split("](")[-1].rstrip(")") if "](" in stripped else ""
            if target and not (BASE / target).exists():
                problems.append(f"brak pliku grafiki w linii {number}: {target}")

    headings = sum(1 for x in lines if x.startswith("#"))
    images = sum(1 for x in lines if x.startswith("!["))
    print(f"tabele: {tables}, naglowki: {headings}, grafiki: {images}, wiersze: {len(lines)}")
    print("problems:", len(problems))
    for item in problems[:15]:
        print(" -", item)


if __name__ == "__main__":
    main()
