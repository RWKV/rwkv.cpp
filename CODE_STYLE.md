# Code Style

Please follow this code style when contributing to `rwkv.cpp`.

This list is not complete.

## General

Overall, keep code in similar style as it was before.

- Keep lines at 180 characters or shorter.
- Separate logically grouped pieces of code with empty lines.
- Surround `if`, `for`, `while`, `do` and other similar statements with empty lines.
- Write documentation for public functions indended for outside use.
- Place single-line comments on the line before, not right after the code line.
- Start comments with a capital letter, use correct grammar and punctuation.

## C/C++

- Use 4 spaces for indentation.
- Use [The One True Brace Style](https://en.wikipedia.org/wiki/Indentation_style#Variant:_1TBS_(OTBS)):
  - Place braces on the same line as the statement.
  - Always add braces to `if`, `for`, `while`, `do` and other similar statements.

## Python

- Use 2 spaces for indentation.
- Specify types for functions and parameters.
  - For `void` functions, specify `-> None`.
- Specifying types for local variables:
  - required, if they are global
  - required, if they are compound (lists, dicts, optionals, etc.)
  - optional otherwise.
- Use types from `typing` (`List`, `Dict`) instead of built-in (`list`, `dict`).
