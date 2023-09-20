# Code Style

Please follow this code style when contributing to `rwkv.cpp`.

This list is not complete.

## General

Overall, keep code in similar style as it was before.

- Keep lines at 180 characters or shorter.
- Separate logically grouped pieces of code with empty lines.
- Surround `if`, `for`, `while`, `do` and other similar statements with empty lines.
- Add trailing new line to the end of the file.

### Comments and messages

- Write documentation for public functions indended for outside use.
- Place single-line comments on the line before, not right after the code line.
- Begin comments with a capital letter, use correct grammar and punctuation.
- Begin messages, including error messages, with a capital letter.

## C/C++

- Use 4 spaces for indentation.
- Use [The One True Brace Style](https://en.wikipedia.org/wiki/Indentation_style#Variant:_1TBS_(OTBS)):
  - Place braces on the same line as the statement.
  - Always add braces to `if`, `for`, `while`, `do` and other similar statements.
- Prefix top-level function and struct names with `rwkv_`.
- Mark all functions that are not members of public API as `static`.
  - Public API is the set of functions defined in `rwkv.h`.
- Mark all immutable function arguments as `const`.
  - This is not required for local variables.

## Python

- Use 2 spaces for indentation.
- Specify types for functions and parameters.
  - For `void` functions, specify `-> None`.
- Specifying types for local variables:
  - required, if they are global
  - required, if they are compound (lists, dicts, optionals, etc.)
  - optional otherwise.
- Use types from `typing` (`List`, `Dict`) instead of built-in (`list`, `dict`).
