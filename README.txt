This combines two howtos:
1. Using CFFI in Python: https://dbader.org/blog/python-cffi
   He uses the fastest method (i.e. fastest in runtime performance).
2. Using Check for unit-testing in C: https://stackoverflow.com/a/15046864/9519712

I've combined the two for writing a Python extension in C. So we have here a Makefile that will
build and run C tests, build a C library, and call the C library from Python in slightly different ways.
(Basically, `testPoint.py` is simpler for the author and slightly grungier for the caller;
`testWrappedPoint.py` shows a way to hide details from the caller.)
See ctypes2.md for details.

