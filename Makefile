TEST_LIB_DIRS=-L.
TEST_LIBS=-lcheck -lrbf

# Housekeeping:

all: basic_py_test wrapped_py_test c_test

clean:
	rm -f *.o *.so *.html *_test_aux.c c_test

# Main:

%.o: %.c
	gcc -c -Wall -Werror -fpic $^

librbf.so: rbf_train.o rbf_io.o rbf_query.o
	gcc -shared $^ -o $@

doc:
	pandoc ctypes2.md > ctypes2.html
	firefox ctypes2.html

# Python tests:

basic_py_test: librbf.so
	./test_rbf.py

wrapped_py_test: librbf.so
	./test_wrapped_rbf.py

# C tests:

rbf_test_aux.c: rbf_test.check
	checkmk $^ >$@

c_test: rbf_test.c rbf_test_aux.c librbf.so
	gcc $^ $(TEST_LIB_DIRS) $(TEST_LIBS) -o $@
	LD_LIBRARY_PATH=${LD_LIBRATH_PATH}:. ./$@
