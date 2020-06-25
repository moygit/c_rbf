CFLAGS=-Wall -Werror -fpic -fopenmp -O3 -march=native
LDFLAGS=-fopenmp
TEST_LIB_DIRS=-L.
TEST_LIBS=-lcheck -lrbf

# Housekeeping:

# all: basic_py_test wrapped_py_test c_test
all: c_test

clean:
	rm -f *.o *.so *.html *_test_aux.c c_test

# Main:

%.o: %.c rbf_utils.c
	gcc -c $(CFLAGS) $^

librbf.so: rbf_utils.o rbf_train.o rbf_io.o rbf_query.o
	gcc -shared $(LDFLAGS) $^ -o $@

doc:
	pandoc ctypes2.md > ctypes2.html
	firefox ctypes2.html

# Python tests:

wrapped_py_test: librbf.so
	./test_wrapped_rbf.py

# C tests:

rbf_test_aux.c: rbf_test.check
	checkmk $^ >$@

c_test: rbf_test.c rbf_test_aux.c librbf.so
	# TODO: use full CFLAGS
	# gcc $(CFLAGS) $(LDFLAGS) $(TEST_LIB_DIRS) $(TEST_LIBS) $^ -o $@
	gcc -fopenmp $^ $(TEST_LIB_DIRS) $(TEST_LIBS) -o $@
	LD_LIBRARY_PATH=${LD_LIBRATH_PATH}:. ./$@
