# Two binaries from one source: the serial reference (no -fopenmp, pragmas ignored)
# and the OpenMP build. FPFLAGS=-ffp-contract=off gives the exact-equality test build.
CC      ?= gcc
OPT     ?= -O3
FPFLAGS ?=
CFLAGS  = -std=gnu99 $(OPT) $(FPFLAGS) -Wall -Wextra -Wno-unused-function
LDLIBS  = -lm
SRC     = src/main.c src/pipeline.c src/io.c
HDR     = src/pipeline.h src/io.h
BIN     ?= bin

all: $(BIN)/frangi_serial $(BIN)/frangi_omp

$(BIN)/frangi_serial: $(SRC) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-variable -o $@ $(SRC) $(LDLIBS)

$(BIN)/frangi_omp: $(SRC) $(HDR)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(SRC) $(LDLIBS)

clean:
	rm -rf $(BIN)

.PHONY: all clean
