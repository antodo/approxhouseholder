# LAPACK_DIR = $(HOME)/lapack-3.7.0
# LAPACK_INC = -I$(LAPACK_DIR)/LAPACKE/include -I$(LAPACK_DIR)/CBLAS/include
# LAPACK_LIB = -L$(LAPACK_DIR) -lcblas -llapacke -llapack -lrefblas -lgfortran
LAPACK_LIB = -llapack -lblas

# LAPACK_INC = -I$(MKLROOT)/include
# LAPACK_LIB = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lrt
# LAPACK_LIB = -L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lrt # -lstdc++ -lm -lgfortran

CC = gcc -fopenmp
CFLAGS = -std=gnu99 -Wall -O3 $(LAPACK_INC) # -DUSE_FLOAT
LDFLAGS = $(LAPACK_LIB) -lm

# CC = icc
# CFLAGS = -qopenmp -O3 $(LAPACK_INC)
# LDFLAGS = -qopenmp $(LAPACK_LIB)

.SECONDARY:

EXE = memo_qr # qr

all: $(EXE)

qr: qr.o common.o
	$(CC) $^ $(LDFLAGS) -o $@

memo_qr: memo_qr.o memo_qr_panel.o common.o
	$(CC) $^ $(LDFLAGS) -o $@

clean:
	/bin/rm -rf tags core *.o $(EXE)

tags: *.c *.h
	ctags -R *.c *.h

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $< -o $@
