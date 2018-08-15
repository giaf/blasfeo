CC = gcc

CFLAGS = -O2 -fPIC -mavx -DOS_LINUX
ASFLAGS = -mavx -DOS_LINUX

#OBJS = kernel/avx/kernel_dgemm_4x4_lib.o tests/test.o
OBJS = kernel/avx2/kernel_dgemm_12x4_lib.o kernel/avx2/kernel_dgemm_8x4_lib.o kernel/avx2/kernel_dgemm_4x4_lib.o tests/test.o

build: $(OBJS)
	$(CC) $(OBJS) /home/gianluca/blasfeo/lib/libblasfeo.a -lm -o tests/test.out
	@echo ""
	@echo "Done!"
	@echo ""

run:
	mkdir -p tests/data
#	./tests/test.out
	./tests/test.out > tests/data/blas.m

print:
	cp tests/print_blas.m tests/data
	cd tests/data; octave-cli print_blas.m
	cd tests/data/ ; epstopdf blas.eps -out blas.pdf ; rm blas.eps
clean:
	rm *.o
