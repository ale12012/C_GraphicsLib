all:
	mkdir -p exec
	gcc -c src/mandelbrot.c -o slib/mandelbrot.o
	ar rcs slib/libmandelbrot.a slib/mandelbrot.o
	gcc libAUI.c -o exec/LibAUI -lgdi32 -Lslib -lmandelbrot
	rm -f slib/*.o

compile:
	gcc libAUI.c -o exec/libAUI -lgdi32 -Lslib -lmandelbrot

cuda:
	mkdir -p exec
	nvcc -arch=sm_80 -c src/mandelbrot.cu -o slib/mandelbrot.o
	ar rcs slib/libmandelbrot.a slib/mandelbrot.o
	gcc libAUI.c -o exec/LibAUI -lgdi32 -Lslib -lmandelbrot -L/usr/local/cuda/lib64 -lcudart
	rm -f slib/*.o

uninstall:
	rm -rf exec