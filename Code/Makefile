all:
	mkdir -p exec slib
	gcc -c src/mandelbrot.c -o slib/mandelbrot.o
	ar rcs slib/libmandelbrot.a slib/mandelbrot.o
	gcc libAUI.c -o exec/LibAUI -lgdi32 -Lslib -lmandelbrot
	rm -f slib/*.o

compile:
	gcc libAUI.c -o exec/libAUI -lgdi32 -Lslib -lmandelbrot

uninstall:
	rm -rf exec slib/*.a slib/*.o