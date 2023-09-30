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
	nvcc -c src/mandelbrot.cu -o slib/mandelbrot.obj
	ar rcs slib/mandelbrot.lib slib/mandelbrot.obj
	nvcc slib/mandelbrot.obj libAUI.c -o exec/LibAUI -lgdi32 -lUser32 -Lslib slib/mandelbrot.lib -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64" -lcudart
	rm -f slib/*.obj

uninstall:
	rm -rf exec