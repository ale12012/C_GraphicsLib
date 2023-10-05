all:
	mkdir -p exec slib
	gcc -c src/newtonsfract.c -o slib/newtonsfract.o
	ar rcs slib/libnewtonsfract.a slib/newtonsfract.o
	gcc libAUI.c -o exec/LibAUI -lgdi32 -Lslib -lnewtonsfract
	rm -f slib/*.o

compile:
	gcc libAUI.c -o exec/libAUI -lgdi32 -Lslib -lnewtonsfract

uninstall:
	rm -rf exec slib/*.a slib/*.o