all:
	mkdir -p exec slib
	gcc -c src/AUI.c -o slib/AUI.o
	ar rcs slib/libAUI.a slib/AUI.o
	gcc main.c -o exec/main -lgdi32 -Lslib -lAUI
	rm -f slib/*.o

compile:
	gcc main.c -o exec/main -lgdi32 -Lslib -lAUI

uninstall:
	rm -rf exec slib/*.a slib/*.o
