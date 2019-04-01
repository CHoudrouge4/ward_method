CC = g++
CFLAGS = -g -Wall
C = -c
VERSION = -std=c++14
OUTPUT  = co.exe
SOURCES = main.cpp wards.cpp
OBJECTS = $(SOURCES:.cpp=.o)
REMOVE  = wards.exe *.o

$(OUTPUT): $(OBJECTS)
	$(CC) $(CFLAGS) $(VERSION) $(OBJECTS) -o $(OUTPUT)
main.o: $(SOURCES)
	$(CC) $(VERSION) $(C) $(CFLAGS) main.cpp
wards.o: wards.cpp wards.h
	$(CC) $(VERSION) $(C) $(CFLAGS) wards.cpp
.PHONY: clean
clean:
	@rm -rf $(REMOVE)
