CC = g++
CFLAGS = -g -Wall
C = -c
VERSION = -std=c++14
OUTPUT  = wards.exe
SOURCES = main.cpp hc.cpp ward.cpp
OBJECTS = $(SOURCES:.cpp=.o)
REMOVE  = wards.exe *.o

$(OUTPUT): $(OBJECTS)
	$(CC) $(CFLAGS) $(VERSION) $(OBJECTS) -o $(OUTPUT)
main.o: $(SOURCES)
	$(CC) $(VERSION) $(C) $(CFLAGS) main.cpp
ward.o: ward.cpp ward.h
	$(CC) $(VERSION) $(C) $(CFLAGS) ward.cpp
hc.o: hc.cpp hc.h
	$(CC) $(VERSION) $(C) $(CFLAGS) hc.cpp
.PHONY: clean
clean:
	@rm -rf $(REMOVE)
