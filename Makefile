CC = g++
CFLAGS = -g -Wall
C = -c
VERSION = --std=c++11
OUTPUT  = wards.exe
SOURCES = main.cc hierarchical_clustering.cc ward.cc
OBJECTS = $(SOURCES:.cpp=.o)
REMOVE  = wards.exe *.o

$(OUTPUT): $(OBJECTS)
	$(CC) $(CFLAGS) $(VERSION) $(OBJECTS) -o $(OUTPUT)
main.o: $(SOURCES)
	$(CC) $(VERSION) $(C) $(CFLAGS) main.cc
ward.o: ward.cc ward.h
	$(CC) $(VERSION) $(C) $(CFLAGS) ward.cc
hierarchical_clustering.o: hierarchical_clustering.cc hierarchical_clustering.h
	$(CC) $(VERSION) $(C) $(CFLAGS) hierarchical_clustering.cc
.PHONY: clean
clean:
	@rm -rf $(REMOVE)
