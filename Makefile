# Compiler settings - Using clang for Objective-C support
CC = clang

# Compiler flags
CFLAGS = -Wall -Wextra -std=c11 -Iinclude/cmt

# Linker flags
LDFLAGS = -Llib -lcmt -lobjc -framework Metal -framework Foundation -framework CoreGraphics

SRC_DIR = src
EXEC = alloy

SOURCES = src/alloy.c src/error_handling.m

# Default target
all: $(EXEC)

# Link the program
$(EXEC): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Clean up
clean:
	rm -f $(EXEC) 
	rm -rf $(EXEC).dSYM

debug: CFLAGS += -g
debug: $(EXEC)

# Phony targets
.PHONY: all clean
