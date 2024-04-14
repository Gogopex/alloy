# Compiler settings - Can change to clang if preferred
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -std=c11 -Iinclude

# Linker flags
LDFLAGS = -Llib -lcmt_lib -framework Metal

# Source directory and executable name
SRC_DIR = src
EXEC = Alloy

# Find all source files in the source directory, sorted by most recently modified
SOURCES = $(shell find $(SRC_DIR) -name '*.c' | sort -k 1nr | cut -f2-)

# Default target
all: $(EXEC)

# Link the program
$(EXEC): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Clean up
clean:
	rm -f $(EXEC)

# Phony targets
.PHONY: all clean
