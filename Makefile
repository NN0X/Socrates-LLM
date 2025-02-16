CXX = clang++
CFLAGS = -O3 -Wall -Wextra -Wpedantic -I./include/
LDFLAGS = -L./lib/cicero -lcicero
SRC = src
OUT = main

all:
	rm -f -r build
	mkdir build
	$(CXX) $(CFLAGS) $(SRC)/*.cpp -o build/$(OUT) $(LDFLAGS) 2> build/make.log
	cp -f -r resources build/resources
	cp LICENSE.md build/LICENSE.md
	cp README.md build/README.md
	cp NOTICE.md build/NOTICE.md
	@echo "Build complete. Executable is located at build/$(OUT)"
