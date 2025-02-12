CXX = clang++
CFLAGS = -O3 -Wall -Wextra -Wpedantic -std=c++17
SRC = src
OUT = main

all:
	rm -f -r build
	mkdir build
	$(CXX) $(CFLAGS) $(SRC)/*.cpp -o build/$(OUT) 2> build/make.log
	cp -f -r resources build/resources
	cp LICENSE.md build/LICENSE.md
	cp README.md build/README.md
	cp -f -r licenses build/licenses
	@echo "Build complete. Executable is located at build/$(OUT)"
