CXX ?= c++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude
.PHONY: all test clean
all: nanlm
nanlm: src/main.cpp include/nanlm/universe.hpp include/nanlm/wordnet.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/main.cpp -o $@
test_universe: tests/test_universe.cpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_universe.cpp -o $@
test_wordnet: tests/test_wordnet.cpp include/nanlm/universe.hpp include/nanlm/wordnet.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_wordnet.cpp -o $@

test_teacher: tests/test_teacher.cpp include/nanlm/teacher.hpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_teacher.cpp -o $@
test_quaternion: tests/test_quaternion.cpp include/nanlm/quaternion.hpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_quaternion.cpp -o $@
test: test_universe test_wordnet test_teacher test_quaternion
	./test_universe
	./test_wordnet
	./test_teacher
	./test_quaternion
clean:
	rm -f nanlm test_universe test_wordnet test_teacher test_quaternion
