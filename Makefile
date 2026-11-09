CXX ?= c++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude
.PHONY: all test validate clean
all: nanlm
nanlm: src/main.cpp include/nanlm/universe.hpp include/nanlm/wordnet.hpp include/nanlm/byte_model.hpp include/nanlm/recurrent_byte.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/main.cpp -o $@
test_universe: tests/test_universe.cpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_universe.cpp -o $@
test_wordnet: tests/test_wordnet.cpp include/nanlm/universe.hpp include/nanlm/wordnet.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_wordnet.cpp -o $@

test_teacher: tests/test_teacher.cpp include/nanlm/teacher.hpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_teacher.cpp -o $@
test_quaternion: tests/test_quaternion.cpp include/nanlm/quaternion.hpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_quaternion.cpp -o $@
test_multiloss: tests/test_multiloss.cpp include/nanlm/multiloss.hpp include/nanlm/teacher.hpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_multiloss.cpp -o $@
test_learning_validation: tests/test_learning_validation.cpp include/nanlm/teacher.hpp include/nanlm/universe.hpp include/nanlm/quaternion.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_learning_validation.cpp -o $@
test_byte_model: tests/test_byte_model.cpp include/nanlm/byte_model.hpp include/nanlm/teacher.hpp include/nanlm/universe.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_byte_model.cpp -o $@
test_recurrent_byte: tests/test_recurrent_byte.cpp include/nanlm/recurrent_byte.hpp include/nanlm/byte_model.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_recurrent_byte.cpp -o $@
validate: test_learning_validation
	./test_learning_validation
test: test_universe test_wordnet test_teacher test_quaternion test_multiloss test_byte_model test_recurrent_byte
	./test_universe
	./test_wordnet
	./test_teacher
	./test_quaternion
	./test_multiloss
	./test_byte_model
	./test_recurrent_byte
clean:
	rm -f nanlm test_universe test_wordnet test_teacher test_quaternion test_multiloss test_learning_validation test_byte_model test_recurrent_byte
