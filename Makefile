CXX ?= c++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude
.PHONY: all test validate validate-harmony clean
all: nanlm
nanlm: src/main.cpp include/nanlm/universe.hpp include/nanlm/wordnet.hpp $(wildcard include/nanlm/*.hpp) include/nanlm/nan16.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/main.cpp -o $@
nanlm-diagnostics: src/diagnostics_main.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/diagnostics_main.cpp -o $@
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
validate: test_learning_validation
	./test_learning_validation
test_harmony: tests/test_harmony.cpp $(wildcard include/nanlm/*.hpp) include/nanlm/nan16.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_nan16: tests/test_nan16.c include/nanlm/nan16.h
	$(CC) $(CPPFLAGS) -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror $< -lm -o $@

test: test_byte_model test_recurrent_byte test_unified test_unified_wave test_temporal_credit test_wave_bptt test_sequence_bptt test_frequency_bptt test_hypermatrix test_triad_waveformer test_episodic_memory test_associative_memory test_hybrid_byte test_nan16 test_harmony test_universe test_fabric_integrity test_wordnet test_teacher test_quaternion test_multiloss
	./test_nan16
	./test_harmony
	python3 tests/test_context_lm.py
	./test_universe
	./test_fabric_integrity
	./test_wordnet
	./test_teacher
	./test_quaternion
	./test_multiloss
	./test_byte_model
	./test_recurrent_byte
	./test_unified
	./test_hybrid_byte
	./test_unified_wave
	./test_temporal_credit
	./test_wave_bptt
	./test_sequence_bptt
	./test_frequency_bptt
	./test_hypermatrix
	./test_triad_waveformer
	./test_episodic_memory
	./test_associative_memory
clean:
	rm -f test_frequency_bptt test_sequence_bptt test_wave_bptt test_fabric_integrity test_temporal_credit nanlm-diagnostics test_hypermatrix test_triad_waveformer test_episodic_memory test_associative_memory test_adaptive_triads test_unified_wave test_hybrid_byte test_unified test_recurrent_byte test_byte_model test_nan16 validate_harmony test_harmony nanlm test_universe test_wordnet test_teacher test_quaternion test_multiloss test_learning_validation

validate_harmony: tests/validate_harmony.cpp $(wildcard include/nanlm/*.hpp) include/nanlm/nan16.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
validate-harmony: validate_harmony
	./validate_harmony

test_recurrent_byte: tests/test_recurrent_byte.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_unified: tests/test_unified.cpp $(wildcard include/nanlm/*.hpp) include/nanlm/nan16.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_unified_wave: tests/test_unified_wave.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_hybrid_byte: tests/test_hybrid_byte.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_hypermatrix: tests/test_hypermatrix.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_fabric_integrity: tests/test_fabric_integrity.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_temporal_credit: tests/test_temporal_credit.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_wave_bptt: tests/test_wave_bptt.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_sequence_bptt: tests/test_sequence_bptt.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_frequency_bptt: tests/test_frequency_bptt.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_triad_waveformer: tests/test_triad_waveformer.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_adaptive_triads: tests/test_adaptive_triads.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# Source-attributed exact-recall layer over the same 8 MiB mapped root.
test_episodic_memory: tests/test_episodic_memory.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# Fixed-seed control separating phase, edit-distance, and 85-triad hotspots.
test_associative_memory: tests/test_associative_memory.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# HW4 full-corpus batch geometry learner; one C++ training pass, no BPTT.
hw4-onepass: tools/hw4_onepass_native.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# HW5 Gutenberg full-document byte-transition geometric learner and read-only eval.
hw5-onepass: tools/hw5_geometric_native.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

# HW5 context: trained ONLY on Gutenberg train titles; native NaN16 mirror bank + sparse exact triples.
hw5-context: tools/hw5_context_mirror_native.cpp $(wildcard include/nanlm/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
