# Standalone Makefile for AArch64HeteroOpt
# Usage: make [test|test_asan|test_ubsan|clean|help]
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wunreachable-code -Wshadow -Iinclude

TEST_SRCS := test_target_profile test_features test_syntax
TEST_BINS := $(TEST_SRCS)

.PHONY: all test test_asan test_ubsan clean plugin help

all: test

# ── Normal test build ─────────────────────────────────────────────────────────
$(TEST_BINS): %: test/%.cpp include/*.h
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
	  echo "--- $$t ---"; ./$$t || exit 1; \
	done
	@echo "=== All standalone tests passed ==="

# ── ASan / UBSan builds ───────────────────────────────────────────────────────
test_asan: CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -O1
test_asan: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "--- $$t (asan) ---"; ./$$t || exit 1; done

test_ubsan: CXXFLAGS += -fsanitize=undefined -O1
test_ubsan: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "--- $$t (ubsan) ---"; ./$$t || exit 1; done

# ── LLVM plugin build hint ────────────────────────────────────────────────────
plugin:
	@echo "Building the LLVM plugin requires LLVM dev headers."
	@echo "  cmake -B build -DLLVM_DIR=\$$(llvm-config --cmakedir) ."
	@echo "  cmake --build build"

clean:
	rm -f $(TEST_BINS)
	rm -rf build/

help:
	@echo "Targets:"
	@echo "  test        Build and run all standalone tests"
	@echo "  test_asan   Same with AddressSanitizer"
	@echo "  test_ubsan  Same with UBSanitizer"
	@echo "  plugin      Print LLVM plugin build instructions"
	@echo "  clean       Remove build artifacts"
