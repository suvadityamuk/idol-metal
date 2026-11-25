# Native Idol port (Objective-C++ / Metal / MPS). No Python, no libtorch in the Idol-model path.
CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -fno-objc-arc -Wall -Wno-nullability-completeness -Wno-unused-variable -Wno-sign-compare -ffp-contract=off
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders
BUILD = build
RT = src/runtime.mm src/ops.mm
HDR = src/runtime.h src/ops.h src/safetensors.h
SAMPLER = src/sampler.mm

all: $(BUILD)/kerneltest $(BUILD)/idol_sample

$(BUILD)/idol_sample: tools/idol_sample.mm $(RT) $(SAMPLER) $(HDR) src/sampler.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tools/idol_sample.mm $(RT) $(SAMPLER) $(FRAMEWORKS) -o $@

$(BUILD)/kerneltest: tests/kerneltest.mm $(RT) $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/kerneltest.mm $(RT) $(FRAMEWORKS) -o $@

clean:
	rm -rf $(BUILD)
