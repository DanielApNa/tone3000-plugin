// Per-lane noise gate
//
// Stereo chain mode runs two different amps, one per lane, and each wants
// its own gate. These pin:
//
//   - NoiseGate thresholds are per channel: one channel can pass a signal
//     another channel's threshold closes on,
//   - in stereo chain mode each lane follows its own gate, and the gate
//     link restores one shared gate on both lanes,
//   - sessions saved before the Right gate existed restore it from the
//     main gate (which then gated both lanes), so old projects sound the
//     same.
#include "NoiseGate.h"
#include "Processor.h"

#include <gtest/gtest.h>

#include <cmath>

namespace {

constexpr double kFs = 48000.0;
constexpr int kBlock = 512;

// Peak of the last block after ~0.5 s of a steady 440 Hz sine at `peak`,
// so the envelope, hold and release have all settled.
std::array<float, 2> settledPeaks(NoiseGate& gate, float peak) {
  juce::AudioBuffer<float> buffer(2, kBlock);
  int n = 0;
  std::array<float, 2> out{};
  for (int block = 0; block < 48; ++block) {
    for (int i = 0; i < kBlock; ++i, ++n) {
      const float x =
          peak * std::sin(juce::MathConstants<float>::twoPi * 440.0f * static_cast<float>(n) /
                          static_cast<float>(kFs));
      buffer.setSample(0, i, x);
      buffer.setSample(1, i, x);
    }
    gate.process(buffer);
  }
  for (int ch = 0; ch < 2; ++ch)
    out[static_cast<size_t>(ch)] = buffer.getMagnitude(ch, 0, kBlock);
  return out;
}

}  // namespace

TEST(GateTest, ThresholdsArePerChannel) {
  NoiseGate gate;
  gate.prepare(kFs);
  gate.setThresholdDb(0, -60.0f);  // well under the -30 dB signal: open
  gate.setThresholdDb(1, -10.0f);  // well over it: closed

  const float peak = juce::Decibels::decibelsToGain(-30.0f);
  const auto out = settledPeaks(gate, peak);
  EXPECT_NEAR(out[0], peak, peak * 0.05f) << "open channel must pass the signal untouched";
  EXPECT_LT(out[1], peak * 0.1f) << "closed channel must attenuate by well over 20 dB";
}

TEST(GateTest, SharedThresholdStillCoversBothChannels) {
  NoiseGate gate;
  gate.prepare(kFs);
  gate.setThresholdDb(-10.0f);

  const float peak = juce::Decibels::decibelsToGain(-30.0f);
  const auto out = settledPeaks(gate, peak);
  EXPECT_LT(out[0], peak * 0.1f);
  EXPECT_LT(out[1], peak * 0.1f);
}

TEST(GateTest, StereoLanesGateIndependentlyUnlessLinked) {
  // Empty stereo chains, hard-panned (default): output L is the Left lane's
  // gated input, output R the Right lane's. A -30 dB tone passes a -60 dB
  // gate and closes a -10 dB one.
  TONE3000Processor proc;
  proc.setPlayConfigDetails(2, 2, kFs, kBlock);
  proc.setStereoMode(true);
  proc.prepareToPlay(kFs, kBlock);
  auto setDb = [&](const char* id, float db) {
    auto* p = proc.parameters.getParameter(id);
    p->setValueNotifyingHost(p->convertTo0to1(db));
  };
  setDb("gateThreshold", -60.0f);
  setDb("gateThresholdRight", -10.0f);

  const float peak = juce::Decibels::decibelsToGain(-30.0f);
  juce::MidiBuffer midi;
  int n = 0;
  const auto settledOut = [&] {
    juce::AudioBuffer<float> buffer(2, kBlock);
    for (int block = 0; block < 48; ++block) {
      for (int i = 0; i < kBlock; ++i, ++n) {
        const float x = peak * std::sin(juce::MathConstants<float>::twoPi * 997.0f *
                                        static_cast<float>(n) / static_cast<float>(kFs));
        buffer.setSample(0, i, x);
        buffer.setSample(1, i, x);
      }
      proc.processBlock(buffer, midi);
    }
    return std::array<float, 2>{buffer.getMagnitude(0, 0, kBlock),
                                buffer.getMagnitude(1, 0, kBlock)};
  };

  auto out = settledOut();
  EXPECT_NEAR(out[0], peak, peak * 0.05f) << "Left lane gate must stay open";
  EXPECT_LT(out[1], peak * 0.1f) << "Right lane must follow its own, closed gate";

  // Linked: the main gate covers both lanes, the Right params are ignored.
  proc.parameters.getParameter("gateLinked")->setValueNotifyingHost(1.0f);
  out = settledOut();
  EXPECT_NEAR(out[0], peak, peak * 0.05f);
  EXPECT_NEAR(out[1], peak, peak * 0.05f) << "linked gate must apply the main threshold";
}

TEST(GateTest, RightGateIsIndependentParameterState) {
  juce::MemoryBlock state;
  {
    TONE3000Processor a;
    auto* main = a.parameters.getParameter("gateThreshold");
    auto* right = a.parameters.getParameter("gateThresholdRight");
    main->setValueNotifyingHost(main->convertTo0to1(-40.0f));
    right->setValueNotifyingHost(right->convertTo0to1(-20.0f));
    a.parameters.getParameter("gateEnabledRight")->setValueNotifyingHost(0.0f);
    a.getStateInformation(state);
  }

  TONE3000Processor b;
  b.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
  EXPECT_NEAR(b.parameters.getRawParameterValue("gateThreshold")->load(), -40.0f, 1e-3f);
  EXPECT_NEAR(b.parameters.getRawParameterValue("gateThresholdRight")->load(), -20.0f, 1e-3f);
  EXPECT_EQ(b.parameters.getRawParameterValue("gateEnabled")->load(), 1.0f);
  EXPECT_EQ(b.parameters.getRawParameterValue("gateEnabledRight")->load(), 0.0f);
}

TEST(GateTest, LegacySessionSeedsRightGateFromMainGate) {
  // A save from this build with the Right gate params stripped out looks
  // exactly like a session saved before they existed.
  juce::MemoryBlock saved;
  {
    TONE3000Processor old;
    auto* main = old.parameters.getParameter("gateThreshold");
    main->setValueNotifyingHost(main->convertTo0to1(-35.0f));
    old.parameters.getParameter("gateEnabled")->setValueNotifyingHost(0.0f);
    old.getStateInformation(saved);
  }
  juce::ValueTree tree = juce::ValueTree::readFromData(
      static_cast<const char*>(saved.getData()) + 4, saved.getSize() - 4);
  ASSERT_TRUE(tree.isValid());
  juce::ValueTree params = tree.getChildWithName("PARAMETERS");
  ASSERT_TRUE(params.isValid());
  for (const char* id : {"gateThresholdRight", "gateEnabledRight"}) {
    const auto child = params.getChildWithProperty("id", id);
    ASSERT_TRUE(child.isValid());
    params.removeChild(child, nullptr);
  }
  juce::MemoryBlock legacy;
  {
    juce::MemoryOutputStream out(legacy, false);
    out.write("T3KB", 4);
    tree.writeToStream(out);
  }

  TONE3000Processor proc;
  proc.setStateInformation(legacy.getData(), static_cast<int>(legacy.getSize()));
  EXPECT_NEAR(proc.parameters.getRawParameterValue("gateThresholdRight")->load(), -35.0f, 1e-3f);
  EXPECT_EQ(proc.parameters.getRawParameterValue("gateEnabledRight")->load(), 0.0f);
}
