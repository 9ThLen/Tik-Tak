# tiktak-desktop

The development harness. See
[`docs/adr/0003-windows-host-no-mac.md`](../docs/adr/0003-windows-host-no-mac.md)
for why it exists: the target is an iPhone, the machine is a Windows box, and a
metronome is an app about milliseconds. Measuring milliseconds through a
TestFlight cycle is a dozen attempts a day instead of a hundred, so the timing
is debugged here against real devices and the phone gets a thin shell over the
same core.

**Nothing here may hold logic the phone also needs.** What is here is device
access, argument parsing and measurement — the parts that cannot be portable.
The metronome itself is `core/src/render/metronome`, and this drives it with the
same call the phone will.

## Build

```sh
cmake -S desktop -B desktop/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build desktop/build
```

It builds `core/` itself, so there is nothing to install first.
[miniaudio](https://github.com/mackron/miniaudio) is fetched at configure time,
pinned to a tag. Its backends are loaded at runtime, so no audio SDK is a build
dependency — which is how this compiles on a CI runner with no sound card.

## Commands

```sh
tiktak devices                        # what the machine has
tiktak play --bpm 120 --seconds 30    # a metronome, on a real device
tiktak render --bpm 137 -o out.wav    # the same thing, to a file, no device
tiktak measure --seconds 30           # round trip and jitter, measured
```

`--latency-ms` is the output latency to compensate. Measure it before trusting
it; the driver's own figure is a starting point, not the truth.

### render

The same callback a device drives, against a virtual clock, written to a WAV.

This is not a lesser mode — it is what makes the metronome's timing testable
without hardware, on any machine and on every push. `desktop/tools/check_render.py`
reads the file back and asserts every beat is on its sample; CI runs it at
137 BPM, a tempo that does not divide the sample rate, because that is the case
that catches a click rounding into the wrong buffer.

```sh
tiktak render --bpm 137 --seconds 60 -o out.wav
python3 desktop/tools/check_render.py out.wav --bpm 137 --start 0.5
```

### measure

Plays the metronome and records the input at the same time, then finds each
click in the recording and reports how long it took to come back.

The output has to reach the input for this to mean anything: a loopback cable,
or speakers and a microphone in a quiet room with the volume up.

Two numbers come out, and the second is the one that matters:

- **round trip** — the mean. This is latency, and latency can be compensated;
  feed it back in through `--latency-ms`.
- **jitter** — the spread. This cannot be compensated, and it is what a player
  hears as a metronome that will not sit still.

Nothing is compensated during the measurement itself, so the offset that comes
back is the whole round trip and nothing else.

## What the run reports

Every command ends with the metronome's counters, and every one of them should
be zero:

| | |
|---|---|
| beats given up as late | the grid could not place a beat in time |
| clicks past their buffer | a click arrived after the buffer it belonged in |
| clicks refused | the queue was full — the lookahead is bigger than expected |
| clicks cut short | every voice was busy |
| buffers out of sequence | the device dropped or repeated a buffer |

A metronome quietly dropping every tenth beat sounds like a metronome until
something counts, which is why they are counted and why a run that is not clean
exits non-zero.

## Known limits

The harness validates logic and algorithms, not iPhone numbers. WASAPI, ALSA and
CoreAudio latencies have nothing to do with what iOS will do, so absolute
latency still has to be measured on the device — just far less often, and with
something specific to look for. Haptics cannot be checked here at all.
