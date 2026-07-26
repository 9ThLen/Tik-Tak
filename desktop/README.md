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
tiktak track song.mp3 --seconds 30    # the file, click riding its own beats
tiktak listen --seconds 30            # the room, click riding what it hears
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

### track

The Phase 4 scenario end to end: a WAV/FLAC/MP3 in, the offline analysis finds
its beat grid, and the track plays with the click exactly on its own beats —
count-in first, and `--loop A:B` to cycle a difficult phrase by bars.

```sh
tiktak track song.mp3 --count-in 4 --loop 8:12
tiktak track song.mp3 --hint 96          # manual mode: fix the tempo, find the phase
tiktak track song.mp3 --seconds 8 -o out.wav   # same callback, virtual clock
```

The grid is analysed once and cached next to the file
(`.tiktak/<content-hash>.grid`), so the second start is instant. The key is a
hash of the encoded bytes: a renamed file still hits, a re-encoded one
correctly misses, and a grid analysed under a `--hint` never masquerades as
the automatic one. `--no-cache` forces a fresh analysis.

#### Bar lines

The analysis also finds where the bar starts and how many beats it holds, and
the click accents the one:

```
song.mp3 — 184.0 s at 44100 Hz
beat grid: analysed — 372 beats at 121.0 BPM (confidence 0.81)
bar lines: 4 beats to the bar (strength 1.40, phase margin 0.80, metre margin 0.87)
```

`strength` is how far the chosen bar lines stand above the beats around them.
The two margins are separate because they answer separate questions, and
conflating them hid real errors: the **phase margin** says which beat starts the
bar is settled, and the **metre margin** says no other bar length fits nearly as
well. A piece read in three can be completely unambiguous about where its bars
start — a large phase margin — while four fits it just as well, because every
rival the phase margin weighs has already accepted three.

Both have to clear their thresholds before the accent is used. When they do not,
the harness says so and the click stays even:

```
bar lines: 3 beats to the bar (strength 0.44, phase margin 0.04, metre margin 0.02) — too close to call
no accent — every beat clicks the same
```

Counting fours from the first beat instead would be an arbitrary accent worn
with the same confidence as a detected one, and a player following it phrases to
a bar line that is not there.

`--beats N` overrides the bar *length*. The number you type is an assertion about
the music, the same way `--hint` is. It says nothing about which beat starts the
bar, so when the analysis agreed about the metre the phase still comes from the
audio:

```
bar lines: 4 beats to the bar (strength 1.40, phase margin 0.80, metre margin 0.87)
bar starts on beat 1, from the audio
```

and when it did not, the harness says it is overruling the audio and counts from
no invented first beat as the downbeat:

```
using --beats 3 over the 4 the audio suggests — phase unknown, every beat clicks the same
```

The thresholds are provisional. `research/eval/downbeat_benchmark.py` is what
sets them: it sweeps both, reports coverage against the wrong-accent rate, and
picks the most generous pair inside a budget. See `research/eval/README.md`.

The click is not latency-compensated against the track, on purpose: both leave
through the same device buffer, so a click written on the beat's sample arrives
with it whatever the output latency is. `--latency-ms` is therefore not needed
here — it exists for the standalone metronome, whose reference is the wall
clock rather than a track in the same buffer.

### listen

The microphone path: the tracker follows what the room is playing and the click
goes out on its beat. This is the mode the harness pays for itself on — live
sound in a real room, a hundred attempts an hour instead of a dozen a day.

```sh
tiktak listen --seconds 30 --latency-ms 24     # follow the room
tiktak listen --hint 96                        # start from a tempo instead of searching
tiktak listen --manual 96                      # hold 96, take only the phase from the room
tiktak listen song.mp3 -o heard.wav            # no microphone: drive it from a file
```

**`--latency-ms` here is the *round trip*, not the output latency** — the figure
`measure` prints. The tracker's clock is the capture stream's, so a click has to
leave early by the whole way out and back to be heard on the beat. Left at zero,
the click is late by exactly the device's round trip.

Given a file, `listen` drives the same tracker from it against a virtual clock
and writes the room and the click it played over it. That is not a lesser mode:
it is what makes the microphone path testable on a machine with no microphone
(every CI runner is one), and the only way to feed the tracker the same input
twice. It gates its own click there too, exactly as it would through a
microphone, because the point is to exercise the path rather than to flatter it.

What comes back is a tempo, a confidence and the counters. Confidence is the one
to watch: it is the product of how much the tracker's particle cloud agrees with
itself and how much of the room's onset energy keeps landing where it predicted,
so silence, noise and a change of song all pull it down — and below the lock
threshold the metronome coasts at the last tempo it was sure of rather than
lunging at whatever it hears next.

#### `--manual N` — the tempo is yours, the phase is the room's

With `--manual` the tempo stops being a question. The click holds the BPM given,
waits for the room to start, falls in on its phase, and then keeps going whether
the room does or not — because the tempo was never the room's to take away. It
follows a player drifting within about 2% of the number set and free-runs
against anything further off.

Finding a phase at a known tempo is a far smaller problem than finding a tempo,
so this mode works on material the automatic one cannot follow at all. The other
half of that bargain is that it **refuses**: asked for a beat the room does not
contain, it plays nothing and says so, rather than clicking somewhere and calling
it synchronised.

```
$ tiktak listen click_120.mp3 --manual 120 --seconds 10
live tracker: manual 120.0 BPM, synchronised to the room

$ tiktak listen click_120.mp3 --manual 137 --seconds 10
live tracker: manual 137.0 BPM, still listening for a beat to fall in with
  beats played        0
```

Live, the progress line shows `listening` or `in sync` and a phase figure — how
concentrated the room's onsets are at one point in the bar, which is the meter a
UI would put behind "listening…".

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
