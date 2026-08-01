#pragma once

namespace tiktak::desktop {

// Keys as they are pressed, without waiting for a line.
//
// A terminal normally hands over a whole line at once, which for this purpose
// destroys the measurement: every tap in a phrase would arrive at the moment
// Enter was pressed. So the terminal is put in a mode where a keystroke is
// readable immediately, and put back on the way out — including when the
// command returns early, which is why this is a class and not two functions.
//
// This is the one piece of the harness that has to know what a terminal is, and
// it lives here rather than in the command so the command reads as the
// experiment it is. Nothing in core/ may ever include it.
class KeyReader {
public:
    KeyReader();
    ~KeyReader();

    KeyReader(const KeyReader&) = delete;
    KeyReader& operator=(const KeyReader&) = delete;

    // The next key, or -1 when none is waiting. Never blocks: the caller is in
    // a loop that also has to notice the track ending.
    int poll();

private:
    bool restore_ = false;
#if !defined(_WIN32)
    // Opaque storage for the previous terminal settings, so this header does
    // not drag <termios.h> into every file that includes it.
    alignas(long) unsigned char saved_[64] = {};
#endif
};

}  // namespace tiktak::desktop
