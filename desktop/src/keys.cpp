#include "keys.hpp"

#if defined(_WIN32)

#include <conio.h>

namespace tiktak::desktop {

// Windows hands over keystrokes without any terminal mode change, so there is
// nothing to save and nothing to restore.
KeyReader::KeyReader() = default;
KeyReader::~KeyReader() = default;

int KeyReader::poll() {
    if (!_kbhit()) return -1;
    const int key = _getch();
    // Function and arrow keys arrive as two bytes. Read and discard the second
    // so it is not delivered later as a phantom tap.
    if (key == 0 || key == 0xE0) {
        if (_kbhit()) _getch();
        return -1;
    }
    return key;
}

}  // namespace tiktak::desktop

#else

#include <termios.h>
#include <unistd.h>

#include <cstring>

namespace tiktak::desktop {
namespace {
static_assert(sizeof(termios) <= 64, "saved_ is too small for this platform");
}

KeyReader::KeyReader() {
    termios current{};
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &current) != 0) return;
    std::memcpy(saved_, &current, sizeof(current));
    restore_ = true;

    termios raw = current;
    // Canonical mode is what buffers until Enter; echo would print the taps
    // over the running count.
    raw.c_lflag = static_cast<tcflag_t>(raw.c_lflag & ~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

KeyReader::~KeyReader() {
    if (!restore_) return;
    termios previous{};
    std::memcpy(&previous, saved_, sizeof(previous));
    tcsetattr(STDIN_FILENO, TCSANOW, &previous);
}

int KeyReader::poll() {
    unsigned char c = 0;
    const ssize_t got = read(STDIN_FILENO, &c, 1);
    return got == 1 ? static_cast<int>(c) : -1;
}

}  // namespace tiktak::desktop

#endif
