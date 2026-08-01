// The desktop harness.
//
// Its reason to exist is in docs/adr/0003-windows-host-no-mac.md: the target is
// an iPhone, the development machine is a Windows box, and a metronome is an
// app about milliseconds. Measuring milliseconds through a TestFlight cycle
// means a dozen attempts a day instead of a hundred, so the algorithms are
// debugged here, against real devices, and the phone gets a thin shell over the
// same core.
//
// Nothing in desktop/ may hold logic the phone also needs. What is here is
// device access, argument parsing and measurement — the parts that cannot be
// portable.
#include <cstdio>
#include <string>
#include <vector>

#include "commands.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        tiktak::desktop::printUsage();
        return 2;
    }

    const std::string command = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);

    if (command == "help" || command == "--help" || command == "-h") {
        tiktak::desktop::printUsage();
        return 0;
    }

    if (command == "devices") return tiktak::desktop::cmdDevices();

    tiktak::desktop::Options options;
    std::string error;
    if (!tiktak::desktop::parseOptions(args, options, error)) {
        std::fprintf(stderr, "tiktak: %s\n\n", error.c_str());
        tiktak::desktop::printUsage();
        return 2;
    }

    if (command == "render") return tiktak::desktop::cmdRender(options);
    if (command == "play") return tiktak::desktop::cmdPlay(options);
    if (command == "measure") return tiktak::desktop::cmdMeasure(options);
    if (command == "track") return tiktak::desktop::cmdTrack(options);
    if (command == "listen") return tiktak::desktop::cmdListen(options);
    if (command == "tap") return tiktak::desktop::cmdTap(options);

    std::fprintf(stderr, "tiktak: unknown command '%s'\n\n", command.c_str());
    tiktak::desktop::printUsage();
    return 2;
}
