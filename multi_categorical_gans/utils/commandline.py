from __future__ import print_function

import signal
import sys


def parse_int_list(comma_separated_ints):
    if comma_separated_ints is None or comma_separated_ints == "":
        return []
    return [int(i) for i in comma_separated_ints.split(",")]


class DelayedKeyboardInterrupt(object):

    SIGNALS = [signal.SIGINT, signal.SIGTERM]

    def __init__(self):
        self.signal_received = {}
        self.old_handler = {}

    def __enter__(self):
        self.signal_received = {}
        self.old_handler = {}
        for sig in self.SIGNALS:
            self.old_handler[sig] = signal.signal(sig, self.handler)

    def handler(self, sig, frame):
        self.signal_received[sig] = frame
        print('Delaying received signal', sig)

    def __exit__(self, type, value, traceback):
        for sig in self.SIGNALS:
            signal.signal(sig, self.old_handler[sig])
        for sig, frame in self.signal_received.items():
            old_handler = self.old_handler[sig]
            print('Resuming received signal', sig)
            if callable(old_handler):
                old_handler(sig, frame)
            elif old_handler == signal.SIG_DFL:
                sys.exit(0)
        self.signal_received = {}
        self.old_handler = {}
