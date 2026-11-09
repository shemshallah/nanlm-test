#!/usr/bin/env python3
"""Execute every translated truth table in the upstream NaN0S VM.
Usage: python3 tools/check_nan0s.py ../NaN0S
No upstream files are modified except its normal build output.
"""
import pathlib, subprocess, sys, tempfile
repo = pathlib.Path(sys.argv[1]).resolve()
subprocess.run(['make', '-s', '-C', str(repo/'aarch64/android'), 'nan_cli'], check=True)
heads = [0x7ff1, 0xfff1, 0x7ff8, 0xfff8]
def word(n, payload):
    far = heads[n & 3]
    return f'{heads[n >> 2]:04X}_{payload:08X}_{((far & 255)<<8)|(far>>8):04X}'
# Execute BOOL in VM, convert low result nibble to ASCII A..P, PUTC, HALT.
with tempfile.TemporaryDirectory() as d:
    for n in range(16):
        image = pathlib.Path(d)/'truth.nan'
        image.write_text('\n'.join([
            word(15, 0x0120cccc), word(15, 0x0220cccc),
            word(15, 0x0130aaaa), word(15, 0x0230aaaa),
            word(n, 0x00120003), word(15, 0x0140000f),
            word(8, 0x00110004), word(15, 0x01400041),
            word(15, 0x03110004), word(15, 0x15100000), word(15, 0xff000000)])+'\n')
        result = subprocess.run([str(repo/'aarch64/android/nan_cli'), '--img', str(image), '--steps', '100', '--serial', '--dump-text'], capture_output=True, text=True, check=True)
        if result.stdout.strip() != chr(65+n):
            raise RuntimeError(f'BOOL {n}: {result.stdout!r} {result.stderr!r}')
print('PASS: all 16 translated Boolean instructions execute in NaN0S VM')
