#!/usr/bin/env python3
"""exeinfo.py - read the MZ header of a DOS executable.

Written because "it linked" is not the same as "it is a valid DOS program":
the header says how much memory the program needs before DOS will even
load it, which is exactly the thing that goes wrong when a 16-bit build
asks for megabytes.
"""
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "build_dos/skynet.exe"
data = open(path, "rb").read()

print("file       : %s" % path)
print("size       : %d bytes" % len(data))

if data[:2] != b"MZ":
    print("ERROR      : not an MZ executable")
    sys.exit(1)

(last_page, pages, relocs, hdr_paras, min_alloc, max_alloc,
 ss, sp, cs, ip, csum, ip_field) = struct.unpack("<HHHHHHHHHHHH", data[2:26])

image = (pages - 1) * 512 + (last_page if last_page else 512) - hdr_paras * 16

print("image      : %d bytes of loadable code+data" % image)
print("header     : %d bytes" % (hdr_paras * 16))
print("relocs     : %d" % relocs)
print("min alloc  : %d bytes (extra heap DOS must provide)" % (min_alloc * 16))
print("max alloc  : %d bytes" % (max_alloc * 16))
print("entry      : %04X:%04X" % (cs, ip))
print("stack      : %04X:%04X" % (ss, sp))

need = image + min_alloc * 16
print("total need : ~%d bytes (%.0f KB)" % (need, need / 1024.0))
if need > 640 * 1024:
    print("WARNING    : exceeds 640 KB, will not load in real mode")
else:
    print("fits       : yes, in conventional memory")
