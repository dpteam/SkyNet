#!/usr/bin/env python3
"""Generate the SkyNet logo from the prime sequence.

The logo is not drawn, it is computed: nodes are primes placed on a
spiral, and the radius of each node encodes how far along the prime
series it is. The emblem is therefore a literal picture of the Prime
Clock - the same sequence sk_clock.c emits.

Run:  python tools/make_logo.py
Output: assets/skynet-logo.svg, assets/skynet-icon.svg
"""
import math
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def sieve(limit):
    flags = bytearray([1]) * limit
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i::i] = bytearray(len(flags[i * i::i]))
    return [i for i, f in enumerate(flags) if f]


def spiral_nodes(primes, count, radius_max, turns):
    """Place primes on an Archimedean spiral: r = a * theta."""
    nodes = []
    for i, p in enumerate(primes[:count]):
        t = i / max(1, count - 1)
        theta = t * turns * 2.0 * math.pi
        r = radius_max * t
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        # node size grows with the prime's magnitude, log-scaled so the
        # late (large) primes do not swamp the early ones
        size = 0.9 + 2.6 * (math.log(p) / math.log(primes[count - 1]))
        nodes.append((x, y, size, p, i))
    return nodes


def is_superprime(primes, p):
    """A super-prime is p_n where n is itself prime (OEIS A006450)."""
    try:
        idx = primes.index(p) + 1
    except ValueError:
        return False
    return idx in primes


def build_svg(primes, size, with_text):
    cx = cy = size / 2.0
    radius_max = size * 0.40
    count = 260
    nodes = spiral_nodes(primes, count, radius_max, 3.4)

    green = "#00ff66"
    dim = "#0a7a35"
    parts = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" role="img" aria-label="SkyNet">'
    )
    parts.append(f'<rect width="{size}" height="{size}" fill="#000000"/>')

    # faint radial glow behind the spiral
    parts.append(
        f'<defs><radialGradient id="g" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="{green}" stop-opacity="0.16"/>'
        f'<stop offset="60%" stop-color="{green}" stop-opacity="0.03"/>'
        f'<stop offset="100%" stop-color="{green}" stop-opacity="0"/>'
        f'</radialGradient></defs>'
    )
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius_max * 1.5}" fill="url(#g)"/>')

    # the spiral thread itself
    thread = []
    for x, y, _, _, _ in nodes:
        thread.append(f"{cx + x:.2f},{cy + y:.2f}")
    parts.append(
        f'<polyline points="{" ".join(thread)}" fill="none" stroke="{dim}" '
        f'stroke-width="{max(0.6, size * 0.0016):.2f}" stroke-linejoin="round"/>'
    )

    # nodes, super-primes brighter and ringed
    for x, y, s, p, i in nodes:
        px, py = cx + x, cy + y
        if is_superprime(primes, p):
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{s * 2.4:.2f}" '
                         f'fill="none" stroke="{green}" stroke-width="{max(0.5, size*0.0012):.2f}" '
                         f'stroke-opacity="0.55"/>')
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{s * 1.15:.2f}" fill="#d6ffe8"/>')
        else:
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{s:.2f}" fill="{green}" '
                         f'fill-opacity="{0.45 + 0.5 * (i / len(nodes)):.2f}"/>')

    if with_text:
        # pixel-ish wordmark: letters drawn as blocks so it reads at any size
        fs = size * 0.115
        ty = size * 0.925
        parts.append(
            f'<text x="{cx}" y="{ty}" text-anchor="middle" '
            f'font-family="Consolas,\'Courier New\',monospace" font-weight="bold" '
            f'font-size="{fs:.2f}" letter-spacing="{fs * 0.18:.2f}" fill="{green}">'
            f'SKYNET</text>'
        )

    # scanlines
    parts.append(
        f'<g opacity="0.10">'
        f'<rect width="{size}" height="{size}" fill="url(#s)"/>'
        f'</g>'
    )
    parts.append(
        f'<defs><pattern id="s" width="4" height="4" patternUnits="userSpaceOnUse">'
        f'<rect width="4" height="2" fill="#000000"/></pattern></defs>'
    )

    parts.append('</svg>')
    return "\n".join(parts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    primes = sieve(20000)
    logo = build_svg(primes, 512, with_text=True)
    icon = build_svg(primes, 256, with_text=False)

    for name, data in (("skynet-logo.svg", logo), ("skynet-icon.svg", icon)):
        path = os.path.join(OUT_DIR, name)
        # newline="\n" so the file is byte-identical on Windows and Linux.
        # Without it Python writes CRLF on Windows, and the CI check that
        # regenerates the logo would see a spurious diff.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
        print(f"wrote {path} ({len(data)} bytes)")

    supers = [p for p in primes[:400] if is_superprime(primes, p)]
    print(f"first super-primes in the emblem: {supers[:12]}")


if __name__ == "__main__":
    main()
