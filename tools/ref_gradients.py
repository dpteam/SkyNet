"""ref_gradients.py - the reference (PyTorch autograd) gradient for our model.

refs/Tiny-2.1-main/training/train2.py is the trainer that produced TINY.BIN,
so its autograd is the ground truth for the hand-written backward pass in
src/sk_train.c. This script loads the same weights, runs the same token
sequence, and prints dL/dp for every parameter kind the fixed-point trainer
tracks. tools/compare_gradients.py diffs the two.

Why this matters: the numerical checker in tests/gradcheck.c can only see
gradients large enough to move a Q16.16 loss. This one sees everything, so it
confirms the FORMULAS - medians of ours/theirs should sit at 1.0 - while the
checker confirms the fixed-point arithmetic.

  python tools/ref_gradients.py tiny2.pt build/dataset.txt 20 > ref.txt

Needs torch, which the core does not: this is a development tool, not part of
the program.
"""
import sys, os, io

REFDIR = os.path.join('refs', 'Tiny-2.1-main', 'training')
sys.path.insert(0, REFDIR)

import torch
from train2 import load_model, VOCAB

KINDS = {
    'GF': 'nf.g',
    'G2': 'blocks.{last}.n2.g',
    'F1': 'blocks.{last}.f1.weight',
    'F2': 'blocks.{last}.f2.weight',
    'OUT': 'emb.weight',
}


def toks_of(line):
    """Byte-for-byte the same tokenisation as tokenize() in gradcheck.c."""
    q, a = line.split('|', 1)
    a = a.rstrip('\n\r')
    t = []
    for c in q:
        v = ord(c) - 30
        t.append(2 if v < 2 else (VOCAB - 1 if v >= VOCAB else v))
    t.append(1)
    for c in a:
        v = ord(c) - 30
        t.append(2 if v < 2 else (VOCAB - 1 if v >= VOCAB else v))
    t.append(1)
    return t, len(q) + 1


def rowscale_grad(w, g):
    """dL/ds_i where the stored row is s_i * code_i.

    export() writes s_i = max|W_i| / 127, so code = round(W_i / s_i) and the
    derivative with respect to the scale is what the fixed-point trainer
    stores in its g_* accumulators. Without this the numbers are off by the
    scale itself, which is not a constant across rows and reads as a
    structural error.
    """
    out = []
    with torch.no_grad():
        for i in range(w.shape[0]):
            s = float(w[i].abs().max().item()) / 127.0
            if s <= 0.0:
                out.append(0.0)
                continue
            code = torch.round(w[i] / s).clamp(-127, 127)
            out.append(float((g[i] * code).sum().item()))
    return out


def main():
    mpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REFDIR, 'tiny2.pt')
    dpath = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        'build', 'dataset.txt')
    nlines = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    torch.set_grad_enabled(True)
    m = load_model(mpath)

    samples = []
    with io.open(dpath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(samples) >= nlines:
                break
            t, af = toks_of(line)
            # Same filter as tests/gradcheck.c: MAXTOK is 64 there, so longer
            # lines are skipped there and must be skipped here, or the two
            # sides average over different examples.
            if af < len(t) - 1 and len(t) <= 64:
                samples.append((t, af))

    total, npos = 0.0, 0
    for toks, af in samples:
        logits = m(torch.tensor([toks]))[0]
        for t in range(af, len(toks) - 1):
            total = total + torch.logsumexp(logits[t], 0) - logits[t][toks[t + 1]]
            npos += 1
    if npos == 0:
        print('no usable line', file=sys.stderr)
        return 2

    loss = total / npos
    loss.backward()
    print('# loss %.9f positions %d lines %d' % (loss.item(), npos,
                                                 len(samples)))

    last = m.blocks[-1]
    named = {
        'GF': (m.nf.g, m.nf.g.grad),
        'G2': (last.n2.g, last.n2.g.grad),
        'F1': (last.f1.weight.detach(), last.f1.weight.grad),
        'F2': (last.f2.weight.detach(), last.f2.weight.grad),
        'OUT': (m.emb.weight.detach(), m.emb.weight.grad),
    }
    for key in ('GF', 'G2', 'F1', 'F2', 'OUT'):
        w, g = named[key]
        if g is None:
            print('%s 0' % key)
            continue
        vals = (g.tolist() if w.dim() == 1
                else rowscale_grad(w, g))
        print('%s %d' % (key, len(vals)))
        print(' '.join('%.12g' % v for v in vals))
    return 0


if __name__ == '__main__':
    sys.exit(main())
