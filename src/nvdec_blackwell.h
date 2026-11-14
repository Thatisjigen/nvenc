// SPDX-License-Identifier: MIT
// Copyright (c) 2026 kevineduardo
// nvdec_blackwell.h: what Blackwell's NVDEC (class 0xcfb0) needs beyond Ada's methods (clc9b0.h).
// No public header lists the methods of 0xcfb0 (open-gpu-kernel-modules' clcfb0.h names the class alone).
#pragma once
// Method 0x4dc: the GPU address (>> 8) of a buffer Blackwell's engine uses for inter pictures. Without it every picture
// that predicts from a reference (VP9, AV1 and HEVC inter frames; H.264 does not need it) ends in a channel error
// (notifier 0xffff) while intra pictures decode bit-exact. Seen in the methods NullMoth's own HEVC decoder sends on an
// RTX 5060 Ti, and in the AROS nvgfx driver, tinygrad (ops_nv.py) and cagedbird043/nvrm; not in clc9b0.h.
#define NVCFB0_SET_INTER_SCRATCH_OFFSET 0x000004dc
// Its size, and the setup word that goes with it: ceil(width / 128) + 1 units of 16 KB (the AROS nvgfx driver's
// reading of NVIDIA's driver on class CFB0: NVDEC_BLK_UNITS / NVDEC_BLK_BUF_SIZE)
#define NVCFB0_BLK_UNITS(w) ((((unsigned)(w) + 127) / 128) + 1)
#define NVCFB0_INTER_SCRATCH_SIZE(w, h) ((size_t)NVCFB0_BLK_UNITS((w) > (h) ? (w) : (h)) * 16384)
// The picture setup's Blackwell fields, as NVIDIA's driver fills them for class CFB0 (traced independently by the AROS
// nvgfx driver and by cagedbird043/nvrm; the values are facts of the hardware interface): the GOB kinds of the decoded
// planes, luma | chroma << 2 (8-bit samples: 1 | 2 << 2 = 9, 16-bit: 2 | 0 << 2 = 2), which select the Blackwell plane
// layouts (blocklinear.h; without them the engine writes the Turing layout and inter prediction reads garbage), and a
// surface format code, 0 / 3 / 6 for 8 / 10 / 12-bit. VP9: reserved5 and reserved1[2], and the BLK units word at +0x108;
// AV1: reserved_stat (kinds | code << 4), reserved12 = 0xffff, and the units word at +0x244.
#define NVCFB0_PLANE_KINDS(bps) ((bps) == 2 ? 2u : 9u)
#define NVCFB0_SURFACE_FORMAT(bit_depth) (3u * (((unsigned)(bit_depth) - 8) / 2))
#define NVDEC_CLASS_BLACKWELL 0xcfb0
