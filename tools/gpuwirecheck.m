// gpuwirecheck.m: prove the out_copy GPU-untile wiring bit-exact, per codec, with no NVDEC and no install. For each
// codec's plane profile it drives the REAL gpu_untile_nv12 (vt/nvencvt_gpuntile.m) into a real NV12 / P010 / NV24
// IOSurface, and compares against that codec's REAL read_nv12 (src/*dec.c) run into a linear buffer with the IOSurface's
// own strides. The reference comes from the decoders' own read_nv12, so a mistake in the plane parameters the wiring
// passes (width in bytes, rows, pitch, mode, bpp) shows up as a mismatch. Random tiled sources: same bytes, same offset
// math on both sides, so equality is bit-exactness of the whole GPU path (kernel + IOSurface geometry + two planes).
//   ./gpuwirecheck [W H]            default 250x130 (not a multiple of the 64-byte tile: stresses the row/column tails)
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#include "dec.h"
#include "vp9dec.h"
#include "hevcdec.h"
#include "mpeg2dec.h"
#include "gpuntile.h"
#include "blocklinear.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern int nvdec_gob_blackwell;

enum { C_H264, C_HEVC, C_VP9, C_MPEG2 };

static uint8_t *valloc_rand(size_t n, unsigned seed) {
    size_t pg = (size_t)getpagesize(); uint8_t *p = valloc((n + pg - 1) & ~(pg - 1));
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)((i * 131u + seed * 7u + (i >> 9) * 13u) & 0xff);
    return p;
}

// compare `rows` rows of `width` bytes between two planes that share stride `stride`
static size_t cmp_plane(const uint8_t *ref, const uint8_t *got, size_t stride, size_t width, size_t rows,
                        size_t *fr, size_t *fx) {
    size_t d = 0; *fr = *fx = (size_t)-1;
    for (size_t r = 0; r < rows; r++)
        for (size_t x = 0; x < width; x++)
            if (ref[r * stride + x] != got[r * stride + x]) { if (!d) { *fr = r; *fx = x; } d++; }
    return d;
}

static int run(const char *name, int codec, OSType pf, unsigned w, unsigned h, int bps, int chroma_fmt, int blackwell)
{
    nvdec_gob_blackwell = blackwell;
    const uint32_t b = (uint32_t)bps;
    const uint32_t luma_wb = w * b;
    const uint32_t pitch = (luma_wb + 63u) & ~63u;
    // chroma geometry, mirroring each read_nv12
    uint32_t chroma_wb, crows, cp = pitch, lumaMode, chromaMode, lumaBpp, chromaBpp;
    if (codec == C_H264) {
        chroma_wb = w; crows = h / 2; lumaMode = chromaMode = blackwell ? GPU_UNTILE_NVDEC_BLACKWELL : GPU_UNTILE_NVENC_IN;
        lumaBpp = 1; chromaBpp = 2;
    } else if (codec == C_HEVC) {
        const uint32_t cw = chroma_fmt == 3 ? w : (w + 1) / 2;
        crows = chroma_fmt == 3 ? h : (h + 1) / 2; cp = chroma_fmt == 3 ? 2 * pitch : pitch;
        chroma_wb = 2 * cw * b; lumaMode = chromaMode = GPU_UNTILE_NVENC_IN; lumaBpp = chromaBpp = b;
    } else if (codec == C_VP9) {
        chroma_wb = ((w + 1) & ~1u) * b; crows = (h + 1) / 2; lumaMode = chromaMode = GPU_UNTILE_NVENC_IN; lumaBpp = chromaBpp = b;
    } else { // MPEG2
        chroma_wb = w; crows = (h + 1) / 2; lumaMode = chromaMode = GPU_UNTILE_NVENC_IN; lumaBpp = chromaBpp = 1;
    }

    // random tiled sources (shared by the CPU reference and the GPU test)
    const size_t sL = nvenc_in_plane_size(pitch, h), sC = nvenc_in_plane_size(cp, crows);
    uint8_t *tL = valloc_rand(sL, 1), *tC = valloc_rand(sC, 2);
    RmMem mL = {0}, mC = {0}; mL.cpu = tL; mL.size = sL; mC.cpu = tC; mC.size = sC;

    // the output IOSurface (pixel buffer), and the reference planes at its strides
    NSDictionary *at = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
    CVPixelBufferRef pb = NULL;
    OSStatus st = CVPixelBufferCreate(NULL, w, h, pf, (__bridge CFDictionaryRef)at, &pb);
    if (st || !pb || !CVPixelBufferGetIOSurface(pb)) { printf("  %-22s SKIP (no IOSurface for %.4s: %d)\n", name, (char *)&pf, (int)st); if (pb) CVPixelBufferRelease(pb); free(tL); free(tC); return 0; }
    CVPixelBufferLockBaseAddress(pb, 0);
    const size_t ys = CVPixelBufferGetBytesPerRowOfPlane(pb, 0), uvs = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
    uint8_t *refY = calloc(ys, h + 1), *refUV = calloc(uvs, crows + 1);

    // reference: the decoder's own read_nv12, into refY/refUV with the IOSurface strides
    if (codec == C_H264) { H264Dec *d = calloc(1, sizeof *d); d->pitch = pitch; d->luma[0] = mL; d->chroma[0] = mC; d->x0[0] = d->y0[0] = 0;
        h264dec_read_nv12(d, 0, refY, ys, refUV, uvs, w, h); free(d); }
    else if (codec == C_HEVC) { HevcDec *d = calloc(1, sizeof *d); d->bps = bps; d->chroma_format = chroma_fmt; d->pitch = pitch; d->luma[0] = mL; d->chroma[0] = mC;
        hevcdec_read_nv12(d, 0, refY, ys, refUV, uvs, w, h, 0, 0); free(d); }
    else if (codec == C_VP9) { Vp9Dec *d = calloc(1, sizeof *d); d->bps = bps; d->pitch = pitch; d->luma[0] = mL; d->chroma[0] = mC;
        vp9dec_read_nv12(d, 0, refY, ys, refUV, uvs, w, h); free(d); }
    else { Mpeg2Dec *d = calloc(1, sizeof *d); d->pitch = pitch; d->luma[0] = mL; d->chroma[0] = mC;
        mpeg2dec_read_nv12(d, 0, refY, ys, refUV, uvs, w, h); free(d); }

    // test: the real gpu_untile_nv12 into the IOSurface, same plane parameters the wiring (out_plane_jobs) passes
    GpuPlaneJob g[2];
    g[0] = (GpuPlaneJob){ .src = &mL, .width = luma_wb,  .rows = h,     .pitch = pitch, .mode = lumaMode,   .bpp = lumaBpp };
    g[1] = (GpuPlaneJob){ .src = &mC, .width = chroma_wb, .rows = crows, .pitch = cp,    .mode = chromaMode, .bpp = chromaBpp };
    if (!gpu_untile_ready()) { printf("  %-22s SKIP (no integrated GPU)\n", name); CVPixelBufferUnlockBaseAddress(pb, 0); CVPixelBufferRelease(pb); free(refY); free(refUV); free(tL); free(tC); return 0; }
    IOSurfaceRef ios = CVPixelBufferGetIOSurface(pb);
    int rc = gpu_untile_nv12((void *)ios, &g[0], &g[1]);

    const uint8_t *gotY = CVPixelBufferGetBaseAddressOfPlane(pb, 0), *gotUV = CVPixelBufferGetBaseAddressOfPlane(pb, 1);
    size_t fr, fx, dY = cmp_plane(refY, gotY, ys, luma_wb, h, &fr, &fx);
    size_t gr, gx, dC = cmp_plane(refUV, gotUV, uvs, chroma_wb, crows, &gr, &gx);

    // second pass into the same, cleared surface: exercises the cached wrappers (hit path); must be bit-exact too
    memset((void *)gotY, 0, ys * (size_t)h); memset((void *)gotUV, 0, uvs * (size_t)crows);
    int rc2 = gpu_untile_nv12((void *)ios, &g[0], &g[1]);
    size_t fr2, fx2, dY2 = cmp_plane(refY, gotY, ys, luma_wb, h, &fr2, &fx2);
    size_t gr2, gx2, dC2 = cmp_plane(refUV, gotUV, uvs, chroma_wb, crows, &gr2, &gx2);

    int fail = rc != 0 || rc2 != 0 || dY || dC || dY2 || dC2;
    printf("  %-22s %s  luma %ux%uB %s  chroma %ux%uB %s  cache %s%s\n", name, fail ? "FAIL" : "PASS",
           luma_wb, h, dY ? "MISMATCH" : "ok", chroma_wb, crows, dC ? "MISMATCH" : "ok",
           (dY2 || dC2) ? "MISMATCH" : "ok", (rc || rc2) ? "  (gpu_untile_nv12 returned nonzero)" : "");
    if (dY)  printf("      luma   first diff row %zu x %zu: ref=%02x gpu=%02x (%zu bytes)\n", fr, fx, refY[fr*ys+fx], gotY[fr*ys+fx], dY);
    if (dC)  printf("      chroma first diff row %zu x %zu: ref=%02x gpu=%02x (%zu bytes)\n", gr, gx, refUV[gr*uvs+gx], gotUV[gr*uvs+gx], dC);
    if (dY2 || dC2) printf("      cache  pass differs from the first pass (%zu luma, %zu chroma bytes)\n", dY2, dC2);

    // the decoder drops a surface's wrapper (via rm_mem_free -> gpu_untile_forget_src) before freeing it; do the same,
    // so the next profile reusing this address can't hit a stale wrapper and the invalidation path gets exercised.
    gpu_untile_forget_src(tL); gpu_untile_forget_src(tC);
    CVPixelBufferUnlockBaseAddress(pb, 0); CVPixelBufferRelease(pb); free(refY); free(refUV); free(tL); free(tC);
    return fail;
}

int main(int argc, char **argv)
{
    unsigned W = argc > 2 ? (unsigned)atoi(argv[1]) : 250, H = argc > 2 ? (unsigned)atoi(argv[2]) : 130;
    const OSType NV12 = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    const OSType P010 = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange;
    const OSType NV24 = kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange;
    printf("gpuwirecheck %ux%u: real gpu_untile_nv12 vs each decoder's read_nv12 (bit-exact, no NVDEC)\n", W, H);
    int f = 0;
    f += run("h264 nv12 blackwell", C_H264, NV12, W, H, 1, 1, 1);
    f += run("h264 nv12 turing",    C_H264, NV12, W, H, 1, 1, 0);
    f += run("hevc nv12 8-bit",     C_HEVC, NV12, W, H, 1, 1, 0);
    f += run("hevc p010 10-bit",    C_HEVC, P010, W, H, 2, 1, 0);
    f += run("hevc nv24 4:4:4",     C_HEVC, NV24, W, H, 1, 3, 0);
    f += run("vp9 p010 10-bit",     C_VP9,  P010, W, H, 2, 1, 0);
    f += run("mpeg2 nv12",          C_MPEG2,NV12, W, H, 1, 1, 0);
    gpu_untile_flush();
    printf("\nRESULT: %s (%d profile%s failed)\n", f ? "WIRING BIT-EXACT FAILURES" : "all profiles bit-exact", f, f == 1 ? "" : "s");
    return f ? 1 : 0;
}
