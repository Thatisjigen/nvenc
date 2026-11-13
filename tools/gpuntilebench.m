// gpuntilebench.m: measure the per-frame cost of the integrated-GPU untile (vt/nvencvt_gpuntile.m) offline, with no
// NVDEC and no install. Builds a 4K output IOSurface and random block-linear sources, then times gpu_untile_nv12 over
// many iterations into the same surfaces (the decoder reuses its surfaces across frames, so this is the steady state).
// Reports wall time per frame (CPU submit + GPU wait, as out_copy pays it) and the GPU execution time per frame.
//   build it like gpuwirecheck: clang -O2 -fobjc-arc tools/gpuntilebench.m vt/nvencvt_gpuntile.m src/*.c -I src <vendor -I...>
//   ./gpuntilebench [iters]          default 600
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#include "dec.h"
#include "gpuntile.h"
#include "blocklinear.h"
#include <mach/mach_time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern int nvdec_gob_blackwell;

enum { C_H264, C_HEVC };

static uint8_t *valloc_rand(size_t n, unsigned seed) {
    size_t pg = (size_t)getpagesize(); uint8_t *p = valloc((n + pg - 1) & ~(pg - 1));
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)((i * 131u + seed * 7u + (i >> 9) * 13u) & 0xff);
    return p;
}

static double ms_now(void) {
    static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static void bench(const char *name, int codec, OSType pf, unsigned w, unsigned h, int bps, int chroma_fmt, int blackwell, int iters) {
    nvdec_gob_blackwell = blackwell;
    const uint32_t b = (uint32_t)bps;
    const uint32_t luma_wb = w * b;
    const uint32_t pitch = (luma_wb + 63u) & ~63u;
    uint32_t chroma_wb, crows, cp = pitch, lumaMode, chromaMode, lumaBpp, chromaBpp;
    if (codec == C_H264) {
        chroma_wb = w; crows = h / 2; lumaMode = chromaMode = blackwell ? GPU_UNTILE_NVDEC_BLACKWELL : GPU_UNTILE_NVENC_IN;
        lumaBpp = 1; chromaBpp = 2;
    } else { // HEVC 4:2:0
        const uint32_t cw = chroma_fmt == 3 ? w : (w + 1) / 2;
        crows = chroma_fmt == 3 ? h : (h + 1) / 2; cp = chroma_fmt == 3 ? 2 * pitch : pitch;
        chroma_wb = 2 * cw * b; lumaMode = chromaMode = GPU_UNTILE_NVENC_IN; lumaBpp = chromaBpp = b;
    }
    const size_t sL = nvenc_in_plane_size(pitch, h), sC = nvenc_in_plane_size(cp, crows);
    uint8_t *tL = valloc_rand(sL, 1), *tC = valloc_rand(sC, 2);
    RmMem mL = {0}, mC = {0}; mL.cpu = tL; mL.size = sL; mC.cpu = tC; mC.size = sC;

    NSDictionary *at = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
    CVPixelBufferRef pb = NULL;
    if (CVPixelBufferCreate(NULL, w, h, pf, (__bridge CFDictionaryRef)at, &pb) || !pb || !CVPixelBufferGetIOSurface(pb)) {
        printf("  %-26s SKIP (no IOSurface for %.4s)\n", name, (char *)&pf); if (pb) CVPixelBufferRelease(pb); free(tL); free(tC); return;
    }
    IOSurfaceRef ios = CVPixelBufferGetIOSurface(pb);
    GpuPlaneJob g[2];
    g[0] = (GpuPlaneJob){ .src = &mL, .width = luma_wb,  .rows = h,     .pitch = pitch, .mode = lumaMode,   .bpp = lumaBpp };
    g[1] = (GpuPlaneJob){ .src = &mC, .width = chroma_wb, .rows = crows, .pitch = cp,    .mode = chromaMode, .bpp = chromaBpp };

    if (!gpu_untile_ready()) { printf("  %-26s SKIP (no integrated GPU)\n", name); CVPixelBufferRelease(pb); free(tL); free(tC); return; }

    for (int i = 0; i < 16; i++) {   // warm up: first calls create the wrappers (and prime the caches)
        CVPixelBufferLockBaseAddress(pb, 0); gpu_untile_nv12((void *)ios, &g[0], &g[1]); CVPixelBufferUnlockBaseAddress(pb, 0);
    }
    double wall = 0, gpu = 0; int rc = 0;
    for (int i = 0; i < iters; i++) {
        double t0 = ms_now();
        CVPixelBufferLockBaseAddress(pb, 0);
        rc = gpu_untile_nv12((void *)ios, &g[0], &g[1]);
        CVPixelBufferUnlockBaseAddress(pb, 0);
        wall += ms_now() - t0;
        gpu += gpu_untile_last_gpu_ms();
        if (rc) { printf("  %-26s FAIL (rc=%d)\n", name, rc); break; }
    }
    if (!rc) printf("  %-26s wall %.4f ms/frame  gpu %.4f ms/frame  (%d iters, %ux%u)\n", name, wall / iters, gpu / iters, iters, w, h);
    CVPixelBufferRelease(pb); free(tL); free(tC);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 600;
    const OSType NV12 = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    const OSType P010 = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange;
    printf("gpuntilebench: steady-state cost of gpu_untile_nv12 (no NVDEC), %d iters\n", iters);
    bench("hevc nv12 8-bit 4K",     C_HEVC, NV12, 3840, 2160, 1, 1, 0, iters);
    bench("hevc p010 10-bit 4K",    C_HEVC, P010, 3840, 2160, 2, 1, 0, iters);
    bench("h264 nv12 blackwell 4K", C_H264, NV12, 3840, 2160, 1, 1, 1, iters);
    return 0;
}
