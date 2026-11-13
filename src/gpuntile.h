// gpuntile.h: untile the decoder's block-linear output planes into an NV12 IOSurface on the integrated GPU, so the
// per-frame de-tiling runs on the (otherwise idle) iGPU instead of the CPU. The CPU path (blocklinear.c) stays as the
// fallback. The offset math is identical to blocklinear.c's nvenc_in_offset / nvdec_offset (bit-exact by construction).
#pragma once
#include "rm.h"
#include <stdint.h>
#include <stddef.h>

enum { GPU_UNTILE_NVENC_IN = 0,        // the nvenc_in layout (HEVC, VP9, AV1, VP8, VC1, MPEG-2/4, and H.264 pre-Blackwell)
       GPU_UNTILE_NVDEC_BLACKWELL = 1 };   // the Blackwell GOBs (H.264 on NVDEC 0xcfb0); bpp selects the 8/16-bit GOB

typedef struct {
    const RmMem *src;     // the decoder's block-linear surface (system memory, GPU-mapped)
    uint32_t width;       // plane width in bytes
    uint32_t rows;        // plane height in rows
    uint32_t pitch;       // the surface pitch in bytes
    uint32_t mode;        // GPU_UNTILE_*
    uint32_t bpp;         // 1 or 2 (Blackwell GOB order + copy run); ignored for the nvenc_in layout
} GpuPlaneJob;

#ifdef __cplusplus
extern "C" {
#endif

// 1 if the integrated-GPU untile is initialized and usable (lazy init on first call; NVENCVT_NOGPU=1 disables it),
// 0 to fall back to the CPU untile. Safe to call from any thread.
int gpu_untile_ready(void);

// Untile luma (job[0]) and chroma (job[1]) into the NV12 IOSurface (passed as a void* IOSurfaceRef) in one command
// buffer on the integrated GPU. Returns 0 on success, non-zero to fall back to the CPU untile.
int gpu_untile_nv12(void *iosurface, const GpuPlaneJob *luma, const GpuPlaneJob *chroma);

// Drop any cached wrapper over the source mapping at `cpu`. rm_mem_free calls this (as the registered hook) before it
// unmaps a decoder surface, so a wrapper is never kept over freed pages. No-op when nothing is cached for `cpu`.
void gpu_untile_forget_src(const void *cpu);

// Release every cached wrapper (and the IOSurfaces retained for the destination wrappers). For shutdown and tests.
void gpu_untile_flush(void);

// GPU execution time of the last gpu_untile_nv12 command buffer, in milliseconds (diagnostic for the microbench).
double gpu_untile_last_gpu_ms(void);

#ifdef __cplusplus
}
#endif
