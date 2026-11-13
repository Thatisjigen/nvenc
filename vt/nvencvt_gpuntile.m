// nvencvt_gpuntile.m: the integrated-GPU implementation of gpuntile.h. A Metal compute shader runs the exact
// block-linear untile math (blocklinear.c nvenc_in_offset / nvdec_offset) on the integrated GPU, writing the decoded
// planes straight into the output CVPixelBuffer's IOSurface. Validated bit-exact against the CPU untile for every mode.
//
// The source surfaces and the destination IOSurface are wrapped as Metal buffers over the same system memory (no copy).
// Those wrappers are cached and reused across frames: creating three of them per frame (mapping ~25 MB to the GPU and
// releasing it again, 60x a second) was the bulk of the untile's CPU cost. A wrapper lives only while its memory does —
// rm_mem_free notifies us before it unmaps a source surface, and we retain the destination IOSurface while we hold a
// wrapper over it — so a wrapper is never used over freed pages.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include "gpuntile.h"
#include <os/lock.h>
#include <stdlib.h>
#include <unistd.h>

// One thread per copy run: `run` contiguous destination bytes fed from `run` contiguous source bytes, whose block-linear
// offset is computed from the run's first byte. run = 16 (nvenc_in, and the Blackwell 16-bit GOB) or 8 (Blackwell 8-bit).
// A whole, aligned run is copied with one vector load/store (uint4 = 16 B, uint2 = 8 B); a row tail (partial run) or an
// unaligned surface falls back to the byte copy. Same bytes on either path, so the result is bit-exact.
static NSString *KERNEL_SRC =
@"#include <metal_stdlib>\n"
@"using namespace metal;\n"
@"struct Params { uint width; uint rows; uint pitch; uint dst_stride; uint dst_offset; uint mode; uint bpp; uint run; };\n"
@"static uint off_nvenc_in(uint x, uint y, uint pitch){ uint in=(x%16)+(y%4)*16+((x/16)%2)*64+((y/4)%2)*128+((x/32)%2)*256+((y/8)%2)*512; return (y/16)*pitch*16+(x/64)*1024+in; }\n"
@"static uint off_nvdec_bw(uint x, uint y, uint pitch, uint bpp){ uint gob; if(bpp==1) gob=(x%8)+(y%8)*8+((x/8)%8)*64; else gob=(x%16)+(y%4)*16+((y/4)%2)*64+((x/16)%2)*128+((x/32)%2)*256; return (y/16)*pitch*16+(x/64)*1024+((y/8)%2)*512+gob; }\n"
@"kernel void untile(device uchar *dst [[buffer(0)]], device const uchar *src [[buffer(1)]], constant Params &p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]){\n"
@"  uint x=gid.x*p.run, y=gid.y; if(x>=p.width||y>=p.rows) return;\n"
@"  uint off=(p.mode==0)?off_nvenc_in(x,y,p.pitch):off_nvdec_bw(x,y,p.pitch,p.bpp);\n"
@"  uint n=min(p.run,p.width-x), db=p.dst_offset+y*p.dst_stride+x;\n"
@"  if(n==16 && ((off|db)&15u)==0){ *(device uint4*)(dst+db)=*(device const uint4*)(src+off); return; }\n"
@"  if(n==8 && ((off|db)&7u)==0){ *(device uint2*)(dst+db)=*(device const uint2*)(src+off); return; }\n"
@"  for(uint i=0;i<n;i++) dst[db+i]=src[off+i];\n"
@"}\n";

typedef struct { uint32_t width, rows, pitch, dst_stride, dst_offset, mode, bpp, run; } Params;

static dispatch_once_t gOnce;
static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQ;
static id<MTLComputePipelineState> gPipe;
static int gReady = 0;
static double gLastGpuMs;    // GPU time of the last command buffer in ms (diagnostic; read by the microbench)

// ---- wrapper caches (so we map each surface to the GPU once, not every frame) ----
// Shared across the two decode streams of the Aerial wallpaper, so guarded by a lock. Lookups return an ARC-retained
// buffer that outlives any concurrent eviction for the duration of the call.
static os_unfair_lock gLock = OS_UNFAIR_LOCK_INIT;
static uint64_t gClock;

// source: keyed by the decoder surface's CPU mapping. The decoder allocates its surfaces once per session and reuses
// them across frames; gpu_untile_forget_src drops an entry when the mapping is torn down.
#define SRC_SLOTS 256
typedef struct { const void *cpu; size_t len; CFTypeRef buf; uint64_t used; } SrcSlot;
static SrcSlot gSrc[SRC_SLOTS];

// destination: keyed by IOSurfaceID. We retain the IOSurface while a wrapper over it is cached, so the pixel-buffer
// pool can't free its pages under us; a small LRU bounds how many we pin.
#define DST_SLOTS 16
typedef struct { uint32_t sid; size_t len; IOSurfaceRef ios; CFTypeRef buf; uint64_t used; } DstSlot;
static DstSlot gDst[DST_SLOTS];

static size_t pageup(size_t n) { size_t p = (size_t)getpagesize(); return (n + p - 1) & ~(p - 1); }

// A +1 (ARC-retained) wrapper for the source memory [cpu, cpu+size); creates and caches one on a miss. Call under gLock.
static id<MTLBuffer> src_buffer_locked(const void *cpu, size_t size) {
    if (!cpu) return nil;
    size_t len = pageup(size);
    int victim = 0; uint64_t best = UINT64_MAX;
    for (int i = 0; i < SRC_SLOTS; i++) {
        SrcSlot *s = &gSrc[i];
        if (s->buf && s->cpu == cpu && s->len == len) { s->used = ++gClock; return (__bridge id<MTLBuffer>)s->buf; }
        uint64_t rank = s->buf ? s->used : 0;    // an empty slot (rank 0) is evicted before any live one
        if (rank < best) { best = rank; victim = i; }
    }
    id<MTLBuffer> b = [gDev newBufferWithBytesNoCopy:(void *)cpu length:len options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return nil;
    if (gSrc[victim].buf) CFRelease(gSrc[victim].buf);
    gSrc[victim] = (SrcSlot){ cpu, len, (__bridge_retained CFTypeRef)b, ++gClock };
    return (__bridge id<MTLBuffer>)gSrc[victim].buf;
}

// A +1 (ARC-retained) wrapper for the destination IOSurface; creates one and retains the IOSurface on a miss. Under gLock.
static id<MTLBuffer> dst_buffer_locked(IOSurfaceRef ios) {
    uint32_t sid = IOSurfaceGetID(ios);
    size_t len = pageup(IOSurfaceGetAllocSize(ios));
    int victim = 0; uint64_t best = UINT64_MAX;
    for (int i = 0; i < DST_SLOTS; i++) {
        DstSlot *d = &gDst[i];
        if (d->buf && d->sid == sid && d->len == len) { d->used = ++gClock; return (__bridge id<MTLBuffer>)d->buf; }
        uint64_t rank = d->buf ? d->used : 0;
        if (rank < best) { best = rank; victim = i; }
    }
    void *base = IOSurfaceGetBaseAddress(ios);
    id<MTLBuffer> b = [gDev newBufferWithBytesNoCopy:base length:len options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return nil;
    if (gDst[victim].buf) { CFRelease(gDst[victim].buf); CFRelease(gDst[victim].ios); }
    CFRetain(ios);
    gDst[victim] = (DstSlot){ sid, len, ios, (__bridge_retained CFTypeRef)b, ++gClock };
    return (__bridge id<MTLBuffer>)gDst[victim].buf;
}

static void gpu_setup(void) {
    @autoreleasepool {
        if (getenv("NVENCVT_NOGPU")) return;                      // opt-out: stay on the CPU untile
        id<MTLDevice> intel = nil;
        NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();        // pick the integrated GPU EXPLICITLY (the default is
        for (id<MTLDevice> d in devs)                             // the discrete GPU in com.apple.WebKit.GPU / Safari)
            if (!intel && (d.isLowPower || [d.name containsString:@"Intel"])) intel = d;
        if (!intel) return;                                       // no integrated GPU exposed: stay on the CPU
        gDev = intel;
        NSError *e = nil;
        id<MTLLibrary> lib = [gDev newLibraryWithSource:KERNEL_SRC options:nil error:&e];
        if (!lib) { gDev = nil; return; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"untile"];
        gPipe = [gDev newComputePipelineStateWithFunction:fn error:&e];
        if (!gPipe) { gDev = nil; return; }
        gQ = [gDev newCommandQueue];
        if (!gQ) { gDev = nil; gPipe = nil; return; }
        rm_set_mem_free_hook(gpu_untile_forget_src);              // drop a cached source wrapper before its pages vanish
        gReady = 1;
    }
}

int gpu_untile_ready(void) { dispatch_once(&gOnce, ^{ gpu_setup(); }); return gReady; }

static void encode_plane(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> dst, id<MTLBuffer> src,
                         const GpuPlaneJob *j, uint32_t dst_off, uint32_t dst_stride) {
    Params p = { j->width, j->rows, j->pitch, dst_stride, dst_off, j->mode,
                 j->bpp, j->mode == GPU_UNTILE_NVDEC_BLACKWELL && j->bpp == 1 ? 8u : 16u };
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof p atIndex:2];
    [enc dispatchThreads:MTLSizeMake((j->width + p.run - 1) / p.run, j->rows, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
}

int gpu_untile_nv12(void *iosurface, const GpuPlaneJob *luma, const GpuPlaneJob *chroma) {
    if (!gReady) return -1;
    IOSurfaceRef ios = (IOSurfaceRef)iosurface;
    if (!ios) return -1;
    @autoreleasepool {
        // Look up (or create) the wrappers under the lock and hold strong refs for the whole call, so another stream
        // evicting a slot can't free a buffer we're still submitting.
        os_unfair_lock_lock(&gLock);
        id<MTLBuffer> bL = src_buffer_locked(luma->src->cpu, luma->src->size);
        id<MTLBuffer> bC = src_buffer_locked(chroma->src->cpu, chroma->src->size);
        id<MTLBuffer> bDst = dst_buffer_locked(ios);
        os_unfair_lock_unlock(&gLock);
        if (!bL || !bC || !bDst) return -1;

        void *base = IOSurfaceGetBaseAddress(ios);
        uint32_t off0 = (uint32_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(ios, 0) - (uint8_t *)base);
        uint32_t off1 = (uint32_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(ios, 1) - (uint8_t *)base);
        uint32_t ys = (uint32_t)IOSurfaceGetBytesPerRowOfPlane(ios, 0);
        uint32_t uvs = (uint32_t)IOSurfaceGetBytesPerRowOfPlane(ios, 1);
        id<MTLCommandBuffer> cb = [gQ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPipe];
        encode_plane(enc, bDst, bL, luma, off0, ys);
        encode_plane(enc, bDst, bC, chroma, off1, uvs);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return -1;
        gLastGpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    }
    return 0;
}

void gpu_untile_forget_src(const void *cpu) {
    if (!cpu) return;
    os_unfair_lock_lock(&gLock);
    for (int i = 0; i < SRC_SLOTS; i++)
        if (gSrc[i].buf && gSrc[i].cpu == cpu) { CFRelease(gSrc[i].buf); gSrc[i] = (SrcSlot){0}; }
    os_unfair_lock_unlock(&gLock);
}

void gpu_untile_flush(void) {
    os_unfair_lock_lock(&gLock);
    for (int i = 0; i < SRC_SLOTS; i++) if (gSrc[i].buf) { CFRelease(gSrc[i].buf); gSrc[i] = (SrcSlot){0}; }
    for (int i = 0; i < DST_SLOTS; i++) if (gDst[i].buf) { CFRelease(gDst[i].buf); CFRelease(gDst[i].ios); gDst[i] = (DstSlot){0}; }
    os_unfair_lock_unlock(&gLock);
}

double gpu_untile_last_gpu_ms(void) { return gLastGpuMs; }
