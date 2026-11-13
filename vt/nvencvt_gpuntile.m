// nvencvt_gpuntile.m: the integrated-GPU implementation of gpuntile.h. A Metal compute shader runs the exact
// block-linear untile math (blocklinear.c nvenc_in_offset / nvdec_offset) on the integrated GPU, writing the decoded
// planes straight into the output CVPixelBuffer's IOSurface. Validated bit-exact against the CPU untile for every mode.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include "gpuntile.h"
#include <stdlib.h>
#include <unistd.h>

// byte-wise, one copy run per thread; run = 16 (nvenc_in / Blackwell 16-bit) or 8 (Blackwell 8-bit).
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
@"  for(uint i=0;i<n;i++) dst[db+i]=src[off+i];\n"
@"}\n";

typedef struct { uint32_t width, rows, pitch, dst_stride, dst_offset, mode, bpp, run; } Params;

static dispatch_once_t gOnce;
static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQ;
static id<MTLComputePipelineState> gPipe;
static int gReady = 0;

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
        gReady = 1;
    }
}

int gpu_untile_ready(void) { dispatch_once(&gOnce, ^{ gpu_setup(); }); return gReady; }

static size_t pageup(size_t n) { size_t p = (size_t)getpagesize(); return (n + p - 1) & ~(p - 1); }

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
        // wrap the two source surfaces and the whole destination IOSurface as shared buffers (no copy; the iGPU reads
        // and writes system memory directly). Per-call wrapping keeps it correct across surface and pool recycling.
        id<MTLBuffer> bL = [gDev newBufferWithBytesNoCopy:luma->src->cpu   length:pageup(luma->src->size)
                                                 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bC = [gDev newBufferWithBytesNoCopy:chroma->src->cpu length:pageup(chroma->src->size)
                                                 options:MTLResourceStorageModeShared deallocator:nil];
        void *base = IOSurfaceGetBaseAddress(ios);
        id<MTLBuffer> bDst = [gDev newBufferWithBytesNoCopy:base length:pageup(IOSurfaceGetAllocSize(ios))
                                                   options:MTLResourceStorageModeShared deallocator:nil];
        if (!bL || !bC || !bDst) return -1;
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
    }
    return 0;
}
