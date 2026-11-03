// untilecheck.c: nvenc_in_untile_plane against the per-run reference (nvenc_in_offset), on one and on four threads, into
// destinations on cache lines (non-temporal stores) and off them, with sizes that leave partial tiles and a short last band.
//   ./untilecheck        prints what failed and how long a 4K 10-bit plane takes
#include "blocklinear.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }

static int check(size_t width, size_t rows, size_t pitch, size_t stride, size_t skew, unsigned threads)
{
    const size_t size = nvenc_in_plane_size(pitch, rows);
    uint8_t *src = aligned_alloc(4096, (size + 4095) & ~(size_t)4095), *dst0 = aligned_alloc(4096, (stride * rows + skew + 4095) & ~(size_t)4095), *dst = dst0 + skew;
    for (size_t i = 0; i < size; i++) src[i] = (uint8_t)((i * 2654435761u) >> 13);
    memset(dst0, 0xa5, stride * rows + skew);
    nvenc_in_untile_plane_n(dst, stride, src, width, rows, pitch, threads);
    int bad = 0;
    for (size_t y = 0; y < rows && !bad; y++) {
        for (size_t x = 0; x < width; x++) if (dst[y * stride + x] != src[nvenc_in_offset(x, y, pitch)]) { printf("  %zux%zu pitch %zu stride %zu skew %zu threads %u: byte (%zu, %zu) wrong\n", width, rows, pitch, stride, skew, threads, x, y); bad = 1; break; }
        for (size_t x = width; x < stride && !bad; x++) if (dst[y * stride + x] != 0xa5) { printf("  %zux%zu stride %zu skew %zu threads %u: written past the row at (%zu, %zu)\n", width, rows, stride, skew, threads, x, y); bad = 1; }
    }
    free(src); free(dst0);
    return bad;
}

int main(void)
{
    static const size_t sizes[][3] = { { 7680, 2160, 7680 }, { 7680, 1080, 7680 }, { 3840, 2160, 3840 }, { 3840, 1088, 3840 }, { 7680, 1088, 8192 }, { 1920, 1080, 1920 }, { 640, 480, 640 }, { 3838, 1081, 3840 }, { 4000, 520, 4032 } };
    int bad = 0, runs = 0;
    for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++)
        for (unsigned t = 0; t <= 1; t++)
            for (size_t skew = 0; skew <= 16; skew += 16)
                for (size_t st = 0; st <= 1; st++) {
                    const size_t stride = st ? ((sizes[i][0] + 127) & ~(size_t)63) + 16 : (sizes[i][0] + 63) & ~(size_t)63;
                    bad += check(sizes[i][0], sizes[i][1], sizes[i][2], stride, skew, t); runs++;
                }
    printf("%d runs, %d failed\n", runs, bad);
    const size_t w = 7680, h = 2160, size = nvenc_in_plane_size(w, h);
    uint8_t *src = aligned_alloc(4096, (size + 4095) & ~(size_t)4095), *dst = aligned_alloc(4096, w * h);
    memset(src, 1, size);
    for (unsigned t = 0; t <= 1; t++) { double t0 = now_ms(); for (int k = 0; k < 20; k++) nvenc_in_untile_plane_n(dst, w, src, w, h, w, t); printf("4K 10-bit luma, %s: %.2f ms\n", t ? "one thread" : "four threads", (now_ms() - t0) / 20); }
    return bad != 0;
}
