// Minimal Vec3 / Mat4 helpers for the tiled_deferred example.
//
// No third-party math library — the demo only needs look-at, perspective,
// matrix multiplication, and matrix inverse. Conventions mirror the wgpu
// reference's `glam` usage so the WGSL uniforms can be uploaded byte-for-byte
// with what `mat4_look_at_rh` / `mat4_perspective_rh` produce here:
//   * Column-major storage: `m[col * 4 + row]`. Matches glam's
//     `to_cols_array_2d()` layout that the WGSL `mat4x4<f32>` reads.
//   * Right-handed view space (`mat4_look_at_rh`): forward = normalize(center - eye),
//     side = forward × up, up = side × forward, eye translation negated.
//   * Right-handed perspective with WebGPU NDC depth [0, 1]
//     (`mat4_perspective_rh`): identical to glam's `Mat4::perspective_rh`.
//   * `mat4_inverse` is the cofactor-expansion adjugate / determinant —
//     general enough for the deferred lighting subpass's
//     world-position-from-depth reconstruction.
//
// All helpers are `static inline` since this header is included from a single
// translation unit (`main.c`).

#ifndef YAWGPU_TILED_DEFERRED_MATH_H
#define YAWGPU_TILED_DEFERRED_MATH_H

#include <math.h>
#include <stdbool.h>

typedef struct Vec3 {
    float x;
    float y;
    float z;
} Vec3;

typedef struct Mat4 {
    float m[16];
} Mat4;

static inline Vec3 vec3_make(float x, float y, float z) {
    return (Vec3){ .x = x, .y = y, .z = z };
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return vec3_make(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return vec3_make(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static inline float vec3_length(Vec3 v) {
    return sqrtf(vec3_dot(v, v));
}

static inline Vec3 vec3_normalize(Vec3 v) {
    float length = vec3_length(v);
    if (length <= 0.0f) {
        return vec3_make(0.0f, 0.0f, 0.0f);
    }
    float inv = 1.0f / length;
    return vec3_make(v.x * inv, v.y * inv, v.z * inv);
}

static inline float mat4_get(Mat4 m, int row, int col) {
    return m.m[col * 4 + row];
}

static inline void mat4_set(Mat4 *m, int row, int col, float value) {
    m->m[col * 4 + row] = value;
}

static inline Mat4 mat4_identity(void) {
    return (Mat4){
        .m = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        },
    };
}

static inline Mat4 mat4_mul(Mat4 a, Mat4 b) {
    Mat4 out = {0};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) {
                value += mat4_get(a, row, k) * mat4_get(b, k, col);
            }
            mat4_set(&out, row, col, value);
        }
    }
    return out;
}

static inline Mat4 mat4_look_at_rh(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = vec3_normalize(vec3_sub(center, eye));
    Vec3 s = vec3_normalize(vec3_cross(f, up));
    Vec3 u = vec3_cross(s, f);

    // Column-major storage (m[col*4 + row]), matching glam's `from_cols`.
    // col 0 = (s.x, u.x, -f.x, 0)
    // col 1 = (s.y, u.y, -f.y, 0)
    // col 2 = (s.z, u.z, -f.z, 0)
    // col 3 = (-eye·s, -eye·u, eye·f, 1)
    Mat4 out = mat4_identity();
    out.m[0] = s.x;
    out.m[1] = u.x;
    out.m[2] = -f.x;
    out.m[4] = s.y;
    out.m[5] = u.y;
    out.m[6] = -f.y;
    out.m[8] = s.z;
    out.m[9] = u.z;
    out.m[10] = -f.z;
    out.m[12] = -vec3_dot(s, eye);
    out.m[13] = -vec3_dot(u, eye);
    out.m[14] = vec3_dot(f, eye);
    return out;
}

static inline Mat4 mat4_perspective_rh(float fovy, float aspect, float z_near, float z_far) {
    float f = 1.0f / tanf(fovy * 0.5f);
    Mat4 out = {0};
    out.m[0] = f / aspect;
    out.m[5] = f;
    out.m[10] = z_far / (z_near - z_far);
    out.m[11] = -1.0f;
    out.m[14] = (z_near * z_far) / (z_near - z_far);
    return out;
}

static inline bool mat4_inverse(Mat4 matrix, Mat4 *out) {
    float inv[16];
    const float *m = matrix.m;

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] +
             m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] -
             m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] +
              m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] +
             m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] +
             m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] -
              m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] +
             m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] +
              m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (fabsf(det) < 0.000001f) {
        return false;
    }
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) {
        out->m[i] = inv[i] * det;
    }
    return true;
}

#endif
