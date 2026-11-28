#ifndef YAWGPU_DAWN_H_
#define YAWGPU_DAWN_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DawnProcTable DawnProcTable;

/* The table is immutable and valid for the lifetime of the process. */
const DawnProcTable* yawgpuDawnGetProcTable(void);

#ifdef __cplusplus
}
#endif

#endif
