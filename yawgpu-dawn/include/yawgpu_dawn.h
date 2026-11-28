#ifndef YAWGPU_DAWN_H_
#define YAWGPU_DAWN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DawnProcTable DawnProcTable;

/* The table is immutable and valid for the lifetime of the process. */
const DawnProcTable* yawgpuDawnGetProcTable(void);

/* Atomically set the process-wide default for subsequent proc-table createInstance
 * calls without an explicit backend selector. Use a YAWGPU_INSTANCE_BACKEND_* value
 * from yawgpu.h. */
void yawgpuDawnSetDefaultBackend(uint32_t backend);

#ifdef __cplusplus
}
#endif

#endif
