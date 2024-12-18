/*
 * Copyright (c), Recep Aslantas.
 * MIT License (MIT), http://opensource.org/licenses/MIT
 */

#ifndef cmt_argument_buffer_h
#define cmt_argument_buffer_h
#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"
#include "types.h"
#include "enums.h"
#include "resource.h"

MT_EXPORT
MT_API_AVAILABLE(mt_macos(10.13), mt_ios(11.0))
MtArgumentDescriptor*
mtNewArgumentDescriptor(void);

#ifdef __cplusplus
}
#endif
#endif /* cmt_argument_buffer_h */
