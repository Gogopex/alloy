#ifndef cmt_error_handling_h
#define cmt_error_handling_h

#ifdef __cplusplus
extern "C" {
#endif

#include "error.h"

void printNSError(NsError *error);

#ifdef __cplusplus
}
#endif

#endif /* cmt_error_handling_h */
