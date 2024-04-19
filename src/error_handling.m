#include "error_handling.h"
#import <Foundation/Foundation.h>

void printNSError(NsError *error) {
    NSError *objcError = (__bridge NSError *)error;
    NSLog(@"Error: %@, Code: %ld", [objcError localizedDescription], (long)[objcError code]);
}
