// Everything here is for local use only.
// For exported objects please see rbf.h.

#ifndef __RBF_UTILS_H__
#define __RBF_UTILS_H__

#include <apr_pools.h>

apr_pool_t *memory_pool;

void die_alloc_err(char *func_name, char *vars);

#endif /* __RBF_UTILS_H__ */
