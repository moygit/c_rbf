/*
 * EVERYTHING HERE IS FOR LOCAL USE ONLY.
 * FOR EXPORTED OBJECTS PLEASE SEE rbf.h.
 */

#ifndef __RBF_UTILS_H__
#define __RBF_UTILS_H__

#include <apr_pools.h>

apr_pool_t *memory_pool;

void die_alloc_err(char *func_name, char *vars);

#endif /* __RBF_UTILS_H__ */
