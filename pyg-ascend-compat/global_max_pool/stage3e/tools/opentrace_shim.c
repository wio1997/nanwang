/* Stage 3E / Task D: LD_PRELOAD shim that records every file the process opens whose path
 * mentions the ScatterMaxV1 mission (scatter_max / ScatterMax / customize).
 *
 * Purpose: prove *which* custom OPP package supplied the kernel binary/metadata that the ACL
 * runtime loaded, by logging the literal path it opened.
 *
 * The shim never calls into libc's open; it issues the syscall directly, so there is no recursion.
 * Build:  gcc -shared -fPIC -O2 -o opentrace_shim.so opentrace_shim.c
 * Use  :  STAGE3E_OPEN_TRACE=<logfile> LD_PRELOAD=./opentrace_shim.so <command>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static int g_fd = 0; /* 0 = not opened yet, -1 = disabled */

static void note(const char *path)
{
    if (path == NULL) {
        return;
    }
    if (strstr(path, "scatter_max") == NULL && strstr(path, "ScatterMax") == NULL &&
        strstr(path, "customize") == NULL) {
        return;
    }
    if (g_fd == 0) {
        const char *out = getenv("STAGE3E_OPEN_TRACE");
        if (out == NULL) {
            g_fd = -1;
            return;
        }
        g_fd = (int)syscall(SYS_openat, AT_FDCWD, out, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (g_fd < 0) {
            g_fd = -1;
            return;
        }
    }
    if (g_fd < 0) {
        return;
    }
    char buf[4200];
    int n = snprintf(buf, sizeof(buf), "%d %s\n", (int)syscall(SYS_gettid), path);
    if (n > 0) {
        (void)syscall(SYS_write, g_fd, buf, (size_t)n);
    }
}

int open(const char *path, int flags, ...)
{
    va_list ap;
    va_start(ap, flags);
    mode_t mode = (mode_t)va_arg(ap, int);
    va_end(ap);
    note(path);
    return (int)syscall(SYS_openat, AT_FDCWD, path, flags, mode);
}

int open64(const char *path, int flags, ...)
{
    va_list ap;
    va_start(ap, flags);
    mode_t mode = (mode_t)va_arg(ap, int);
    va_end(ap);
    note(path);
    return (int)syscall(SYS_openat, AT_FDCWD, path, flags, mode);
}

int openat(int dirfd, const char *path, int flags, ...)
{
    va_list ap;
    va_start(ap, flags);
    mode_t mode = (mode_t)va_arg(ap, int);
    va_end(ap);
    note(path);
    return (int)syscall(SYS_openat, dirfd, path, flags, mode);
}

int openat64(int dirfd, const char *path, int flags, ...)
{
    va_list ap;
    va_start(ap, flags);
    mode_t mode = (mode_t)va_arg(ap, int);
    va_end(ap);
    note(path);
    return (int)syscall(SYS_openat, dirfd, path, flags, mode);
}

FILE *fopen(const char *path, const char *mode)
{
    static FILE *(*real)(const char *, const char *);
    if (real == NULL) {
        real = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "fopen");
    }
    note(path);
    return real == NULL ? NULL : real(path, mode);
}
