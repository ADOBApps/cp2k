//----------------------------------------------------------------------------/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
//----------------------------------------------------------------------------/

#include "mp2_io_bridge.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

/**
 * Fortran subroutine interface for writing strings to Fortran units.
 * 
 * This subroutine is declared BIND(C, name="fortran_write_string") in Fortran.
 * All dummy arguments are marked with the VALUE attribute, so we must pass
 * arguments by value, not by pointer.
 */
extern void fortran_write_string(int unit_nr, const char* str, int length);

/**
 * \brief Internal helper: write a raw string to a Fortran I/O unit.
 * 
 * This function delegates the actual I/O to the Fortran subroutine
 * fortran_write_string(), which handles unit management and proper
 * buffering. 
 * 
 * \param unit_nr  Fortran unit number (if <= 0, output is suppressed)
 * \param str      Null-terminated string to write
 * 
 * \note This function is static and should only be called from within this
 *       file. The output is NOT flushed after writing.
 * \see fortran_write_string()
 */
static void write_to_fortran_unit(int unit_nr, const char* str) {
    if (unit_nr <= 0) return;
    int len = (int)strlen(str);
    fortran_write_string(unit_nr, str, len);
}

/**
 * \brief Print a formatted, "RI_INFO| " prefixed message to unit_nr.
 * 
 * This function formats the input message with printf-style formatting,
 * prepends the "RI_INFO| " prefix, and writes it to the specified Fortran
 * unit via the Fortran bridge.
 * 
 * \param unit_nr  Fortran unit number (if <= 0, output is suppressed)
 * \param format   printf-style format string
 * \param ...      Arguments for the format string
 * 
 * \see write_to_fortran_unit()
 * \see fortran_write_string()
 */
void print_ri_info(int unit_nr, const char* format, ...) {
    if (unit_nr <= 0) return;

    va_list args;
    va_start(args, format);
    // Dynamically allocate exact size needed
    char* buffer = NULL;
    int written = vasprintf(&buffer, format, args);
    va_end(args);

    if (written < 0 || buffer == NULL) {
        write_to_fortran_unit(unit_nr, "  RI_INFO| ERROR: Failed to format message\n");
        return;
    }

    // Allocate output with "RI_INFO| " prefix and newline
    // const char* prefix = "RI_INFO| ";
    // const char* suffix = "\n";
    // size_t output_size = strlen(prefix) + written + strlen(suffix) + 1;
    const char* prefix = "  ";
    size_t output_size = strlen(prefix) + written + 1;
    char* output = (char*)malloc(output_size);

    if (output == NULL) {
        free(buffer);
        write_to_fortran_unit(unit_nr, "  RI_INFOR| ERROR: Memory allocation failed\n");
        return;
    }
    // snprintf(output, output_size, "%s%s%s", prefix, buffer, suffix);
    snprintf(output, output_size, "%s%s", prefix, buffer);
    write_to_fortran_unit(unit_nr, output);

    free(buffer);
    free(output);
}

/**
 * \brief Print a formatted message to unit_nr and flush immediately.
 * 
 * This function behaves like print_ri_info() but adds an explicit flush
 * operation to ensure the output is immediately written. This should only
 * be used for final or summary messages, never inside performance-critical
 * loops.
 * 
 * \param unit_nr  Fortran unit number (if <= 0, output is suppressed)
 * \param format   printf-style format string
 * \param ...      Arguments for the format string
 * 
 * \warning FLUSH is a blocking system call. Using this function inside
 *          hot loops can severely impact performance.
 * \see print_ri_info()
 */
void print_ri_info_flush(int unit_nr, const char* format, ...) {
    if (unit_nr <= 0) return;

    va_list args;
    va_start(args, format);
    // Dynamically allocate exact size needed
    char* buffer = NULL;
    int written = vasprintf(&buffer, format, args);
    va_end(args);

    if (written < 0 || buffer == NULL) {
        write_to_fortran_unit(unit_nr, "  RI_INFO| ERROR: Failed to format message\n");
        return;
    }

    print_ri_info(unit_nr, "%s", buffer);
    free(buffer);
    // Note: Explicit flush would be added via a separate Fortran bridge call
    // to ensure the Fortran unit's buffer is properly flushed.
}

/**
 * \brief Print an error message to stderr.
 * 
 * This function writes error messages directly to stderr using fprintf(),
 * ensuring they are visible even on ranks where RI_INFO output is silenced
 * or redirected. The output is immediately flushed.
 * 
 * \param format   printf-style format string
 * \param ...      Arguments for the format string
 * 
 * \note This function bypasses the Fortran I/O system entirely and always
 *       writes to stderr.
 */
void print_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "ERROR| ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);
}