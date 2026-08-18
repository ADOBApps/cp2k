//----------------------------------------------------------------------------/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
//----------------------------------------------------------------------------/

#ifndef MP2_IO_BRIDGE_H
#define MP2_IO_BRIDGE_H

#include <stdarg.h>

/**
 * \brief Print a formatted message to a Fortran I/O unit.
 * 
 * This function formats a message using printf-style syntax and sends it
 * to the specified Fortran unit number. The actual I/O is performed by
 * Fortran to ensure proper unit handling and consistency with CP2K's
 * logging system.
 * 
 * \param unit_nr  Fortran unit number (if <= 0, output is suppressed)
 * \param format   printf-style format string
 * \param ...      Arguments for the format string
 * 
 * \note This function does NOT flush the output buffer. For messages that
 *       require immediate flushing, use print_ri_info_flush().
 * \see print_ri_info_flush()
 */
void print_ri_info(int unit_nr, const char* format, ...);

/**
 * \brief Print a formatted message to a Fortran I/O unit and flush.
 * 
 * This function is identical to print_ri_info() but explicitly flushes
 * the I/O buffer after writing. Should ONLY be used for final or summary
 * messages, never inside hot loops, as FLUSH is a blocking syscall.
 * 
 * \param unit_nr  Fortran unit number (if <= 0, output is suppressed)
 * \param format   printf-style format string
 * \param ...      Arguments for the format string
 * 
 * \warning Calling this function inside a performance-critical loop will
 *          significantly degrade performance.
 * \see print_ri_info()
 */
void print_ri_info_flush(int unit_nr, const char* format, ...);

/**
 * \brief Print an error message to stderr.
 * 
 * This function writes error messages directly to stderr using fprintf(),
 * bypassing the Fortran I/O system. This ensures errors are visible even
 * on MPI ranks where RI_INFO output is silenced.
 * 
 * \param format   printf-style format string
 * \param ...      Arguments for the format string
 * 
 * \note Error messages are always written to stderr and are immediately
 *       flushed, independent of unit_nr.
 */
void print_error(const char* format, ...);

#endif MP2_IO_BRIDGE_H