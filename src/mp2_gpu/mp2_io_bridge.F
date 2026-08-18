!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

!--------------------------------------------------------------------------------------------------!
!> \brief: Write a string to a Fortran unit from C.
!>
!> This subroutine is called from C code to wirte strings to FOrtran I/O
!> units. All dummy arguments use the VALUE attribute, matching the
!> by-value calling convention of C. Units numbers <= 0 are treated as
!> "do not write", consistent with the C-side guard in write_to_fortran_unit.
!>
!> \param unit_nr Fortran unit number to write to
!> \param string  Character array containing the string to write
!> \param length  Lengthh of the string in characters
!>
!> \note This subroutine IS BIND(C) and intented for C interoperability.
!>       It writes without advancing to allow C-side control over
!>       line breaks and formatting.
!--------------------------------------------------------------------------------------------------!
SUBROUTINE fortran_write_string(unit_nr, string, length) &
    BIND(C, name="fortran_write_string")
    USE ISO_C_BINDING, ONLY: C_INT, C_CHAR
    INTEGER(C_INT), VALUE, INTENT(IN) :: unit_nr
    CHARACTER(KIND=C_CHAR), &
    DIMENSION(*), INTENT(IN)          :: string
    INTEGER(C_INT), VALUE, INTENT(IN) :: length

    CHARACTER(LEN=length)             :: str
    INTEGER                           :: i

    ! avoid invalid or disabled units
    IF (unit_nr <= 0) RETURN
    IF (length <= 0) RETURN

    ! Copy C character array to Fortran string
    DO i = 1, length
        str(i:i) = string(i)
    END DO

    ! Write to fortran unit
    WRITE(unit_nr, '(A)', ADVANCE='NO') str
    ! No flush here -- this subroutine may be called from a loop
END SUBROUTINE fortran_write_string