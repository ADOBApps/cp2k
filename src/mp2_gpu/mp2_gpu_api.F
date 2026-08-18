!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

MODULE mp2_gpu_api
   USE ISO_C_BINDING, ONLY   : C_DOUBLE, C_INT, C_BOOL, C_PTR, C_LOC, C_NULL_PTR, C_F_POINTER
   USE kinds, ONLY           : dp
   USE message_passing,  ONLY: mp_comm_type, mp_para_env_type
   USE mp2_types,        ONLY: mp2_type
   USE group_dist_types, ONLY: group_dist_d1_type, get_group_dist, maxsize

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: calc_ri_mp2_energy

CONTAINS

   SUBROUTINE calc_ri_mp2_energy( &
      E_cou, &
      E_ex, &
      E_s, &
      E_t, &
      BIb_C, &
      comm_all, &
      comm_sub, &
      eigenval, &
      homo, &
      nmo, &
      dimen_RI, &
      mp2_env, &
      gd_array, &
      gd_B_virtual, &
      color_sub, &
      calc_ex, &
      unit_nr)
      REAL(KIND=dp), INTENT(OUT)                         :: E_cou
      REAL(KIND=dp), INTENT(OUT)                         :: E_ex
      REAL(KIND=dp), INTENT(OUT)                         :: E_s
      REAL(KIND=dp), INTENT(OUT)                         :: E_t
      
      ! error actual argument for 'bib_c' at (1) must be a pointer or a valid target
      ! for the dummy pointer in a pointer assignment statement
      REAL(KIND=dp), DIMENSION(:, :, :), ALLOCATABLE, TARGET, &
                           INTENT(INOUT)                 :: BIb_C
      ! REAL(KIND=dp), DIMENSION(:, :, :), POINTER, &
      !                      CONTIGUOUS, INTENT(INOUT)     :: BIb_C
      CLASS(mp_comm_type), INTENT(IN)                    :: comm_all
      CLASS(mp_comm_type), INTENT(IN)                    :: comm_sub
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: eigenval
      INTEGER, INTENT(IN)                                :: homo
      INTEGER, INTENT(IN)                                :: nmo
      INTEGER, INTENT(IN)                                :: dimen_RI
      TYPE(mp2_type)                                     :: mp2_env
      TYPE(group_dist_d1_type), INTENT(IN)               :: gd_array
      TYPE(group_dist_d1_type), INTENT(IN)               :: gd_B_virtual
      INTEGER, INTENT(IN)                                :: color_sub
      LOGICAL, INTENT(IN)                                :: calc_ex
      INTEGER, INTENT(IN)                                :: unit_nr

      INTEGER(C_INT)  :: comm_all_handle
      INTEGER(C_INT)  :: comm_sub_handle
      INTEGER(C_INT)  :: maxsize_gd_array
      INTEGER(C_INT)  :: maxsize_gd_B_virtual
      INTEGER(C_INT)  :: maxval_gd_B_virtual
      ! INTEGER(C_INT), ALLOCATABLE, &
      ! TARGET          :: gd_B_virtual_sizes(:)
      REAL(C_DOUBLE)  :: mp2_memory
      INTEGER(C_INT)  :: user_block_size
      LOGICAL(C_BOOL) :: calc_ex_c
      INTEGER(C_INT)  :: unit_nr_c


      INTERFACE
         SUBROUTINE calc_ri_mp2_energy_c( &
            E_cou, &
            E_ex, &
            E_s, &
            E_t, &
            BIb_C, &
            mp2_memory, &
            user_block_size, &
            comm_all, &
            comm_sub, &
            color_sub, &
            gd_array_sizes, &
            gd_array_sizes_size, &
            gd_B_virtual_sizes, &
            gd_B_virtual_sizes_size, &
            eigenval, &
            homo, &
            nmo, &
            dimen_RI, &
            maxsize_gd_array, &
            maxsize_gd_B_virtual, &
            maxval_gd_B_virtual, &
            calc_ex_c, &
            unit_nr)
            IMPORT :: C_DOUBLE, C_INT, C_BOOL, C_PTR
            BIND(C, name="calc_ri_mp2_energy")
            REAL(C_DOUBLE)               :: E_cou
            REAL(C_DOUBLE)               :: E_ex
            REAL(C_DOUBLE)               :: E_s
            REAL(C_DOUBLE)               :: E_t
            TYPE(C_PTR), VALUE           :: BIb_C
            REAL(C_DOUBLE), VALUE        :: mp2_memory
            INTEGER(C_INT), VALUE        :: user_block_size
            INTEGER(C_INT), VALUE        :: comm_all
            INTEGER(C_INT), VALUE        :: comm_sub
            INTEGER(C_INT), VALUE        :: color_sub
            INTEGER(C_INT), DIMENSION(*) :: gd_array_sizes
            INTEGER(C_INT), VALUE        :: gd_array_sizes_size
            INTEGER(C_INT), DIMENSION(*) :: gd_B_virtual_sizes
            INTEGER(C_INT), VALUE        :: gd_B_virtual_sizes_size
            REAL(C_DOUBLE), DIMENSION(*) :: eigenval
            INTEGER(C_INT), VALUE        :: homo
            INTEGER(C_INT), VALUE        :: nmo
            INTEGER(C_INT), VALUE        :: dimen_RI
            INTEGER(C_INT), VALUE        :: maxsize_gd_array
            INTEGER(C_INT), VALUE        :: maxsize_gd_B_virtual
            INTEGER(C_INT), VALUE        :: maxval_gd_B_virtual
            LOGICAL(C_BOOL), VALUE       :: calc_ex_c
            INTEGER(C_INT), VALUE        :: unit_nr
         END SUBROUTINE calc_ri_mp2_energy_c
      END INTERFACE

      ! MPI communicators
      comm_all_handle = comm_all%get_handle()
      comm_sub_handle = comm_sub%get_handle()

      ! ALLOCATE(gd_B_virtual_sizes(SIZE(homo)))

      mp2_memory = mp2_env%mp2_memory
      user_block_size = mp2_env%ri_mp2%block_size

      ! set sizes
      maxsize_gd_array = maxsize(gd_array)
      maxsize_gd_B_virtual = maxsize(gd_B_virtual)
      maxval_gd_B_virtual = maxsize_gd_B_virtual

      calc_ex_c = calc_ex
      unit_nr_c = unit_nr

      ! === KEEP ONLY _j FAMILY VARS ===============
      CALL calc_ri_mp2_energy_c( &
         E_cou, &
         E_ex, &
         E_s, &
         E_t, &
         C_LOC(BIb_C), &
         mp2_memory, &
         user_block_size, &
         comm_all_handle, &
         comm_sub_handle, &
         color_sub, &
         gd_array%sizes, &
         SIZE(gd_array%sizes), &
         gd_B_virtual%sizes, &
         SIZE(gd_B_virtual%sizes), &
         eigenval, &
         homo, &
         nmo, &
         dimen_RI, &
         maxsize_gd_array, &
         maxsize_gd_B_virtual, &
         maxval_gd_B_virtual, &
         calc_ex_c, &
         unit_nr_c)
      ! NULLIFY(BIb_C)

   END SUBROUTINE calc_ri_mp2_energy
END MODULE mp2_gpu_api
