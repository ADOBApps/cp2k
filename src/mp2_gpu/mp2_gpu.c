/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "../mpiwrap/cp_mpi.h"
#include "gemm_c_api.h"

// I use it like a timer
#include "../offload/offload_library.h"

// Helper function to find integration group size
static int find_integ_group_size(int ngroup, int max_repl_group_size) {
    int integ_group_size = ngroup;

    if (max_repl_group_size < 1) {
        max_repl_group_size = 1;
    }

    if(max_repl_group_size > ngroup) {
        max_repl_group_size = ngroup;
    }

    int min_repl_group_size = ngroup / max_repl_group_size;

    if (min_repl_group_size < 1) {
        min_repl_group_size = 1;
    }

    if (min_repl_group_size > ngroup) {
        min_repl_group_size = ngroup;
    }

    // Find smallest divisor >= min_repl_group_size
    for (int i = min_repl_group_size; i <= max_repl_group_size; i++) {
        if (ngroup % i == 0) {
            integ_group_size = i;
            break;
        }
    }
    return integ_group_size;
}

/**
 * Helper: modulo operation that works like fortran MOD,
 * ensuring non-negative results.
 */
static int modulo_frotran(int a, int b) {
    int result = a % b;
    if (result < 0) {
        result += b;
    }
    return result;
}

void c_mp2_ri_get_integ_group_size(
    int* integ_group_size_out,
    int* ngroup_out,
    int* num_integ_group_out,
    int ngroup,
    int num_integ_group,
    int integ_group_size,
    double mp2_memory,
    int homo,
    int virtual,
    int dimen_RI,
    int maxsize_gd_array,
    int maxsize_gd_B_virtual,
    int maxval_gd_B_virtual,
    int maxval_virtual,
    int max_homo
){
    // Local variables
    int block_size = 1;
    int max_repl_group_size = 1;
    
    // Memory calculation variables
    double mem_real = 0.0;
    double mem_base = 0.0;
    double mem_per_blk = 0.0;
    double mem_per_repl = 0.0;
    double mem_per_repl_blk = 0.0;
    double mem_min = 0.0;
    double factor = 0.0;

    mem_real = mp2_memory;
    
    // BIB_C_copy: MAX(MAX(homo*maxsize(gd_array_sizes)), dimen_RI) * maxsize(gd_B_virtual_sizes)
    double max_homo_gd = (double)homo * maxsize_gd_array;
    double max_compare = (max_homo_gd > (double)dimen_RI) ? max_homo_gd : (double)dimen_RI;
    mem_per_repl += max_compare * maxsize_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C: SUM(homo*maxsize(gd_B_virtual_sizes)) * maxsize(gd_array_sizes)
    double sum_homo_gd_B = (double)homo * maxsize_gd_B_virtual;
    mem_per_repl += sum_homo_gd_B * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C_rec: maxsize(gd_B_virtual_sizes) * maxsize(gd_array_sizes)
    mem_per_repl_blk += (double)maxval_gd_B_virtual * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // local_i_aL+local_j_aL: 2 * maxsize(gd_B_virtual_sizes) * dimen_RI
    mem_per_blk += 2.0 * maxval_gd_B_virtual * (double)dimen_RI * 8.0 / (1024.0 * 1024.0);
    
    // local_ab: MAX(virtual*maxsize(gd_B_virtual_sizes))
    double max_virtual_gd_B = (double)virtual * maxsize_gd_B_virtual;

    mem_base += max_virtual_gd_B * 8.0 / (1024.0 * 1024.0);
    
    // external_ab/external_i_aL: MAX(dimen_RI, max_virtual) * maxsize(gd_B_virtual_sizes)
    int max_dim = (dimen_RI > maxval_virtual) ? dimen_RI : maxval_virtual;
    mem_base += (double)max_dim * maxval_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    
    block_size = (int)sqrt((double)homo);
    // block_size = MAX(1, MIN(FLOOR(SQRT(REAL(MINVAL(homo), KIND=dp))), FLOOR(MINVAL(homo)/SQRT(2.0_dp*ngroup))))
    int block_size_pre = (int)(homo / sqrt(2.0 * ngroup));
    block_size = (block_size_pre < block_size) ? block_size_pre : block_size;
    block_size = (block_size < 1) ? 1 : block_size;
    
    mem_min = mem_base + mem_per_repl + (mem_per_blk + mem_per_repl_blk) * block_size;
    
    // Using printf for now - would use CP2K logging in production
    printf("RI_INFO| Minimum available memory per MPI process: %9.2f MiB\n", mem_real);
    printf("RI_INFO| Minimum required memory per MPI process: %9.2f MiB\n", mem_min);
    
    // Calculate factor for communication model
    // factor = SUM(homovirtual) - SUM((MAX(homo)/block_size + block_size - 2)*homovirtual)/ngroup
    double factor_homo = (double)homo * virtual;
    
    double sum_factor = ((double)max_homo / block_size + block_size - 2.0) * homo * virtual;
    factor = factor_homo - sum_factor / ngroup;
    
    // Determine integration group size
    integ_group_size = ngroup;  // Default
    
    if (factor <= 0.0) {
        // Calculate max replication group size
        double numerator = mem_real - mem_base - mem_per_blk * block_size;
        double denominator = mem_per_repl + mem_per_repl_blk * block_size;
            
        if (denominator > 0.0) {
            max_repl_group_size = (int)(numerator / denominator);
        } else {
            max_repl_group_size = 1;
        }
            
        // Clamp
        if (max_repl_group_size < 1) max_repl_group_size = 1;
        if (max_repl_group_size > ngroup) max_repl_group_size = ngroup;
            
        // Find integration group size
        integ_group_size = find_integ_group_size(ngroup, max_repl_group_size);
    }
    
    printf("RI_INFO| Group size for integral replication: %6d\n", integ_group_size);
    fflush(stdout);
    
    
    // Compute num_integ_group
    num_integ_group = ngroup / integ_group_size;
    
    // Return values
    *integ_group_size_out = integ_group_size;
    *ngroup_out = ngroup;
    *num_integ_group_out = num_integ_group;
}


void c_mp2_ri_create_group(
    int* comm_exchange_out,
    int* comm_rep_out,
    int* ranges_info_array,
    int* integ_group_pos2color_sub,
    int my_group_L_start,
    int my_group_L_end,
    int comm_all,
    int para_env_sub_comm,
    int color_sub,
    int integ_group_size,
    int num_integ_group,
    int my_group_L_size
) {
    // Convert Fortran MPI communicators to C MPI communicators
    cp_mpi_comm_t comm_para_env_c_comm = cp_mpi_comm_f2c(comm_all);
    cp_mpi_comm_t comm_para_env_sub_c_comm = cp_mpi_comm_f2c(para_env_sub_comm);

    *comm_exchange_out = comm_para_env_c_comm;

    printf("Get cp_mpi_comm_rank 1st time\n");
    fflush(stdout);
    // Get rank and size of the sub-communicator
    int para_env_rank = cp_mpi_comm_rank(comm_para_env_c_comm);
    
    printf("Get cp_mpi_comm_rank 2nd time\n");
    fflush(stdout);
    int para_env_sub_rank = cp_mpi_comm_rank(comm_para_env_sub_c_comm);

    // Local variables
    cp_mpi_comm_t comm_exchange_c = cp_mpi_comm_null;
    cp_mpi_comm_t comm_rep_c = cp_mpi_comm_null;

    int comm_exchange_rank = 0;
    int comm_exchange_size = 0;
    int comm_rep_rank = 0;
    int comm_rep_size = 0;

    int my_new_group_L_size = my_group_L_size;

    int sub_sub_color_exchange = para_env_sub_rank * num_integ_group + color_sub / integ_group_size;

    // Split the world communicator
    // Use the rank as ket for consistent ordering
    int exchange_key = para_env_rank;

    // Create the exchange communicator
    //
    cp_mpi_comm_split(comm_para_env_c_comm, sub_sub_color_exchange, exchange_key, &comm_exchange_c);
    *comm_exchange_out = cp_mpi_comm_c2f(comm_exchange_c); // convert back to Fortran communicator

    // Get info about exchange communicator
    printf("Get cp_mpi_comm_rank 3rd time\n");
    printf("Get cp_mpi_comm_rank call: color %d\n", sub_sub_color_exchange);
    printf("Get cp_mpi_comm_rank call: key %d\n", exchange_key);
    printf("Get cp_mpi_comm_rank call: comm_exchange_out %p\n", (void*)comm_exchange_out);
    fflush(stdout);

    comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    printf("Get cp_mpi_comm_rank 3.1 time: comm_exchange_rank %d\n", comm_exchange_rank);
    fflush(stdout);
    comm_exchange_size = cp_mpi_comm_size(comm_exchange_c);

    offload_timeset("mp2_ri_create_group\0");
    int sub_sub_color = para_env_sub_rank * comm_exchange_size + comm_exchange_rank;

    // Create replication communicator
    cp_mpi_comm_split(comm_para_env_c_comm, sub_sub_color, exchange_key, &comm_rep_c);

    // Assign replication communicator var
    // *comm_rep_out = comm_rep_c;
    *comm_rep_out = cp_mpi_comm_c2f(comm_rep_c);

    // Get info about replication communicator
    printf("Get cp_mpi_comm_rank 4th time\n");
    fflush(stdout);
    comm_rep_rank = cp_mpi_comm_rank(comm_rep_c);
    comm_rep_size = cp_mpi_comm_size(comm_rep_c);

    // Allocate arrays for gathering replication infor
    int* rep_sizes_array = (int*)malloc(comm_rep_size * sizeof(int));
    int* rep_starts_array = (int*)malloc(comm_rep_size * sizeof(int));
    int* rep_ends_array = (int*)malloc(comm_rep_size * sizeof(int));

    cp_mpi_allgather_int(&my_group_L_size, 1, rep_sizes_array, 1, comm_rep_c);
    cp_mpi_allgather_int(&my_group_L_start, 1, rep_starts_array, 1, comm_rep_c);
    cp_mpi_allgather_int(&my_group_L_end, 1, rep_ends_array, 1, comm_rep_c);

    // Allocate my_ifno array (4 x comm_rep_size)
    int* my_info = (int*)malloc(4 * comm_rep_size * sizeof(int));
    
    // Info of this process
    my_info[0 * comm_rep_size + 0] = my_group_L_start; // start
    my_info[1 * comm_rep_size + 0] = my_group_L_end; // end
    my_info[2 * comm_rep_size + 0] = 1; // local_start
    my_info[3 * comm_rep_size + 0] = my_group_L_size; // local_end

    my_new_group_L_size = my_group_L_size;

    // Loop ove other processes in replication group
    for (int proc_shift = 1; proc_shift < comm_rep_size; proc_shift++) {
        int proc_receive = modulo_frotran(comm_rep_rank - proc_shift, comm_rep_size);
        
        // Update new group size
        my_new_group_L_size += rep_sizes_array[proc_receive];
        my_info[0 * comm_rep_size + proc_shift] = rep_starts_array[proc_receive]; // start
        my_info[1 * comm_rep_size + proc_shift] = rep_ends_array[proc_receive]; // end
        my_info[2 * comm_rep_size + proc_shift] = my_info[3 * comm_rep_size + proc_shift - 1] + 1; // local_start
        my_info[3 * comm_rep_size + proc_shift] = my_new_group_L_size; // local_end
    }

    // Allocate ranges_info_array as a flat array (4 x comm_rep_size x comm_exchange_size)
    int* new_sizes_array = (int*)malloc(comm_exchange_size * sizeof(int));

    cp_mpi_allgather_int(&my_new_group_L_size, 1, new_sizes_array, 1, comm_exchange_c);

    // Gather my_info from all processes in the exchange communicator
    int my_info_size = 4 * comm_rep_size;

    cp_mpi_allgather_int(my_info, my_info_size, ranges_info_array, my_info_size, comm_exchange_c);

    free(rep_sizes_array);
    free(rep_starts_array);
    free(rep_ends_array);
    cp_mpi_allgather_int(&color_sub, 1, integ_group_pos2color_sub, 1, comm_exchange_c);
    cp_mpi_allgather_int(&my_new_group_L_size, 1, new_sizes_array, 1, comm_exchange_c);

    // DEALLOCATE (new_sizes_array)
    free(new_sizes_array);

    // time stop
    offload_timestop();
}

double* c_replicate_iaK_2intgroup(
    double* BIb_C,
    int BIb_C_L_size,
    int comm_exchange,
    int comm_rep,
    int homo,
    int max_L_size,
    int my_B_size,
    int my_group_L_size,
    const int* ranges_info_array
) {
    cp_mpi_comm_t comm_exchange_c = cp_mpi_comm_f2c(comm_exchange);
    cp_mpi_comm_t comm_rep_c = cp_mpi_comm_f2c(comm_rep);
    
    int comm_rep_size = cp_mpi_comm_size(comm_rep_c);
    
    printf("Get cp_mpi_comm_rank 5th time\n");
    fflush(stdout);
    int comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    
    printf("Get cp_mpi_comm_rank 6th time\n");
    fflush(stdout);
    int comm_rep_rank = cp_mpi_comm_rank(comm_rep_c);

    offload_timeset("replicate_iaK_2intgroup\0");

    // Get current BIb_C dimensions
    int current_L_size = BIb_C_L_size;

    // Allocate copy buffer: [L][virtual][occupied]
    size_t copy_size = (size_t)max_L_size * my_B_size * homo;
    printf("6.1th f m: copy_size: %zu\n", copy_size);
    fflush(stdout);
    double* BIb_C_copy = (double*)calloc(copy_size, sizeof(double));

    if (BIb_C_copy == NULL) {
        fprintf(stderr, "Error: cannot be assigned memory to BIb_C_copy");
    }

    // copy data from old BIb_C to copy buffer
    for (int i = 0; i < homo; i++) {
        for (int j = 0; j < my_B_size; j++) {
            size_t src_idx = ((size_t)i * my_B_size + j) * current_L_size;
            size_t dst_idx = ((size_t)i * my_B_size + j) * max_L_size;
            // memcpy(&BIb_C_copy[dst_idx], &(*BIb_C)[src_idx], current_L_size * sizeof(double));
            memcpy(&BIb_C_copy[dst_idx], &(BIb_C)[src_idx], current_L_size * sizeof(double));
        }
    }
    // free(*BIb_C);
    // free(BIb_C);
    
    // Allocate gather buffer: [comm_rep_size][max_L_size][my_B_size][homo]
    size_t gather_size = (size_t)comm_rep_size * max_L_size * my_B_size * homo;

    double* BIb_C_gather = (double*)calloc(gather_size, sizeof(double));
    
    int send_count = (int)(max_L_size * my_B_size * homo);

    printf("6.3th f m: cp_mpi_allgather_double: comm_rep_size=%d, max_L_size=%d, my_B_size=%d, homo=%d\n",
        comm_rep_size,
        max_L_size,
        my_B_size,
        homo
    );
    cp_mpi_allgather_double(BIb_C_copy, send_count, BIb_C_gather, send_count, comm_rep_c);
    
    // Free copy buffer
    free(BIb_C_copy);
    
    // Reorder and store replicated data

    // Allocate new BIb_C: [my_group_L_size][my_B_size][homo]
    size_t new_size = (size_t)my_group_L_size * my_B_size * homo;
    double* BIb_C_new = (double*)calloc(new_size, sizeof(double));
    
    // Reorder data using ranges_info_array
    for (int proc_shift = 0; proc_shift < comm_rep_size; proc_shift++) {
        // Which process are we getting data from?
        int proc_receive = (comm_rep_rank - proc_shift) % comm_rep_size;
        if (proc_receive < 0) {
            proc_receive += comm_rep_size;
        }

        // ranges_info_array flat layout is (4, comm_rep_size, comm_exchange_size),
        // matching the packing done in c_mp2_ri_create_group:
        //   idx(dim, proc_shift, exchange_rank) =
        //       exchange_rank * (4 * comm_rep_size) + dim * comm_rep_size + proc_shift
        // dim=2 (0-based): local_start, dim=3 (0-based): local_end
        int start_point = ranges_info_array[
            comm_exchange_rank * 4 * comm_rep_size +
            2 * comm_rep_size + proc_shift
        ];
        int end_point = ranges_info_array[
            comm_exchange_rank * 4 * comm_rep_size +
            3 * comm_rep_size + proc_shift
        ];
        int L_size = end_point - start_point + 1;
        
        // Calculate offsets
        // Each process's data in gather buffer
        size_t gather_offset = (size_t)proc_receive * max_L_size * my_B_size * homo;

        // Copy data from gather buffer to output
        for (int i = 0; i < homo; i++) {
            for (int a = 0; a < my_B_size; a++) {
                // Source: gather buffer at (proc_receive, i, a, L)
                size_t src = gather_offset + ((size_t)i * my_B_size + a) * max_L_size;
                
                // Destination: new BIb_C at (i, a, L) from start_point to end_point
                size_t dst = ((size_t)i * my_B_size + a) * my_group_L_size + (start_point - 1);
                
                // Copy L_size doubles
                memcpy(
                    &BIb_C_new[dst],
                    &BIb_C_gather[src], 
                    L_size * sizeof(double)
                );
            }
        }
    }
    
    // Free gather buffer
    free(BIb_C_gather);
    
    // Return new BIb_C
    // *BIb_C = BIb_C_new; //INTOUT
    // BIb_C = BIb_C_new; //INTOUT
    BIb_C_L_size = my_group_L_size;

    // stop the timer
    offload_timestop();
    return BIb_C_new;
}


void c_mp2_ri_allocate_no_blk(
    double** local_ab, int virtual, int my_B_size
) {
    //Start timer
    offload_timeset("mp2_ri_allocate_no_blk\0");

    // ALLOCATE(local_ab(virtual(ispin), my_B_size(jspin)))
    // local_ab = 0.0_dp

    // *local_ab = (double*)calloc((size_t)virtual[i_c] * my_B_size[j_c], sizeof(double));
    *local_ab = (double*)calloc((size_t)virtual * my_B_size, sizeof(double));

    //stopo timer
    offload_timestop();
}

void c_mp2_ri_get_block_size(
    int* block_size,
    int* ngroup_out,
    double** buffer_1D,
    int user_block_size,
    cp_mpi_comm_t para_env_comm,
    cp_mpi_comm_t para_env_sub_comm,
    int maxsize_gd_array,
    int maxval_gd_B_virtual,
    int homo,
    int maxval_virtual,
    int dimen_RI,
    int num_integ_group
) {
    //Start timer
    offload_timeset("mp2_ri_get_block_size\0");
    // STEP 1: Calculate ngroup
    int para_env_size = cp_mpi_comm_size(para_env_comm);
    int para_env_sub_size = cp_mpi_comm_size(para_env_sub_comm);
    int ngroup = para_env_size / para_env_sub_size;
    *ngroup_out = ngroup;

    // STEP 2: Get available memory
    int64_t mem_bytes = 0;
    // In fortran-side is call m_memory()
    double mem_real = (double)((mem_bytes + 1024*1024 - 1) / (1024*1024));
    cp_mpi_max_double(&mem_real, 1, para_env_comm);
    
    // STEP 3: Calculate memory components
    double mem_base = 0.0;
    double mem_per_blk = 0.0;
    double mem_per_repl_blk = 0.0;
    
    // external_ab
    // (condiction) ? true : flase
    int max_dim = (dimen_RI > maxval_virtual) ? dimen_RI : maxval_virtual;
    mem_base += (double)maxval_gd_B_virtual * max_dim * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C_rec
    mem_per_repl_blk += (double)maxval_gd_B_virtual * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // local_i_aL + local_j_aL
    mem_per_blk += 2.0 * maxval_gd_B_virtual * (double)dimen_RI * 8.0 / (1024.0 * 1024.0);
    
    // Copy to keep arrays contiguous
    mem_base += (double)maxval_gd_B_virtual * max_dim * 8.0 / (1024.0 * 1024.0);

    // STEP 4: Determine block size
    int best_block_size = 1;
    
    if (user_block_size > 0) {
        best_block_size = user_block_size;
    } else {
        double denominator = mem_per_blk + mem_per_repl_blk * ngroup / num_integ_group;
        if (denominator > 0.0) {
            best_block_size = (int)((mem_real - mem_base) / denominator);
        }
        if (best_block_size < 1) best_block_size = 1;
        
        // Loop to ensure valid block size
        while (1) {
            int num_IJ_blocks = 0;
            
            num_IJ_blocks = (homo - 1) / best_block_size;
            num_IJ_blocks = (num_IJ_blocks * num_IJ_blocks - num_IJ_blocks) / 2;
            
            if ((num_IJ_blocks >= ngroup && num_IJ_blocks > 0) || best_block_size == 1) {
                break;
            } else {
                best_block_size--;
            }
        }
        
        int sqrt_val = (int)sqrt((double)homo);
        best_block_size = (sqrt_val < best_block_size) ? sqrt_val : best_block_size;
    }
    
    *block_size = (best_block_size < 1) ? 1 : best_block_size;
    
    printf("RI_INFO| Block size: %6d\n", *block_size);
    fflush(stdout);

    // STEP 6: Allocate buffer
    int64_t buffer_size = 0;
    int64_t size1 = (int64_t)maxsize_gd_array * (*block_size);
    int64_t size2 = (int64_t)max_dim;
    int64_t max_size = (size1 > size2) ? size1 : size2;
    buffer_size = max_size * maxval_gd_B_virtual;
    
    *buffer_1D = (double*)malloc((size_t)buffer_size * sizeof(double));
    if (*buffer_1D == NULL) {
        fprintf(stderr, "Error: Failed to allocate buffer_1D of size %ld\n", buffer_size);
        exit(1);
    }

    // Stop timer
    offload_timestop();
}

void c_mp2_ri_communication(
    int homo, int block_size, int ngroup, 
    int color_sub, int* total_ij_pairs,
    int** ij_map, int* my_ij_pairs,
    int* total_ij_pairs_blocks_out
){
    // start timer
    offload_timeset("mp2_ri_communication\0");

    *total_ij_pairs = homo * (1 + homo) / 2;
    int num_IJ_blocks = homo / block_size - 1;

    int first_I_block = 1;
    int last_i_block = block_size * (num_IJ_blocks - 1);
    int last_J_block = block_size * (num_IJ_blocks + 1);

    // Count block pairs
    int ij_block_counter = 0;
    for (int iiB = first_I_block; iiB < last_i_block; iiB += block_size) {
        for (int jjB = iiB + block_size + 1; jjB < last_J_block; jjB += block_size) {
            ij_block_counter++;
        }
    }

    int total_ij_block = ij_block_counter;
    int num_block_per_group = total_ij_block / ngroup;
    int assigned_blocks = num_block_per_group * ngroup;
    int total_ij_pairs_blocks = assigned_blocks + ((*total_ij_pairs) - assigned_blocks * (block_size * block_size));

    /**
     * ALLOCATE (ij_marker(homo, homo))
     * ij_marker = .TRUE.
     * array row-major flattened 1D
     * According with some forums benefits by memory, single allocation and dynamic access 
     * bool* arr = malloc(rows * cols * sizeof(bool));
     * to access: arr[i * cols + j]
     */
    bool* ij_marker = (bool*)malloc(homo * homo * sizeof(bool));
    for (int i = 0; i < homo * homo; i++) {
        ij_marker[i] = true;
    }

    // ALLOCATE (ij_map(3, total_ij_pairs_blocks))
    // ij_map = 0
    // ======== FOLLOW THIS VAR AND DO-LOOPS RELATED
    *ij_map = (int*)calloc(3 * total_ij_pairs_blocks, sizeof(int));

    int ij_counter = 0;
    *my_ij_pairs = 0;

    for (int iiB = first_I_block; iiB <= last_i_block; iiB += block_size) {
        for (int jjB = iiB + block_size; jjB <= last_J_block; jjB += block_size) {
            // exit
            if (ij_counter + 1 > assigned_blocks) {break;}
            ij_counter++;

            // ij_marker(iiB:iiB + block_size - 1, jjB:jjB + block_size - 1) = .FALSE.
            // i = iiB - 1 (index 0 in C)
            for (int i = iiB - 1; i <= iiB + block_size - 1; i++) {
                // j = jjB - 1 (index 0 in C)
                for (int j = jjB - 1; j <= jjB + block_size - 1; j++) {
                    ij_marker[i * homo + j] = false;
                }
            }

            (*ij_map)[0 * total_ij_pairs_blocks + (ij_counter - 1)] = iiB;
            (*ij_map)[1 * total_ij_pairs_blocks + (ij_counter - 1)] = jjB;
            (*ij_map)[2 * total_ij_pairs_blocks + (ij_counter - 1)] = block_size;
            if ((ij_counter % ngroup) == color_sub) {
                (*my_ij_pairs)++;
            }

        }
    }

    for (int iiB = 1; iiB <= homo; iiB++) {
        for (int jjB = iiB; jjB <= homo; jjB++) {
            // to access: arr[i * cols + j]
            // 0-based in C-stlr
            if (ij_marker[(iiB - 1) * homo + (jjB - 1)]) {
                ij_counter++;
                (*ij_map)[0 * total_ij_pairs_blocks + (ij_counter - 1)] = iiB;
                (*ij_map)[1 * total_ij_pairs_blocks + (ij_counter - 1)] = jjB;
                (*ij_map)[2 * total_ij_pairs_blocks + (ij_counter - 1)] = 1;
                if ((ij_counter % ngroup) == color_sub) {
                    (*my_ij_pairs)++;
                }
            }
        }
    }
    free(ij_marker);

    if (block_size == 1) {
        printf("RI_INFO| Percentage of ij pairs communicated with block size 1: 100.0\n");
    } else {
        double percentage = 100.0 * (double)((*total_ij_pairs - assigned_blocks * (block_size * block_size))) /  (double)(*total_ij_pairs);
        printf("RI_INFO| Percentage of ij pairs communicated with block size 1: %.1f\n", percentage);
    }

    *total_ij_pairs_blocks_out = total_ij_pairs_blocks;

    // Stop timer
    offload_timestop();
}

void c_mp2_ri_allocate_blk(
    int dimen_RI,
    int my_B_size,
    int block_size,
    double** local_i_aL,
    double** local_j_aL
){
    offload_timeset("mp2_ri_allocate_blk\0");

    *local_i_aL = (double*)calloc(block_size * my_B_size * dimen_RI, sizeof(double));
    *local_j_aL = (double*)calloc(block_size * my_B_size * dimen_RI, sizeof(double));

    offload_timestop();
}

void fill_local_i_aL(
    double* local_i_aL,
    int local_i_aL_L_size,
    int local_i_aL_virtual,
    int local_i_aL_block,
    const int* ranges_info_array,
    int ranges_info_rep_size,
    const double* BIb_C_rec,
    int BIb_C_rec_L_size,
    int BIb_C_rec_virtual
){
    offload_timeset("fill_local_i_aL\0");

    for(int irep = 0; irep < ranges_info_rep_size; irep++) {
        /**
         * Fortran: (dim1, dim2m2, dim3) (colum-major)
         * C: [dim3][dim2][dim1] (row-major) 
         * flattenf like [block_size][my_B_size(ispin)][dimen_RI]
         * to access: arr[i * ranges_info_rep_size + irep]
         */
        int Lstart_pos = ranges_info_array[0 * ranges_info_rep_size + irep];
        int start_point = ranges_info_array[2 * ranges_info_rep_size + irep];
        int end_point = ranges_info_array[3 * ranges_info_rep_size + irep];

        // Number of L-index copy
        // start_point:end_point
        int L_size = end_point - start_point + 1; // Inclusive range

        /**
         * Copy data from BIb_c to local_i_aL
         * Fortran-side: local_i_aL(L, v_pos, i_block)
         * C-side: [(i_block * virtual + v_pos) * L_size + (L - 1)]
         * C-side: [(i_block * virtual * L_size) + (v_pos * L_size) + (L - 1)]
         * 
         * i_block: is the page
         * virtual: is the row
         * v_pos:   is the position in virtual block
         * L:       is the colum
         * (L - 1): is the position in L block
         * 
         * Index = block_offset + virtual offset + l_offset
         * Index = (i_block * virtual * L_size) + (v_pos * L_size) + (L - 1)
         */
        // local_i_aL(Lstart_pos:Lend_pos, :) = BIb_C_rec(start_point:end_point, :)
         for (int i_block = 0; i_block < local_i_aL_block; i_block++) {
            for (int v_pos = 0; v_pos < local_i_aL_virtual; v_pos++) {

                // Origin
                // BIb_C_rec[(i_block * BIb_C_rec_virtual + v_pos) * BIb_C_rec_L_size + (start_point - 1)]
                size_t src_idx = ((size_t)i_block * BIb_C_rec_virtual + v_pos) * BIb_C_rec_L_size + (start_point - 1);

                // Destination
                // local_i_aL[(i_block * local_i_aL_virtual + v_pos) * BIb_C_rec_L_size + (Lstart_pos - 1)]
                // size_t dest_idx = ((size_t)i_block * local_i_aL_virtual + v_pos) * BIb_C_rec_L_size + (Lstart_pos - 1);
                size_t dest_idx = ((size_t)i_block * local_i_aL_virtual + v_pos) * local_i_aL_L_size + (Lstart_pos - 1);

                // void *memcpy(void *dest, const void *src, size_t count);
                memcpy(&local_i_aL[dest_idx], &BIb_C_rec[src_idx], L_size * sizeof(double));
            }
         }
    }

    offload_timestop();
}

void calc_ri_mp2_energy(
    double *E_cou,
    double *E_ex,
    double *E_s,
    double *E_t,
    double *BIb_C,
    double mp2_memory,
    int user_block_size,
    int comm_all_f,
    int comm_sub_f,
    int color_sub,
    int* gd_array_sizes,           // gd_array_size
    int gd_array_sizes_size,      // gd_array_sizes_size
    const int* gd_B_virtual_sizes, // array of size gd_B_virtual_sizes
    int gd_B_virtual_sizes_size,
    const double *eigenval,
    int homo,
    int nmo, 
    int dimen_RI,
    int maxsize_gd_array,
    int maxsize_gd_B_virtual,
    int maxval_gd_B_virtual,
    bool calc_ex
) {
    const cp_mpi_comm_t comm_all = cp_mpi_comm_f2c(comm_all_f);
    const cp_mpi_comm_t comm_sub = cp_mpi_comm_f2c(comm_sub_f);

    gemm_ctx_t *ctx = gemm_ctx_create(GEMM_PU_HOST, GEMM_LIB_BLAS);

    offload_timeset("mp2_ri_gpw_compute_en\0");
    // Calcullate some var instead pass form fortran-side
    int rank_in_subgroup = cp_mpi_comm_rank(comm_sub);
    int my_B_size = gd_B_virtual_sizes[rank_in_subgroup];

    // int my_group_L_size = gd_array_sizes[rank_in_all];
    int my_group_L_size = gd_array_sizes[color_sub];

    int my_group_L_start = 1;
    for (int i = 0; i < color_sub; i++) {
        my_group_L_start += gd_array_sizes[i];
    }
    int my_group_L_end = my_group_L_start + my_group_L_size - 1;

    int my_B_virtual_start = 1;
    for (int i = 0; i < rank_in_subgroup; i++) {
        my_B_virtual_start += gd_B_virtual_sizes[i];
    }

    int virtual = nmo - homo;

    int para_env_size = cp_mpi_comm_size(comm_all);
    
    printf("Get cp_mpi_comm_rank 7th time\n");
    fflush(stdout);
    int para_env_sub_rank = cp_mpi_comm_rank(comm_sub);
    int para_env_sub_size = cp_mpi_comm_size(comm_sub);

    int ngroup = para_env_size / para_env_sub_size;

    int max_homo = homo;

    int integ_group_size = 0;
    int ngroup_out = 0;
    int num_integ_group = 0;
    int maxval_virtual = virtual;
    
    c_mp2_ri_get_integ_group_size(
        &integ_group_size,
        &ngroup_out,
        &num_integ_group,
        ngroup,
        0,
        0,
        mp2_memory,
        homo,
        virtual,
        dimen_RI,
        maxsize_gd_array,
        maxsize_gd_B_virtual,
        maxval_gd_B_virtual,
        maxval_virtual,
        max_homo
    );

    int comm_exchange_out = 0;
    int comm_rep_out = 0;
    int* ranges_info_array = NULL;
    int* integ_group_pos2color_sub = NULL;
    int* sizes_array_orig = NULL;

    int* gd_B_virtual_start = (int*)malloc(gd_B_virtual_sizes_size * sizeof(int));
    int* gd_B_virtual_end = (int*)malloc(gd_B_virtual_sizes_size * sizeof(int));

    // fill it
    int cumulative_val = 1;
    for (int i = 0; i < gd_B_virtual_sizes_size; i++) {
        gd_B_virtual_start[i] = cumulative_val;
        gd_B_virtual_end[i] = cumulative_val + gd_B_virtual_sizes[i] - 1;
        cumulative_val += gd_B_virtual_sizes[i];
    }

    // ranges_info_array dimensions: (4 x rep_size x exchange_size)
    int comm_rep_size = para_env_size / integ_group_size;
    int comm_exchange_size = integ_group_size;

    ranges_info_array = (int*)calloc(4 * comm_rep_size * comm_exchange_size, sizeof(int));
    integ_group_pos2color_sub = (int*)calloc(comm_exchange_size, sizeof(int));

    c_mp2_ri_create_group(
        &comm_exchange_out,
        &comm_rep_out,
        ranges_info_array,
        integ_group_pos2color_sub,
        my_group_L_start,
        my_group_L_end,
        comm_all,
        comm_sub,
        color_sub,
        integ_group_size,
        num_integ_group,
        my_group_L_size
    );

    cp_mpi_comm_t comm_exchange_c = cp_mpi_comm_f2c(comm_exchange_out);
    
    printf("Get cp_mpi_comm_rank 8th time\n");
    fflush(stdout);
    int comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    
    printf("Get cp_mpi_comm_rank 9th time\n");
    fflush(stdout);
    int tag = 42;

    double my_E_cou = 0.0;
    double my_E_ex = 0.0;
    double my_E_s = 0.0;
    double my_E_t = 0.0;

    // maxmum L size across all processes in exchange communicator
    int max_L_size = 0;
    for (int i = 0; i < gd_array_sizes_size; i++) {
        if (gd_array_sizes[i] > max_L_size) {
            max_L_size = gd_array_sizes[i];
        }
    }

    printf("DEBUG: BIb_C in calc_ri_mp2_energy = %p\n", (void*)BIb_C);
    fflush(stdout);
    
    if (BIb_C == NULL) {
        fprintf(stderr, "ERROR: BIb_C is NULL in calc_ri_mp2_energy!\n");
        return;
    }
    
    // Try to read first few values
    // printf("DEBUG: BIb_C[0] = %f\n", BIb_C[0]);
    // printf("DEBUG: BIb_C[1] = %f\n", BIb_C[1]);
    // fflush(stdout);

    double* replicated_BIb_C = c_replicate_iaK_2intgroup(
        BIb_C,
        my_group_L_size,
        comm_exchange_out,
        comm_rep_out,
        homo,
        max_L_size,
        my_B_size,
        my_group_L_size,
        ranges_info_array
    );
            
    // Allocate local arrays (no block) 
    double* local_ab = NULL;

    c_mp2_ri_allocate_no_blk(
        &local_ab, virtual, my_B_size
    );
    
    // Get block size and allocate buffer
    int block_size = 0;
    int ngroup_out2 = 0;
    double* buffer_1D = NULL;

    c_mp2_ri_get_block_size(
        &block_size,
        &ngroup_out2,
        &buffer_1D,
        user_block_size,
        comm_all,
        comm_sub,
        maxsize_gd_array,
        maxval_gd_B_virtual,
        homo,
        maxval_virtual,
        dimen_RI,
        num_integ_group
    );
    
    // Communication pattern
    int total_ij_pairs = 0;
    int total_ij_pairs_blocks = 0;
    int* ij_map = NULL;
    int my_ij_pairs = 0;
    
    c_mp2_ri_communication(
        homo,
        block_size,
        ngroup,
        color_sub,
        &total_ij_pairs,
        &ij_map,
        &my_ij_pairs,
        &total_ij_pairs_blocks
    );
            
    // Gather my_ij_pairs from all processes in exchange communicator
    int* num_ij_pairs = (int*)malloc(comm_exchange_size * sizeof(int));

    //======================== Current point
    cp_mpi_allgather_int(&my_ij_pairs, 1, num_ij_pairs,
                  1, comm_all);
    
    int max_ij_pairs = my_ij_pairs;
    for (int p = 0; p < comm_exchange_size; p++) {
        if (num_ij_pairs[p] > max_ij_pairs) {
            max_ij_pairs = num_ij_pairs[p];
        }
    }
    
    // Allocate block arrays
    double* local_i_aL = NULL;
    double* local_j_aL = NULL;
    double sym_fac;
    // integral part
    double integral;
    double divi_part;
    
    c_mp2_ri_allocate_blk(
        dimen_RI, my_B_size, block_size,
        &local_i_aL, &local_j_aL
    );
            
    // Loop over ij pairs (the main computational loop)
    // Handle 2
    offload_timeset("mp2_ri_gpw_compute_en_RI_loop\0");

    for (int ij_index = 1; ij_index <= max_ij_pairs; ij_index++) {

        if (ij_index <= my_ij_pairs) {
            // Get i, j, and block_size for this pair
            int ij_counter = (ij_index - (color_sub > 0 ? 1 : 0)) * ngroup + color_sub;
            // In real code: get from ij_map
            int my_i = ij_map[0 * total_ij_pairs_blocks + ij_counter - 1];
            int my_j = ij_map[1 * total_ij_pairs_blocks + ij_counter - 1];
            int my_block_size = ij_map[2 * total_ij_pairs_blocks + ij_counter - 1];

            // fill local_i_aL and local_j_aL
            // call fill_local_i_aL

            // int L_size = gd_array_sizes[comm_exchange_rank];
            // int L_size = gd_array_sizes[color_sub];
            int L_size = my_group_L_size;
            // const int* ranges_info_array;
            int ranges_info_rep_size = comm_rep_size;

            double* BIb_C_i_offset = &replicated_BIb_C[((size_t)(my_i - 1) * my_B_size) * my_group_L_size];
            double* BIb_C_j_offset = &replicated_BIb_C[((size_t)(my_j - 1) * my_B_size) * my_group_L_size];

            fill_local_i_aL(
                local_i_aL,                 // Destination
                dimen_RI,                   // local_aL_L_size
                my_B_size,                  // local_aL_virtual
                my_block_size,              // local_aL_block
                ranges_info_array,          // ranges_info_array
                ranges_info_rep_size,
                // replicated_BIb_C,           // Source: BIb_C_rec
                BIb_C_i_offset,
                L_size,                     // BIb_C_rec_L_size
                my_B_size                   // BIb_C_rec_virtual
            );

            fill_local_i_aL(
                local_j_aL,                 // Destination
                dimen_RI,                   // local_aL_L_size
                my_B_size,                  // local_aL_virtual
                my_block_size,              // local_aL_block
                ranges_info_array,          // ranges_info_array
                ranges_info_rep_size,
                // replicated_BIb_C,           // Source: BIb_C_rec
                BIb_C_j_offset,
                L_size,                     // BIb_C_rec_L_size
                my_B_size                   // BIb_C_rec_virtual
            );

            // Handle 3
            offload_timeset("mp2_ri_gpw_compute_en_RI_comm\0");
            //====== use rec_B_virtual 
            for (int proc_shift = 1; proc_shift < comm_exchange_size; proc_shift++) {
                // Calculate send and receive process ranks
                int proc_send = (comm_exchange_rank + proc_shift) % comm_exchange_size;
                int proc_receive = (comm_exchange_rank - proc_shift + comm_exchange_size) % comm_exchange_size;

                //Get the number ij pairs for the sending process
                int send_ij_index = num_ij_pairs[proc_send];

                // Which should I use to get the L-size for the receiving process (rec_L_sizes)
                int rec_color_sub = integ_group_pos2color_sub[proc_receive];
                int rec_L_size = gd_array_sizes[rec_color_sub];

                // Get the L-size for the receiving process (rec_L_sizes)
                // int rec_L_size = gd_array_sizes[proc_receive];

                if (ij_index <= send_ij_index) {
                    // Calculate send indices for this ij pair
                    int correction_send = (integ_group_pos2color_sub[proc_send] > 0) ? 1 : 0;
                    // int ij_counter_send = (ij_index - 1) * ngroup + integ_group_pos2color_sub[proc_send];
                    int ij_counter_send = (ij_index - correction_send) * ngroup + integ_group_pos2color_sub[proc_send];

                    // Assert bounds
                    assert(ij_counter_send >= 1 && ij_counter_send <= total_ij_pairs_blocks);

                    int send_i = ij_map[0 * total_ij_pairs_blocks + ij_counter_send - 1];
                    int send_j = ij_map[1 * total_ij_pairs_blocks + ij_counter_send - 1];

                    assert(send_i >= 1 && send_i <= homo);
                    assert(send_j >= 1 && send_j <= homo);

                    size_t rec_size_i = (size_t)rec_L_size * my_B_size * my_block_size;
                    double* BI_C_rec_i = buffer_1D;
                    // BI_C_rec = 0.0_dp memset can work?
                    // === CHECK
                    // memset(BI_C_rec_i, 0, rec_L_size * sizeof(double));
                    memset(BI_C_rec_i, 0, rec_size_i * sizeof(double));

                    // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                    //                        proc_send, BI_C_rec, proc_receive, tag)
                    size_t send_size_i = (size_t)my_group_L_size * my_B_size * my_block_size;
                    // size_t offser_i = ((size_t)(send_i - 1) * my_B_size[i]);
                    // double* send_buffer_i = &BIb_C[((size_t)(send_i - 1) * my_B_size) * my_group_L_size];
                    // Using the replicated_BIb_C instead of pure BIb_C
                    double* send_buffer_i = &replicated_BIb_C[((size_t)(send_i - 1) * my_B_size) * my_group_L_size];

                    cp_mpi_sendrecv_double(
                        send_buffer_i,
                        (int)send_size_i,
                        proc_send,
                        tag,
                        BI_C_rec_i,
                        (int)rec_size_i,
                        proc_receive,
                        tag,
                        comm_exchange_c
                    );

                    // I use directly the var/values instead of temporal
                    // to avoid clean the initial
                    fill_local_i_aL(
                        local_i_aL,                    // Destination
                        dimen_RI,                      // local_i_aL_L_size
                        my_B_size,                     // local_i_aL_virtual or my_B_size[1]
                        my_block_size,                 // local_i_aL_block
                        ranges_info_array,             // ranges_info_array
                        comm_rep_size,                 // ranges_info_rep_size
                        BI_C_rec_i,                    // Source: BIb_C_rec
                        rec_L_size,                    // BIb_C_rec_L_size
                        my_B_size                      // BIb_C_rec_virtual
                    );

                    // Occupied j: send and receive data
                    size_t rec_size = (size_t)rec_L_size * my_B_size * my_block_size;
                    double* BI_C_rec = buffer_1D + rec_size_i; // Start of receive vuffer for j
                    memset(BI_C_rec, 0, rec_size * sizeof(double));

                    // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                    //                        proc_send, BI_C_rec, proc_receive, tag)
                    size_t send_size = (size_t)my_group_L_size * my_B_size * my_block_size;
                    // size_t offser = ((size_t)(send - 1) * my_B_size[j]);
                    // double* send_buffer = &BIb_C[((size_t)(send - 1) * my_B_size) * my_group_L_size];
                    double* send_buffer = &replicated_BIb_C[((size_t)(send_j - 1) * my_B_size) * my_group_L_size];

                    cp_mpi_sendrecv_double(
                        send_buffer,
                        (int)send_size,
                        proc_send,
                        tag,
                        BI_C_rec,
                        (int)rec_size,
                        proc_receive,
                        tag,
                        comm_exchange_c
                    );

                    fill_local_i_aL(
                        local_j_aL,                      // Destination
                        dimen_RI,                      // local_i_aL_L_size
                        my_B_size,                     // local_i_aL_virtual
                        my_block_size,                 // local_i_aL_block
                        ranges_info_array,             // ranges_info_array
                        comm_rep_size,                 // ranges_info_rep_size
                        BI_C_rec,                      // Source: BIb_C_rec
                        rec_L_size,                    // BIb_C_rec_L_size
                        my_B_size                      // BIb_C_rec_virtual
                    );
                }
                else {
                    // No work to do - we only receive data
                    // OCCUPIED i: Receive data only

                    size_t rec_size_i = (size_t)rec_L_size * my_B_size * my_block_size;
                    double* BI_C_rec_i = buffer_1D;
                    memset(BI_C_rec_i, 0, rec_size_i * sizeof(double));

                    cp_mpi_recv_double(
                        BI_C_rec_i,                 // Receive buffer
                        (int)rec_size_i,            // Receive count
                        proc_receive,               // Source (shoul be proc_send?)
                        tag,                        // Receive tag
                        comm_exchange_c             // Communicator
                    );
                    
                    // Fill local_i_aL with received data
                    // local_i_aL(:, :, 1:my_block_size) = BI_C_rec_i(:, 1:my_B_size(ispin), 1:my_block_size)
                    fill_local_i_aL(
                        local_i_aL,                    // Destination
                        dimen_RI,                      // local_i_aL_L_size
                        my_B_size,                  // local_i_aL_virtual
                        my_block_size,                 // local_i_aL_block
                        ranges_info_array,             // ranges_info_array
                        comm_rep_size,                 // ranges_info_rep_size
                        BI_C_rec_i,                    // Source: BIb_C_rec
                        rec_L_size,                    // BIb_C_rec_L_size
                        my_B_size                      // BIb_C_rec_virtual
                    );
                    
                    // OCCUPIED j: Receive data only
                    size_t rec_size = (size_t)rec_L_size * my_B_size * my_block_size;
                    double* BI_C_rec = buffer_1D + rec_size_i;
                    memset(BI_C_rec, 0, rec_size * sizeof(double));

                    cp_mpi_recv_double(
                        BI_C_rec,                   // Receive buffer
                        (int)rec_size,              // Receive count
                        proc_receive,               // Source (shoul be proc_send?)
                        tag,                        // Receive tag
                        comm_exchange_c             // Communicator
                    );
                    
                    // Fill local_j_aL with received data
                    // local_j_aL(:, :, 1:my_block_size) = BI_C_rec(:, 1:my_B_size(jspin), 1:my_block_size)
                    fill_local_i_aL(
                        local_j_aL,                      // Destination
                        dimen_RI,                      // local_aL_L_size
                        my_B_size,                     // local_aL_virtual
                        my_block_size,                 // local_aL_block
                        ranges_info_array,             // ranges_info_array
                        comm_rep_size,                 // ranges_info_rep_size
                        BI_C_rec,                      // Source: BIb_C_rec
                        rec_L_size,                    // BIb_C_rec_L_size
                        my_B_size                      // BIb_C_rec_virtual
                    );
                }
            }
            // Handle 3
            offload_timestop();

            // loop over the block elements
            for (int iiB = 1; iiB <= my_block_size; iiB++) {
                for (int jjB = 1; jjB <= my_block_size; jjB++) {
                    // ====== EXPASION BLOCK
                    offload_timeset("mp2_ri_gpw_compute_en_RI_expansion\0");
                    // memset(local_ab, 0, (size_t)my_B_size * my_B_size * sizeof(double));
                    memset(local_ab, 0, (size_t)virtual * my_B_size * sizeof(double));
                    // Use pointer to replace ASSOCIATE block
                    double* my_local_i_aL = &local_i_aL[(size_t)(iiB - 1) * my_B_size * dimen_RI];
                    double* my_local_j_aL = &local_j_aL[(size_t)(jjB - 1) * my_B_size * dimen_RI];

                    printf(
                        "DEBUG iiB=%d jjB=%d my_i=%d my_j=%d my_block_size=%d my_B_size=%d dimen_RI=%d virtual=%d\n",
                        iiB,
                        jjB,
                        my_i,
                        my_j,
                        my_block_size,
                        my_B_size,
                        dimen_RI,
                        virtual
                    );
                    printf("DEBUG local_i_aL[0..2] = %g %g %g\n", local_i_aL[0], local_i_aL[1], local_i_aL[2]);
                    printf("DEBUG local_j_aL[0..2] = %g %g %g\n", local_j_aL[0], local_j_aL[1], local_j_aL[2]);
                    fflush(stdout);

                    gemm_ctx_dgemm(
                        ctx, 'T', 'N',
                        my_B_size, my_B_size, dimen_RI,
                        1.0, my_local_i_aL, dimen_RI,
                        my_local_j_aL, dimen_RI,
                        0.0, &local_ab[0], my_B_size
                    );

                    // Collect data from other processes in the subgroup
                    for (int proc_shift = 1; proc_shift < para_env_sub_size; proc_shift++) {
                        int proc_send = (para_env_sub_rank + proc_shift) % para_env_sub_size;
                        int proc_receive = (para_env_sub_rank - proc_shift + para_env_sub_size) % para_env_sub_size;
                        
                        // Get virtual ranges for receiving process
                        int rec_B_virtual_start = gd_B_virtual_start[proc_receive];
                        int rec_B_virtual_end = gd_B_virtual_end[proc_receive];
                        int rec_B_size = rec_B_virtual_end - rec_B_virtual_start + 1;
                        
                        // Allocate external_i_aL in buffer_1D
                        size_t ext_size = (size_t)dimen_RI * rec_B_size;
                        double* external_i_aL = buffer_1D;
                        memset(external_i_aL, 0, ext_size * sizeof(double));
                        
                        // Send my_local_i_aL to proc_send, receive into external_i_aL from proc_receive
                        cp_mpi_sendrecv_double(
                            my_local_i_aL,                      // Send buffer
                            (int)(dimen_RI * my_B_size),     // Send count
                            proc_send,                          // Destination
                            tag,                                // Send tag
                            external_i_aL,                      // Receive buffer
                            (int)ext_size,                      // Receive count
                            proc_receive,                       // Source
                            tag,                                // Receive tag
                            comm_sub                            // Communicator
                        );

                        gemm_ctx_dgemm(
                            ctx, 'T', 'N',
                            rec_B_size,
                            my_B_size,
                            dimen_RI,
                            1.0,
                            external_i_aL,
                            dimen_RI,
                            my_local_j_aL,
                            dimen_RI,
                            1.0,
                            &local_ab[(rec_B_virtual_start - 1) * my_B_size],
                            my_B_size
                        );
                    }

                    offload_timeset("mp2_ri_gpw_compute_en_RI_ener\0");
                    // Calculate Coulomb only MP2
                    sym_fac = (my_i == my_j) ? 1.0 : 2.0;

                    // DO b = 1, my_B_size(jspin)
                    for (int b = 0; b < my_B_size; b++) {
                        int b_global = b + my_B_virtual_start - 1;

                        // DO a = 1, virtual(ispin)
                        for (int a = 0; a < virtual; a++) {
                            integral = local_ab[a * my_B_size + b];
                            divi_part = eigenval[(homo + a)] + 
                                eigenval[(homo + b_global)] -
                                eigenval[(my_i + iiB - 2)] -
                                eigenval[(my_j + jjB - 2)];
                            my_E_cou -= sym_fac * 2.0 * integral * integral / divi_part;
                        }
                    }
                    if (calc_ex) {
                        for (int b = 0; b < my_B_size; b++) {
                            // assert((b + 1) == my_B_size);
                            // int b_global = b + my_B_virtual_start - 1;
                            int b_global = b + my_B_virtual_start - 1;
                            for (int a = 0; a < my_B_size; a++) {
                                int a_global = a + my_B_virtual_start - 1;
                                double denom = eigenval[(homo + a_global)] +
                                               eigenval[(homo + b_global)] -
                                               eigenval[(my_i + iiB - 2)] -
                                               eigenval[(my_j + jjB - 2)];
                                my_E_ex += sym_fac * local_ab[a_global * my_B_size + b] *
                                           local_ab[b_global * my_B_size + a] / denom;
                            }
                        }

                        // External contribution: exchange local_ab slices with other ranks in the subgroup.
                        for (int proc_shift = 1; proc_shift < para_env_sub_size; proc_shift++) {
                            int proc_send = (para_env_sub_rank + proc_shift) % para_env_sub_size;
                            int proc_receive = (para_env_sub_rank - proc_shift + para_env_sub_size) % para_env_sub_size;

                            int rec_B_virtual_start_ex = gd_B_virtual_start[proc_receive];
                            int rec_B_virtual_end_ex = gd_B_virtual_end[proc_receive];
                            int rec_B_size_ex = rec_B_virtual_end_ex - rec_B_virtual_start_ex + 1;

                            int send_B_virtual_start_ex = gd_B_virtual_start[proc_send];
                            int send_B_virtual_end_ex = gd_B_virtual_end[proc_send];

                            size_t ext_ab_size = (size_t)my_B_size * rec_B_size_ex;
                            double* external_ab = buffer_1D;
                            // assert(ext_ab_size <= (size_t)buffer_size); should I need returned buffer_size from c_mp2_ri_get_block_size?
                            memset(external_ab, 0, ext_ab_size * sizeof(double));

                            // Send local_ab(send_B_virtual_start:send_B_virtual_end, 1:my_B_size) to proc_send,
                            // receive into external_ab(1:my_B_size, 1:rec_B_size) from proc_receive.
                            size_t send_ab_size = (size_t)(send_B_virtual_end_ex - send_B_virtual_start_ex + 1) * my_B_size;
                            cp_mpi_sendrecv_double(
                                &local_ab[(send_B_virtual_start_ex - 1) * my_B_size],
                                (int)send_ab_size,
                                proc_send,
                                tag,
                                external_ab,
                                (int)ext_ab_size,
                                proc_receive,
                                tag,
                                comm_sub
                            );

                            for (int b = 0; b < my_B_size; b++) {
                                int b_global = b + my_B_virtual_start - 1;
                                for (int a = 0; a < rec_B_size_ex; a++) {
                                    int a_global = a + rec_B_virtual_start_ex - 1;
                                    double denom = eigenval[(homo + a_global)] +
                                                   eigenval[(homo + b_global)] -
                                                   eigenval[(my_i + iiB - 2)] -
                                                   eigenval[(my_j + jjB - 2)];
                                    // external_ab is laid out (my_B_size, rec_B_size) -> row-major [b][a]
                                    my_E_ex += sym_fac * local_ab[a_global * my_B_size + b] *
                                               external_ab[b * rec_B_size_ex + a] / denom;
                                }
                            }
                        }
                    }
                    offload_timestop();
                }
            }

        } else {
            int my_block_size = 1;
            offload_timeset("mp2_ri_gpw_compute_en_RI_comm\0");
            for (int proc_shift = 1; proc_shift < comm_exchange_size; proc_shift++) {
                // Calculate send and receive process ranks
                int proc_send = (comm_exchange_rank + proc_shift) % comm_exchange_size;
                int proc_receive = (comm_exchange_rank - proc_shift + comm_exchange_size) % comm_exchange_size;
                
                //Get the number ij pairs for the sending process
                int send_ij_index = num_ij_pairs[proc_send];

                // Get the L-size for the receiving process (rec_L_sizes)
                int rec_L_size = gd_array_sizes[proc_receive];

                if (ij_index <= send_ij_index) {
                    // Calculate send indices for this ij pair
                    int ij_counter_send = (ij_index - 1) * ngroup + integ_group_pos2color_sub[proc_send];
                    // int send_i = ij_map[0 * total_ij_pairs + ij_counter_send - 1];
                    // int send = ij_map[1 * total_ij_pairs + ij_counter_send - 1];
                    int send_i = ij_map[0 * total_ij_pairs_blocks + ij_counter_send - 1];
                    int send_j = ij_map[1 * total_ij_pairs_blocks + ij_counter_send - 1];
                    
                    // Occupied i: send and receive data
                    // Fortran BI_C_rec(1:rec_L_size, 1:my_B_size(ispin), 1:my_block_size)
                    // C: Flattened as [block][virtual][L]
                    // index: (i_block * virtual + a) * rec_L_size + (L - 1)

                    size_t rec_size_i = (size_t)rec_L_size * my_B_size * my_block_size;
                    double* BI_C_rec_i = buffer_1D;
                    // memset(BI_C_rec_i, 0, rec_L_size * sizeof(double));
                    memset(BI_C_rec_i, 0, rec_size_i * sizeof(double));

                    // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                    //                        proc_send, BI_C_rec, proc_receive, tag)
                    size_t send_size_i = (size_t)my_group_L_size * my_B_size * my_block_size;
                    // size_t offser_i = ((size_t)(send_i - 1) * my_B_size[i]);
                    // double* send_buffer_i = &BIb_C[((size_t)(send_i - 1) * my_B_size) * my_group_L_size];
                    double* send_buffer_i = &replicated_BIb_C[((size_t)(send_i - 1) * my_B_size) * my_group_L_size];

                    cp_mpi_send_double(
                        send_buffer_i,
                        (int)send_size_i,
                        proc_send,
                        tag,
                        comm_exchange_c
                    );

                    // Occupied j: send and receive data
                    size_t rec_size = (size_t)rec_L_size * my_B_size * my_block_size;
                    double* BI_C_rec = buffer_1D + rec_size_i; // Start of receive vuffer for j
                    // BI_C_rec = 0.0_dp memset can work?
                    memset(BI_C_rec, 0, rec_size * sizeof(double));

                    // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                    //                        proc_send, BI_C_rec, proc_receive, tag)
                    size_t send_size = (size_t)my_group_L_size * my_B_size * my_block_size;
                    // size_t offser = ((size_t)(send - 1) * my_B_size[j]);
                    // double* send_buffer = &BIb_C[((size_t)(send_j - 1) * my_B_size) * my_group_L_size];
                    double* send_buffer = &replicated_BIb_C[((size_t)(send_j - 1) * my_B_size) * my_group_L_size];

                    cp_mpi_send_double(
                        send_buffer,
                        (int)send_size,
                        proc_send,
                        tag,
                        comm_exchange_c
                    );
                }
            }
            offload_timestop();
        }
    }
    // Handle2
    offload_timestop();

    // Free replicared array
    free(replicated_BIb_C);
    free(local_ab);
    free(buffer_1D);
    free(ij_map);
    free(num_ij_pairs);
    free(local_i_aL);
    free(local_j_aL);

    free(ranges_info_array);
    free(integ_group_pos2color_sub);
    if (sizes_array_orig) free(sizes_array_orig);
    
    printf("Energy my_E_cou pre-cp_mpi_sum_double call: %f\n", my_E_cou);
    fflush(stdout);
    cp_mpi_sum_double(&my_E_cou, 1, comm_all);
    cp_mpi_sum_double(&my_E_ex, 1, comm_all);

    // Follow this var
    *E_cou += my_E_cou;
    printf("Energy my_E_cou: %f\n", my_E_cou);
    fflush(stdout);
    *E_ex += my_E_ex;
    *E_s += my_E_s;
    *E_t += my_E_t;
    cp_mpi_comm_free(&comm_exchange_c);

    // Destroy context for all libraries
    gemm_ctx_destroy(ctx);
    offload_timestop();
}

void calc_ri_mp2_energy_c_(
    double *E_cou,
    double *E_ex,
    double *E_s,
    double *E_t,
    double *BIb_C,
    double mp2_memory,
    int user_block_size,
    int comm_all_f,
    int comm_sub_f,
    int color_sub,
    int* gd_array_sizes,         // represents gd_array%sizes
    int gd_array_sizes_size,
    int* gd_B_virtual_sizes,     // array gd_B_virtual%sizes
    int gd_B_virtual_sizes_size,
    const double* eigenval,
    int homo,
    int nmo,
    int dimen_RI,
    int maxsize_gd_array,
    int maxsize_gd_B_virtual,
    int maxval_gd_B_virtual,
    bool calc_ex
) {
    // Just forward to the main function
    calc_ri_mp2_energy(
        E_cou,
        E_ex,
        E_s,
        E_t,
        BIb_C,
        mp2_memory,
        user_block_size,
        comm_all_f,
        comm_sub_f,
        color_sub,
        gd_array_sizes,
        gd_array_sizes_size,
        gd_B_virtual_sizes,
        gd_B_virtual_sizes_size,
        eigenval,
        homo,
        nmo, 
        dimen_RI,
        maxsize_gd_array,
        maxsize_gd_B_virtual,
        maxval_gd_B_virtual,
        calc_ex
    );
}
