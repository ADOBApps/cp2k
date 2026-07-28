#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "../mpiwrap/cp_mpi.h"
#include "gemm_c_api.h"

// I use it like a timer
#include "../offload/offload_library.h"

// Helper function to compute the maximum value in an array
static int array_max(int *array, int size) {
    int max_value = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
        }
    }
    return max_value;
}

// Helper function to compute the sum of products
static int sum_of_product(const int *array1, const int *array2, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array1[i] * array2[i];
    }
    return sum;
}

// Helper function to find integration group size
static int find_integ_group_size(int ngroup, int max_repl_group_size) {
    int integ_group_size = ngroup;
    int min_repl_group_size = ngroup / max_repl_group_size;

    if (max_repl_group_size < 1) {
        max_repl_group_size = 1;
    }

    if(max_repl_group_size > ngroup) {
        max_repl_group_size = ngroup;

    }

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

/**
 * Helper: allocate 4D arrays as flat array with dimensions:
 * ranges_info_array[dim][rep_rank][exchange_rank]
 */
static int* allocate_ranges_info_array(int dim, int rep_size, int exchange_size) {
    // int total_size = 4 * rep_size * exchange_size;
    int total_size = dim * rep_size * exchange_size;
    int* array = (int*)calloc(total_size, sizeof(int));
    return array;
}

/**
 * Helper: index into ranges_info_array
 * ranges_info_array[dim * rep_size * exchange_size + rep_rank * exchange_size + exchange_rank]
 */
static int ranges_info_index(int dim, int rep_rank, int exchange_rank, int rep_size, int exchange_size) {
    return dim * rep_size * exchange_size + rep_rank * exchange_size + exchange_rank;
}

/**
 * Function to calculate the integration group size based on memory constraints and other parameters.
 * =========PARAMETERS=========
 *  // Output parameters
 * int* integ_group_size_out,
 * int* ngroup_out,
 * int* num_integ_group_out,
 * 
 * // Input parameters
 * int ngroup,  // from para_env%num_pr / para_env_sub%num_pr
 * int num_integ_group, // computed from ngroup/integ_grup_size
 * int integ_group_size,
 * 
 * // From mp2_env
 * double mp2_memory, // mp2_env%mp2_memory
 * int number_integration_groups, // mp2_env%ri_mp2%number_integration_groups
 * const int* homo,
 * int homo_size,
 * 
 * // from virtual
 * const int* virtual_arr,
 * int virtual_size,
 * int dimen_RI,
 * bool calc_forces,
 * int unit_nr,
 * 
 * // gd_array%sizes
 * const int* gd_array_sizes,
 * int gd_array_sizes_size,
 * 
 * // gd_B_virtual%sizes
 * const int* gd_B_virtual_sizes,
 * int gd_B_virtual_sizes_size,
 * 
 * // try to extract from gd_array and gd_B_virtual
 * int maxsize_gd_array, // Maximum size of gd_array
 * int maxsize_gd_B_virtual, // Maximum size of gd_B_virtual
 * int maxval_gd_B_virtual, // Maximum value in gd_B_virtual
 * int maxval_virtual, // Maximum value in virtual
 * int max_homo, // Maximum value in homo
 * int sum_homo_virtual, // Sum of products of homo and virtual
 * int product_homo, // Product of all values in homo
 * 
 * // SHOULD BE 1
 * int nspins, // 1 in this case; other case: SIZE(homo)
 * bool calc_group_size, // Flag to calculate group size
 */
void c_mp2_ri_get_integ_group_size(
    int* integ_group_size_out, int* ngroup_out, int* num_integ_group_out,
    int ngroup, int num_integ_group, int integ_group_size,
    double mp2_memory, int number_integration_groups, const int* homo,
    int homo_size, const int* virtual_arr, int virtual_size,
    int dimen_RI, bool calc_forces, int unit_nr,
    const int* gd_array_sizes, int gd_array_sizes_size,
    const int* gd_B_virtual_sizes, int gd_B_virtual_sizes_size,
    int maxsize_gd_array, int maxsize_gd_B_virtual, int maxval_gd_B_virtual,
    int maxval_virtual, int max_homo, int sum_homo_virtual,
    int product_homo, int nspins, bool calc_group_size, cp_mpi_comm_t comm) {
    // Local variables
    bool calc_group_size_local = calc_group_size;
    int block_size = 1;
    int max_repl_group_size = 1;
    int min_integ_group_size = 1;
    
    // Memory calculation variables
    double mem_real = 0.0;
    double mem_base = 0.0;
    double mem_per_blk = 0.0;
    double mem_per_repl = 0.0;
    double mem_per_repl_blk = 0.0;
    double mem_min = 0.0;
    double factor = 0.0;

    mem_real = mp2_memory;
    
    // Calculate memory components (based on Fortran logic)
    
    // BIB_C_copy: MAX(MAX(homo*maxsize(gd_array)), dimen_RI) * maxsize(gd_B_virtual)
    double max_homo_gd = 0.0;
    for (int i = 0; i < homo_size; i++) {
        double temp = (double)homo[i] * maxsize_gd_array;
        if (temp > max_homo_gd) {
            max_homo_gd = temp;
        }
    }
    double max_compare = (max_homo_gd > (double)dimen_RI) ? max_homo_gd : (double)dimen_RI;
    mem_per_repl += max_compare * maxsize_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C: SUM(homo*maxsize(gd_B_virtual)) * maxsize(gd_array)
    double sum_homo_gd_B = 0.0;
    for (int i = 0; i < homo_size; i++) {
        sum_homo_gd_B += (double)homo[i] * maxsize_gd_B_virtual;
    }
    mem_per_repl += sum_homo_gd_B * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C_rec: maxsize(gd_B_virtual) * maxsize(gd_array)
    mem_per_repl_blk += (double)maxval_gd_B_virtual * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // local_i_aL+local_j_aL: 2 * maxsize(gd_B_virtual) * dimen_RI
    mem_per_blk += 2.0 * maxval_gd_B_virtual * (double)dimen_RI * 8.0 / (1024.0 * 1024.0);
    
    // local_ab: MAX(virtual*maxsize(gd_B_virtual))
    double max_virtual_gd_B = 0.0;
    for (int i = 0; i < virtual_size; i++) {
        double temp = (double)virtual_arr[i] * maxsize_gd_B_virtual;
        if (temp > max_virtual_gd_B) {
            max_virtual_gd_B = temp;
        }
    }
    mem_base += max_virtual_gd_B * 8.0 / (1024.0 * 1024.0);
    
    // external_ab/external_i_aL: MAX(dimen_RI, max_virtual) * maxsize(gd_B_virtual)
    int max_dim = (dimen_RI > maxval_virtual) ? dimen_RI : maxval_virtual;
    mem_base += (double)max_dim * maxval_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    
    if (calc_forces) {
        // Gamma_P_ia: SUM(homo*maxsize(gd_array)*maxsize(gd_B_virtual))
        double sum_homo_gd = 0.0;
        for (int i = 0; i < homo_size; i++) {
            sum_homo_gd += (double)homo[i] * maxsize_gd_array * maxsize_gd_B_virtual;
        }
        mem_per_repl += sum_homo_gd * 8.0 / (1024.0 * 1024.0);
        
        // Y_i_aP+Y_j_aP: 2 * maxsize(gd_B_virtual) * dimen_RI
        mem_per_blk += 2.0 * maxval_gd_B_virtual * dimen_RI * 8.0 / (1024.0 * 1024.0);
        
        // local_ba/t_ab: maxsize(gd_B_virtual) * MAX(dimen_RI, max_virtual)
        mem_base += (double)maxval_gd_B_virtual * max_dim * 8.0 / (1024.0 * 1024.0);
        
        // P_ij: SUM(homo*homo)
        double sum_homo_sq = 0.0;
        for (int i = 0; i < homo_size; i++) {
            sum_homo_sq += (double)homo[i] * homo[i];
        }
        mem_base += sum_homo_sq * 8.0 / (1024.0 * 1024.0);
        
        // P_ab: SUM(virtual*maxsize(gd_B_virtual))
        double sum_virtual_gd_B = 0.0;
        for (int i = 0; i < virtual_size; i++) {
            sum_virtual_gd_B += (double)virtual_arr[i] * maxsize_gd_B_virtual;
        }
        mem_base += sum_virtual_gd_B * 8.0 / (1024.0 * 1024.0);
        
        // send_ab/send_i_aL: MAX(dimen_RI, max_virtual) * maxsize(gd_B_virtual)
        mem_base += (double)max_dim * maxval_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    }
    
    // Initial block size guess
    // block_size = MAX(1, MIN(FLOOR(SQRT(MINVAL(homo))), FLOOR(MINVAL(homo)/SQRT(2.0*ngroup))))
    int min_homo = max_homo;
    for (int i = 0; i < homo_size; i++) {
        if (homo[i] < min_homo) min_homo = homo[i];
    }
    block_size = (int)sqrt((double)min_homo);
    int temp = (int)(min_homo / sqrt(2.0 * ngroup));
    block_size = (block_size < temp) ? block_size : temp;
    block_size = (block_size < 1) ? 1 : block_size;
    
    // User-provided block size would be set here
    // if (user_block_size > 0) block_size = user_block_size;
    
    mem_min = mem_base + mem_per_repl + (mem_per_blk + mem_per_repl_blk) * block_size;
    
    // Print memory info if unit_nr > 0
    if (unit_nr > 0) {
        // Using printf for now - would use CP2K logging in production
        printf("RI_INFO| Minimum available memory per MPI process: %9.2f MiB\n", mem_real);
        printf("RI_INFO| Minimum required memory per MPI process: %9.2f MiB\n", mem_min);
    }
    
    // Calculate factor for communication model
    // factor = SUM(homovirtual) - SUM((MAX(homo)/block_size + block_size - 2)*homovirtual)/ngroup
    double factor_homo = 0.0;
    for (int i = 0; i < homo_size; i++) {
        factor_homo += (double)homo[i] * virtual_arr[i];
    }
    
    double sum_factor = 0.0;
    for (int i = 0; i < homo_size; i++) {
        double temp = ((double)max_homo / block_size + block_size - 2.0);
        sum_factor += temp * homo[i] * virtual_arr[i];
    }
    factor = factor_homo - sum_factor / ngroup;
    
    if (nspins == 2) {
        factor = factor - 2.0 * product_homo / block_size / ngroup * sum_homo_virtual;
    }
    
    // Determine integration group size
    integ_group_size = ngroup;  // Default
    
    if (factor <= 0.0) {
        // Calculate max replication group size
        // max_repl_group_size = FLOOR((mem_real - mem_base - mem_per_blk*block_size) /
        //                             (mem_per_repl + mem_per_repl_blk*block_size))
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
    
    // If calc_group_size is false, use user-provided group size
    if (!calc_group_size_local) {
        integ_group_size = ngroup / number_integration_groups;
    }
    
    // Print result
    if (unit_nr > 0) {
        printf("RI_INFO| Group size for integral replication: %6d\n", integ_group_size);
        fflush(stdout);
    }
    
    // Compute num_integ_group
    num_integ_group = ngroup / integ_group_size;
    
    // Return values
    *integ_group_size_out = integ_group_size;
    *ngroup_out = ngroup;
    *num_integ_group_out = num_integ_group;
}

/**
 * now create a group that contains all the proc that have
 * the same virtual starting point in the integ group
 * 
 * This subroutine creates the parallel infrastructure for RI-MP2:
 * 1. Creates exchange communicator (comm_exchange)
 * 2. Creates replication communicator (comm_rep)
 * 3. Builds ranges_info_array for data redistribution
 * 4. Updates sizes_array for new group distribution
 * ===========PARAMETERS=========
 * // OUTPUTS (communicators as Fortran version)
 * int* comm_exchange_out,
 * int* comm_rep_out,
 * 
 * // INPUT/OUTPUT arrays
 * int* sizes_array,
 * int sizes_array_size,
 * 
 * // OUTPUT arrays
 * int* ranges_info_array,
 * int ranges_info_array_dim1,  // Should be 4
 * int ranges_info_array_dim2,  // comm_rep%num_pe
 * int ranges_info_array_dim3,  // comm_exchange%num_pe
 * int* integ_group_pos2color_sub,
 * int integ_group_pos2color_sub_size,
 * int* sizes_array_orig,
 * int sizes_array_orig_size,
 * 
 * // OUTPUT scalars
 * int* my_group_L_size_out,
 * int* my_group_L_size_orig_out,
 * int* my_new_group_L_size_out,
 * 
 * // INPUT scalars
 * int my_group_L_start,
 * int my_group_L_end,
 * 
 * // MPI communicators (Fortran version)
 * int para_env_comm,
 * int para_env_sub_comm,
 * 
 * // Other inputs
 * int color_sub,
 * int integ_group_size,
 * int num_integ_group,
 * bool calc_forces,
 * int my_group_L_size,
 * int my_group_L_size_orig
 */
void c_mp2_ri_create_group(
    int* comm_exchange_out, int* comm_rep_out, int* sizes_array,
    int sizes_array_size, int* ranges_info_array, int ranges_info_array_dim1,
    int ranges_info_array_dim2, int ranges_info_array_dim3, int* integ_group_pos2color_sub,
    int integ_group_pos2color_sub_size, int* sizes_array_orig, int sizes_array_orig_size,
    int* my_group_L_size_out, int* my_group_L_size_orig_out, int* my_new_group_L_size_out,
    int my_group_L_start, int my_group_L_end, int para_env_comm, int para_env_sub_comm,
    int color_sub, int integ_group_size, int num_integ_group, bool calc_forces,
    int my_group_L_size, int my_group_L_size_orig) {
    // Convert Fortran MPI communicators to C MPI communicators
    cp_mpi_comm_t comm_para_env_c_comm = cp_mpi_comm_f2c(para_env_comm);
    cp_mpi_comm_t comm_para_env_sub_c_comm = cp_mpi_comm_f2c(para_env_sub_comm);

    comm_exchange_out = comm_para_env_c_comm;

    // Get rank and size of the sub-communicator
    int para_env_rank = cp_mpi_comm_rank(comm_para_env_c_comm);
    int para_env_size = cp_mpi_comm_size(comm_para_env_c_comm);
    int para_env_sub_rank = cp_mpi_comm_rank(comm_para_env_sub_c_comm);
    int para_env_sub_size = cp_mpi_comm_size(comm_para_env_sub_c_comm);

    // Calculate the ngroup
    int ngroup = para_env_size / integ_group_size;

    // Local variables
    cp_mpi_comm_t comm_exchange_c;
    cp_mpi_comm_t comm_rep_c;

    int comm_exchange_rank = 0;
    int comm_exchange_size = 0;
    int comm_rep_rank = 0;
    int comm_rep_size = 0;

    int my_new_group_L_size = my_group_L_size;
    int my_new_group_L_size_orig = my_group_L_size_orig;


    /**
     * ====================================================
     * STEP 1: Create exchange communicator (comm_exchange)
     * ====================================================
     * From sub_sub_color = para_env_sub%mepos * num_inte_group + color_sub/integ_group_size
     */
    int sub_sub_color_exchange = para_env_sub_rank * num_integ_group + color_sub / integ_group_size;

    // Split the world communicator
    // Use the rank as ket for consistent ordering
    int exchange_key = para_env_rank;
    cp_mpi_comm_t comm_exchange_old;

    // Create the exchange communicator
    MPI_Comm_split(comm_para_env_c_comm, sub_sub_color_exchange, exchange_key, &comm_exchange_c);

    // Get info about exchange communicator
    comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    comm_exchange_size = cp_mpi_comm_size(comm_exchange_c);

    offload_timeset("mp2_ri_create_group\0");
    /**
     * ====================================================
     * STEP 2: Create replication communicator (comm_rep)
     * ====================================================
     * From sub_sub_color = para_env_sub%mepos*comm_exchange%num_pe + comm_exchange%mepos
     */
    int sub_sub_color = para_env_sub_rank * comm_exchange_size + comm_exchange_rank;

    // Create replication communicator
    MPI_Comm_split(comm_para_env_c_comm, sub_sub_color, exchange_key, &comm_rep_c);

    // Assign replication communicator var
    comm_rep_out = comm_rep_c;

    // Get info about replication communicator
    comm_rep_rank = cp_mpi_comm_rank(comm_rep_c);
    comm_rep_size = cp_mpi_comm_size(comm_rep_c);

    /**
     * ====================================================
     * STEP 3: Build replication information
     * ====================================================
     */

    // Allocate arrays for gathering replication infor
    int* rep_sizes_array = (int*)malloc(comm_rep_size * sizeof(int));
    int* rep_starts_array = (int*)malloc(comm_rep_size * sizeof(int));
    int* rep_ends_array = (int*)malloc(comm_rep_size * sizeof(int));

     /**
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(&my_group_L_size, 1, rep_sizes_array, 1, comm_rep_c);
    cp_mpi_allgather_int(&my_group_L_start, 1, rep_starts_array, 1, comm_rep_c);
    cp_mpi_allgather_int(&my_group_L_end, 1, rep_ends_array, 1, comm_rep_c);

    /**
     * ====================================================
     * STEP 4: Calculate new group size and build my_info
     * ====================================================
     */

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

        /**
         * I suppose this is the ring communication pattern,
         * where each process communicates with its neighbor in the replication group.
         * The modulo_frotran function is try to ensure that the rank wraps around correctly (always +).
         * The proc_receive is the rank of the process from which this process will receive data.
         * The my_info array is being filled with the start, end, local_start, and
         * 
         * Process 0 collects from:  0, 3, 2, 1 (counter-clockwise)
         * Process 1 collects from:  1, 0, 3, 2                    
         * Process 2 collects from:  2, 1, 0, 3                    
         * Process 3 collects from:  3, 2, 1, 0
         * Each process gets data from the others
         * 
         * I think for example in the process 2, will be something like:
         * proc_shift=1:  Process 2 -> collects from Process 1
         * proc_shift=2:  Process 2 -> collects from Process 0
         * proc_shift=3:  Process 2 -> collects from Process 3
         */
        int proc_receive = modulo_frotran(comm_rep_rank - proc_shift, comm_rep_size);
        
        // Update new group size
        my_new_group_L_size += rep_sizes_array[proc_receive];

        my_info[0 * comm_rep_size + proc_shift] = rep_starts_array[proc_receive]; // start
        my_info[1 * comm_rep_size + proc_shift] = rep_ends_array[proc_receive]; // end
        my_info[2 * comm_rep_size + proc_shift] = my_info[3 * comm_rep_size + proc_shift - 1] + 1; // local_start
        my_info[3 * comm_rep_size + proc_shift] = my_new_group_L_size; // local_end
    }

    /**
     * ====================================================
     * STEP 5: Build ranges_info_array
     * ====================================================
     */

    // Allocate ranges_info_array as a flat array (4 x comm_rep_size x comm_exchange_size)
    int* new_sizes_array = (int*)malloc(comm_exchange_size * sizeof(int));
    int* my_info_temp = my_info; // Temporary pointer to my_info for easier indexing

    /**
      * Gather my_new_group_L_size from all processes in the exchange communicator
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(&my_new_group_L_size, 1, new_sizes_array, 1, comm_exchange_c);

    // Gather my_info from all processes in the exchange communicator
    int my_info_size = 4 * comm_rep_size;
    int* all_my_info =(int*)malloc(my_info_size * comm_exchange_size * sizeof(int));

    /**
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(my_info, my_info_size, ranges_info_array, my_info_size, comm_exchange_c);

    free(rep_sizes_array);
    free(rep_starts_array);
    free(rep_ends_array);

    /**
      * Build the integ_group_po2color_sub
      * 
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(&color_sub, 1, integ_group_pos2color_sub, 1, comm_exchange_c);

    /**
     * ====================================================
     * STEP 6: Update sizes_array
     * ====================================================
     */

    /**
     * cp_mpi_allgather_int(
     *      const int *sendbuf,
     *      const int sendcount,
     *      int *recvbuf,
     *      const int recvcount,
     *      const cp_mpi_comm_t comm
     * );
     */
    cp_mpi_allgather_int(&my_new_group_L_size, 1, new_sizes_array, 1, comm_exchange_c);
    
    // ALLOCATE (sizes_array(0:integ_group_size - 1))
    sizes_array = (int*)malloc(integ_group_size * sizeof(int));

    // Copy data from new_sizes_array to size_array
    for (int i = 0; i < integ_group_size && i < comm_exchange_size; i++) {
        sizes_array[i] = new_sizes_array[i];
    }

    // DEALLOCATE (new_sizes_array)
    free(new_sizes_array);

    // time stop
    offload_timestop();
}


/**
 * // Input/Output: BIb_C array (3D: L, virtual, occupied)
 * SIZE(Bib_C, 1) = L-size
 * SIZE(Bib_C, 2) = virtual orbitals
 * SIZE(Bib_C, 3) = occupied orbitals
 * 
 * double** BIb_C,          // Pointer to array pointer (can reallocate)
 * int* BIb_C_L_size,       // Current L-size of BIb_C
 * int* BIb_C_virtual,      // Virtual orbitals dimension
 * int* BIb_C_occupied,     // Occupied orbitals dimension
 * 
 * // MPI communicators (Fortran handles)
 * int comm_exchange,
 * int comm_rep,
 * 
 * // Dimensions
 * int homo,                // Number of occupied orbitals
 * const int* sizes_array,  // Distribution of L-indices
 * int sizes_array_size,    // Size of sizes_array
 * int my_B_size,           // Number of virtual orbitals for this process
 * int my_group_L_size,     // New L-size after replication
 * 
 * // Range mapping information
 * // Follow the last subroutine I think:
 * const int* ranges_info_array,  // 4 x rep_size x exchange_size
 * int ranges_info_dim1,          // Should be 4
 * int ranges_info_dim2,          // comm_rep%num_pe
 * int ranges_info_dim3           // comm_exchange%num_pe
 */
void c_replicate_iaK_2intgroup(
    double** BIb_C, int* BIb_C_L_size, int* BIb_C_virtual,
    int* BIb_C_occupied, int comm_exchange, int comm_rep,
    int homo, const int* sizes_array, int sizes_array_size,
    int my_B_size, int my_group_L_size, const int* ranges_info_array,
    int ranges_info_dim1, int ranges_info_dim2, int ranges_info_dim3
) {
    cp_mpi_comm_t comm_exchange_c = cp_mpi_comm_f2c(comm_exchange);
    cp_mpi_comm_t comm_rep_c = cp_mpi_comm_f2c(comm_rep);
    
    int comm_rep_size = cp_mpi_comm_size(comm_rep_c);
    int comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    int comm_rep_rank = cp_mpi_comm_rank(comm_rep_c);

    offload_timeset("replicate_iaK_2intgroup\0");

    // Replication scheme using mpi_allgather
    // get the max L size
    int max_L_size = 0;
    for (int i = 0; i < sizes_array_size; i++) {
        if (sizes_array[i] > max_L_size) {
            max_L_size = sizes_array[i];
        }
    }

    // STEP 1: Create local copy (BIb_C_copy)

    // Get current BIb_C dimensions
    int current_L_size = *BIb_C_L_size;
    int virtual_size = *BIb_C_virtual;
    int occupied_size = *BIb_C_occupied;

    // Allocate copy buffer: [L][virtual][occupied]
    size_t copy_size = (size_t)max_L_size * my_B_size * homo;
    // fill with 0 calloc instead of malloc
    double* BIb_C_copy = (double*)calloc(copy_size, sizeof(double));

    // copy data from old BIb_C to copy buffer
    for (int i = 0; i < homo; i++) {
        for (int j = 0; j < my_B_size; j++) {
            //Source: old BIb_C (L,j,i) - colum-major
            //Source index: (i*my_B_size + j) * current_L_size
            size_t src_idx = ((size_t)i * my_B_size + j) * current_L_size;

            //Destination: new BIb_C (i,j,L) - row-major
            //Destination index: (i*my_B_size + j) * max_L_size
            size_t dst_idx = ((size_t)i * my_B_size + j) * max_L_size;

            // Copy current_L_size doubles
            // memcpy(void *dest, const void *src, size_t count);
            memcpy(&BIb_C_copy[dst_idx], &(*BIb_C)[src_idx], current_L_size * sizeof(double));
        }
    }
     // Free original BIb_C
    free(*BIb_C);
    
    //  Gather all data using cp_mpi_allgather_double
    
    // Allocate gather buffer: [comm_rep_size][max_L_size][my_B_size][homo]
    size_t gather_size = (size_t)comm_rep_size * max_L_size * my_B_size * homo;
    double* BIb_C_gather = (double*)calloc(gather_size, sizeof(double));
    
    int send_count = (int)(max_L_size * my_B_size * homo);

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

        // ranges_info_array(dim, proc_shift, exchange_rank)
        // dim=3: local_start, dim=4: local_end
        int start_point = ranges_info_array[
            3 * comm_rep_size * comm_exchange_rank +
            proc_shift * comm_exchange_rank + 3
        ];
        int end_point = ranges_info_array[
            4 * comm_rep_size * comm_exchange_rank +
            proc_shift * comm_exchange_rank + 4
        ];
        int L_size = end_point - start_point + 1;
        
        // Calculate offsets
        // Each process's data in gather buffer
        size_t gather_offset = (size_t)proc_receive * max_L_size * my_B_size * homo;
        
        // Output starts at start_point-1 (0-based) in L dimension
        size_t output_offset = ((size_t)0 * my_B_size + 0) * my_group_L_size + (start_point - 1);
        
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
    
    *BIb_C = BIb_C_new; //INTOUT
    *BIb_C_L_size = my_group_L_size;
    *BIb_C_virtual = my_B_size;
    *BIb_C_occupied = homo;

    // stop the timer
    offload_timestop();
}

/**
 * 
 * ===========PARAMETERS=========
 * // Output arrays
 * double** local_ab,        // local_ab(virtual(ispin), my_B_size(jspin))
 * double** t_ab,            // t_ab(virtual(ispin), my_B_size(jspin))
 * double** local_ba,        // local_ba(virtual(jspin), my_B_size(ispin))
 * 
 * // Input arrays
 * const int* homo,          // homo(2) - occupied orbitals
 * const int* virtual,       // virtual(2) - virtual orbitals
 * const int* my_B_size,     // my_B_size(2) - virtual per process
 * int my_group_L_size,      // L-size for Gamma_P_ia
 * 
 * // Control
 * bool calc_forces,         // Whether forces are being computed
 * int ispin,                // Spin index for i
 * int jspin,                // Spin index for j
 * 
 * // mp2_env%ri_grad arrays (allocated here if not already)
 * double** P_ij,            // mp2_env%ri_grad%P_ij(jspin)%array
 * double** P_ab,            // mp2_env%ri_grad%P_ab(jspin)%array
 * double** Gamma_P_ia,      // mp2_env%ri_grad%Gamma_P_ia(jspin)%array
 * 
 * // Allocation status flags (tracks if arrays already exist)
 * bool* P_ij_allocated,
 * bool* P_ab_allocated,
 * bool* Gamma_P_ia_allocated
 */
void c_mp2_ri_allocate_no_blk(
    double** local_ab, double** t_ab, double** local_ba,
    const int* homo, const int* virtual, const int* my_B_size,
    int my_group_L_size, bool cal_forces, int ispin, int jspin,
    double** P_ij, double** P_ab, double** Gamma_P_ia,
    bool* P_ij_allocated, bool* P_ab_allocated,bool* Gamma_P_ij_allocated 
) {
    //Start timer
    offload_timeset("mp2_ri_allocate_no_blk\0");

    // From fortran index (1) to C (0)
    int i_c = ispin - 1;
    int j_c = jspin - 1;

    // ALLOCATE(local_ab(virtual(ispin), my_B_size(jspin)))
    // local_ab = 0.0_dp
    *local_ab = (double*)calloc((size_t)virtual[i_c] * my_B_size[j_c], sizeof(double));

    //stopo timer
    offload_timestop();
}

/**
 * SUBROUTINE mp2_ri_get_block_size(mp2_env, para_env, para_env_sub, gd_array, gd_B_virtual, &
 *                                  homo, virtual, dimen_RI, unit_nr, &
 *                                  block_size, ngroup, num_integ_group, &
 *                                  my_open_shell_ss, calc_forces, buffer_1D)
 * 
 * -----------------------------------------------------------------------------------------
 * |            FORTRAN-SIDE               |                   C-SIDE                      |
 * ----------------------------------------------------------------------------------------|
 * |   (VAR TYPE) | (INTENT)               | (VAR TYPE)      |(INTENT) | (VAR NAME)        |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) mp2_env           |                 |         |                   |
 * |    ====>  mp2_env%ri_mp2%block_size   | (int)           |         | (user_block_size) |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) para_env          | (cp_mpi_comm_t) |         | (para_env)        |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) para_env_sub      | (cp_mpi_comm_t) |         | (para_env_sub)    |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) gd_array          | (int*)          | (const) |                   |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE(:))  | (IN) gd_B_virtual      | (int*)          | (const) |                   |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT(:))   | (IN) homo              | (int*)          | (const) | (homo)            |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT(:))   | (IN) virtual           | (int*)          | (const) | (virtual_arr)     |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (IN) dimen_RI          | (int)           |         | (dimen_RI)        |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (IN) unit_nr           | (int)           |         | (unit_nr)         |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (OUT) block_size       | (int*)          | (OUT)   | (block_size)      |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (OUT) ngroup           | (int*)          | (OUT)   | (ngroup_out)      |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (IN) num_integ_group   | (int)           |         | num_integ_group   |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (LOGICAL)  | (IN) my_open_shell_ss  | (bool)          |         | my_open_shell_ss  |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (LOGICAL)  | (IN) calc_forces       | (bool)          |         | calc_forces       |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (REAL(:))  | (OUT) buffer_1D        | (double*)       | (OUT)   | (buffer_1D)       |
 * -----------------------------------------------------------------------------------------
 * 
 * // OUTPUTS
 * int* block_size,
 * int* ngroup_out,
 * double** buffer_1D,
 * 
 * // FROM mp2_env
 * int user_block_size,
 * 
 * // COMMUNICATORS
 * cp_mpi_comm_t para_env_comm,
 * cp_mpi_comm_t para_env_sub_comm,
 * 
 * // FROM gd_array
 * const int* gd_array_sizes,
 * int gd_array_sizes_size,
 * int maxsize_gd_array,
 * 
 * // FROM gd_B_virtual
 * const int* gd_B_virtual_sizes,
 * int gd_B_virtual_sizes_size,
 * int maxsize_gd_B_virtual,
 * int maxval_gd_B_virtual,
 * 
 * // FROM homo
 * const int* homo,
 * int homo_size,
 * 
 * // FROM virtual
 * const int* virtual_arr,
 * int virtual_size,
 * int maxval_virtual,
 * 
 * int dimen_RI,
 * int unit_nr,
 * int num_integ_group,
 * bool my_open_shell_ss,
 * bool calc_forces
 */
void c_mp2_ri_get_block_size(
    int* block_size, int* ngroup_out, double** buffer_1D,
    double mp2_memory, int user_block_size, cp_mpi_comm_t para_env_comm,
    cp_mpi_comm_t para_env_sub_comm, const int* gd_array_sizes, int gd_array_sizes_size,
    int maxsize_gd_array, const int* gd_B_virtual_sizes, int gd_B_virtual_sizes_size,
    int maxsize_gd_B_virtual, int maxval_gd_B_virtual, const int* homo,
    int homo_size, const int* virtual_arr, int virtual_size,
    int maxval_virtual, int dimen_RI, int unit_nr,
    int num_integ_group, bool my_open_shell_ss, bool calc_forces
) {
    //Start timer
    offload_timeset("mp2_ri_get_block_size\0");
    // ========================================================================
    // STEP 1: Calculate ngroup
    // ========================================================================
    int para_env_size = cp_mpi_comm_size(para_env_comm);
    int para_env_sub_size = cp_mpi_comm_size(para_env_sub_comm);
    int ngroup = para_env_size / para_env_sub_size;
    *ngroup_out = ngroup;

    // ========================================================================
    // STEP 2: Get available memory
    // ========================================================================
    int64_t mem_bytes = 0;
    // In fortran-side is call m_memory()
    double mem_real = (double)((mem_bytes + 1024*1024 - 1) / (1024*1024));
    cp_mpi_max_double(&mem_real, 1, para_env_comm);
    
    // ========================================================================
    // STEP 3: Calculate memory components
    // ========================================================================
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

    // ========================================================================
    // STEP 4: Determine block size
    // ========================================================================
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
            if (homo_size == 1) {
                if (!my_open_shell_ss) {
                    num_IJ_blocks = homo[0] / best_block_size;
                    num_IJ_blocks = (num_IJ_blocks * num_IJ_blocks - num_IJ_blocks) / 2;
                } else {
                    num_IJ_blocks = (homo[0] - 1) / best_block_size;
                    num_IJ_blocks = (num_IJ_blocks * num_IJ_blocks - num_IJ_blocks) / 2;
                }
            } else {
                num_IJ_blocks = 1;
                for (int i = 0; i < homo_size; i++) {
                    num_IJ_blocks *= (homo[i] / best_block_size);
                }
            }
            
            if ((num_IJ_blocks >= ngroup && num_IJ_blocks > 0) || best_block_size == 1) {
                break;
            } else {
                best_block_size--;
            }
        }
        
        if (homo_size == 1) {
            if (my_open_shell_ss) {
                int sqrt_val = (int)sqrt((double)(homo[0] - 1));
                best_block_size = (sqrt_val < best_block_size) ? sqrt_val : best_block_size;
            } else {
                int sqrt_val = (int)sqrt((double)homo[0]);
                best_block_size = (sqrt_val < best_block_size) ? sqrt_val : best_block_size;
            }
        }
    }
    
    *block_size = (best_block_size < 1) ? 1 : best_block_size;
    
    // ========================================================================
    // STEP 5: Print info if unit_nr > 0
    // ========================================================================
    if (unit_nr > 0) {
        printf("RI_INFO| Block size: %6d\n", *block_size);
        fflush(stdout);
    }

    // ========================================================================
    // STEP 6: Allocate buffer
    // ========================================================================
    int64_t buffer_size = 0;
    int64_t size1 = (int64_t)maxsize_gd_array * (*block_size);
    int64_t size2 = (int64_t)max_dim;
    int64_t max_size = (size1 > size2) ? size1 : size2;
    buffer_size = max_size * maxval_gd_B_virtual;
    if (calc_forces) buffer_size *= 2;
    
    *buffer_1D = (double*)malloc((size_t)buffer_size * sizeof(double));
    if (*buffer_1D == NULL) {
        fprintf(stderr, "Error: Failed to allocate buffer_1D of size %ld\n", buffer_size);
        exit(1);
    }

    // Stop timer
    offload_timestop();
}


/**
 * * 
 * 
 * SUBROUTINE mp2_ri_communication(my_alpha_beta_case, total_ij_pairs, homo, homo_beta, &
                                   block_size, ngroup, ij_map, color_sub, my_ij_pairs, my_open_shell_SS, unit_nr)
      LOGICAL, INTENT(IN)                                :: my_alpha_beta_case
      INTEGER, INTENT(OUT)                               :: total_ij_pairs
      INTEGER, INTENT(IN)                                :: homo, homo_beta, block_size, ngroup
      INTEGER, ALLOCATABLE, DIMENSION(:, :), INTENT(OUT) :: ij_map
      INTEGER, INTENT(IN)                                :: color_sub
      INTEGER, INTENT(OUT)                               :: my_ij_pairs
      LOGICAL, INTENT(IN)                                :: my_open_shell_SS
      INTEGER, INTENT(IN)                                :: unit_nr
 * ____________________________________________________________________________________________________________
 * |                  FORTRAN-SIDE                   ||                   C-SIDE                              |
 * |_________________________________________________||_______________________________________________________|
 * |   (VAR TYPE) |(INTENT) |       (VAR NAME)       ||    (VAR TYPE)      |(INTENT) |      (VAR NAME)        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (LOGICAL)  |   (IN)  |   my_alpha_beta_case   ||       bool         |         |   my_alpha_beta_case   |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |  (OUT)  |     total_ij_pairs     ||       int*         |         |      total_ij_pairs    |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |         homo           ||       int          |         |          homo          |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |       homo_beta        ||       int          |         |        homo_beta       |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |      block_size        ||       int          |         |       block_size       |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |        ngroup          ||       int          |         |         ngroup         |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |  (INTEGER )  |         |                        ||                    |         |                        |
 * |  (DIM(:,:))  |  (IN)   |         ij_map         ||       int**        |  malloc |       ij_map           |
 * | (ALLOCATABLE)|         |                        ||                    |         |                        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |       color_sub        ||       int          |         |       color_sub        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |  (OUT)  |     my_ij_pairs        ||       int*         |         |      my_ij_pairs       |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (LOGICAL)  |   (IN)  |   my_open_shell_SS     ||       int          |         |    my_open_shell_SS    |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |       unit_nr          ||       int          |         |        unit_nr         |
 * |______________|_________|________________________||____________________|_________|________________________|
 * 
 * // INPUTS
 * bool my_alpha_beta_case   // Alpha-beta case flag
 * int homo                  // Number of occupied orbitals (spin 1)
 * int homo_beta             // Number of occupied orbitals (spin 2)
 * int block_size            // Block size for ij pairs
 * int ngruop                // Number of groups
 * int color_sub             // Process color in subgroup
 * bool my_open_shell_SS     // Open shell same-spin flag
 * int unit_nr               // Output unit number
 * 
 * //OUTPUT
 * int* total_ij_pairs       // Total number of ij pairs
 * int** ij_map              // Map fo ij pairs (3 x total)
 * int* my_ij_pairs          // Number of pairs for this process
 */
void c_mp2_ri_communication(
    bool my_alpha_beta_case, int homo, int homo_beta,
    int block_size, int ngroup, int color_sub,
    bool my_open_shell_SS, int unit_nr, int* total_ij_pairs,
    int** ij_map, int* my_ij_pairs
){
    // start timer
    offload_timeset("mp2_ri_communication\0");

    *total_ij_pairs = homo * (1 + homo) / 2;
    int num_IJ_blocks = homo / block_size - 1;

    int first_I_block = 1;
    int last_i_block = block_size * (num_IJ_blocks - 1);
    int first_J_block = block_size + 1;
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
    int total_ij_pairs_blocks = assigned_blocks + (total_ij_pairs - assigned_blocks * (block_size * block_size));

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
    *ij_map = (int*)calloc(3 * total_ij_pairs_blocks, sizeof(int));

    int ij_counter = 0;
    *my_ij_pairs = 0;

    for (int iiB = first_I_block; iiB < last_i_block; iiB += block_size) {
        for (int jjB = iiB + block_size; jjB < last_J_block; jjB += block_size) {
            // exit
            if (ij_counter + 1 > assigned_blocks) {break;}
            ij_counter++;

            // ij_marker(iiB:iiB + block_size - 1, jjB:jjB + block_size - 1) = .FALSE.
            // i = iiB - 1 (index 0 in C)
            for (int i = iiB; i < iiB + block_size - 1; i++) {
                // j = jjB - 1 (index 0 in C)
                for (int j = jjB; j < jjB + block_size - 1; j++) {
                    ij_marker[i * homo + j] = false;
                }

                (*ij_map)[0 * total_ij_pairs_blocks + (ij_counter - 1)] = iiB;
                (*ij_map)[1 * total_ij_pairs_blocks + (ij_counter - 1)] = jjB;
                (*ij_map)[2 * total_ij_pairs_blocks + (ij_counter - 1)] = block_size;

                if ((ij_block_counter % ngroup) == color_sub) {
                    (*my_ij_pairs)++;
                }
            }

            for (int iiB = 1; iiB < homo; iiB++) {
                for (int jjB = iiB; jjB < homo; jjB++) {
                    // to access: arr[i * cols + j]
                    // 0-based in C-stlr
                    if (ij_marker[(iiB - 1) * homo + (jjB - 1)]) {
                        ij_counter++;
                        (*ij_map)[0 * total_ij_pairs_blocks + (ij_counter -1)] = iiB;
                        (*ij_map)[1 * total_ij_pairs_blocks + (ij_counter -1)] = jjB;
                        (*ij_map)[2 * total_ij_pairs_blocks + (ij_counter -1)] = 1;

                        if ((ij_counter % ngroup) == color_sub) {
                            (*my_ij_pairs)++;
                        }
                    }
                }
            }
            free(ij_marker);
        }
    }

    if (unit_nr > 0) {
        if (block_size == 1) {
            printf("RI_INFO| Percentage of ij pairs communicated with block size 1: 100.0\n");
        } else {
            double percentage = 100.0 * (double)((*total_ij_pairs - assigned_blocks * (block_size * block_size))) /  (double)(*total_ij_pairs);
            printf("RI_INFO| Percentage of ij pairs communicated with block size 1: %.1f\n", percentage);
        }
    }

    // Stop timer
    offload_timestop();
}

/**
 * SUBROUTINE mp2_ri_allocate_blk(dimen_RI, my_B_size, block_size, &
                                  local_i_aL, local_j_aL, calc_forces, &
                                  Y_i_aP, Y_j_aP, ispin, jspin)
      INTEGER, INTENT(IN)                                :: dimen_RI, my_B_size(2), block_size
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :), &
         INTENT(OUT)                                     :: local_i_aL, local_j_aL
      LOGICAL, INTENT(IN)                                :: calc_forces
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :), &
         INTENT(OUT)                                     :: Y_i_aP, Y_j_aP
      INTEGER, INTENT(IN)                                :: ispin, jspin

      CHARACTER(LEN=*), PARAMETER :: routineN = 'mp2_ri_allocate_blk'
 * ____________________________________________________________________________________________________________
 * |                  FORTRAN-SIDE                   ||                   C-SIDE                              |
 * |_________________________________________________||_______________________________________________________|
 * |   (VAR TYPE) |(INTENT) |       (VAR NAME)       ||    (VAR TYPE)      |(INTENT) |      (VAR NAME)        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  | (IN)    |      dimen_RI          ||      int           |         |      dimen_RI          |
 * |______________|_________|________________________||____________________|_________|________________________|
 * | (INTEGER(:)) | (IN)    |      my_B_size         ||      int*          | (const) |      my_B_size         |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  | (IN)    |      block_size        ||       int          |         |      block_size        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (LOGICAL)  | (IN)    |      calc_forces       ||       bool         |         |     calc_forces        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |    (REAL)    |         |                        ||                    |         |                        |
 * | (DIM(:,:,:)) |  (OUT)  |       local_i_aL       ||     double**       |  malloc |       local_i_aL       |
 * | (ALLOCATABLE)|         |                        ||                    |         |                        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |    (REAL)    |         |                        ||                    |         |                        |
 * | (DIM(:,:,:)) |  (OUT)  |       local_j_aL       ||     double**       |  malloc |       local_j_aL       |
 * | (ALLOCATABLE)|         |                        ||                    |         |                        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |    (REAL)    |         |                        ||                    |         |                        |
 * | (DIM(:,:,:)) |  (OUT)  |        Y_i_aP          ||     double**       |  malloc |        Y_i_aP          |
 * | (ALLOCATABLE)|         |                        ||                    |         |                        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |    (REAL)    |         |                        ||                    |         |                        |
 * | (DIM(:,:,:)) |  (OUT)  |        Y_j_aP          ||     double**       |  malloc |        Y_j_aP          |
 * | (ALLOCATABLE)|         |                        ||                    |         |                        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  | (IN)    |         ispin          ||       int          |         |         ispin          |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  | (IN)    |         jspin          ||       int          |         |         jspin          |
 * |______________|_________|________________________||____________________|_________|________________________|
 * 
 * // INPUT
 * int dimen_RI,         // Numeber of RI basis functions
 * const int* my_B_size, // Virtual orbitals per spin (2)
 * int block_size,       // Occupied orbital block size
 * int ispin,            // Spin index for i
 * int jspin,            // Spin index for j
 * 
 * // OUTPUT (I need allocate it)
 * double** local_i_aL,  // (Could flatten like 2D)
 * double** local_j_aL,  // (Could flatten like 2D)
 * double** Y_i_aP,      // skip
 * double** Y_j_aP       // skip
 * 
 */
void c_mp2_ri_allocate_blk(
    int dimen_RI,
    const int* my_B_size,
    int block_size,
    int ispin,
    int jspin,
    double** local_i_aL,
    double** local_j_aL,
    double** Y_i_aP,
    double** Y_j_aP
){

    // Start timer
    offload_timeset("mp2_ri_allocate_blk\0");

    // Convert Fortran 1-based to C 0-based
    int i = ispin - 1;
    int j = jspin - 1;
    /**
     * Fortran: (dim1, dim2m2, dim3) (colum-major)
     * ALLOCATE (local_i_aL(dimen_RI, my_B_size(ispin), block_size))
     * local_i_aL = 0.0_dp
     * 
     * C: [dim3][dim2][dim1] (row-major) 
     * flattenf like [block_size][my_B_size(ispin)][dimen_RI]
     * 
     * double* arr = malloc(dim1 * dim2 * dim3 * sizeof(double));
     * to access: arr[i * cols + j]
     */
    *local_i_aL = (double*)calloc(block_size * my_B_size[i] * dimen_RI, sizeof(double));

     /**
      * ALLOCATE (local_j_aL(dimen_RI, my_B_size(jspin), block_size))
      * local_j_aL = 0.0_dp
      */
    *local_i_aL = (double*)calloc(block_size * my_B_size[j] * dimen_RI, sizeof(double));

    // Stop timer
    offload_timestop();
}


/**
 *  
 * __________________________________________________________________________________________________
 * |                  FORTRAN-SIDE                    ||               C-SIDE                        |
 * |__________________________________________________||_____________________________________________|
 * |   (VAR TYPE)  |(INTENT) |       (VAR NAME)       ||    (VAR TYPE)      |      (VAR NAME)        |
 * |_______________|_________|________________________||____________________|________________________|
 * |REAL DIM(:,:,:)|  (OUT)  |        local_i_aL      ||      double*       |      local_i_aL        |
 * |               |         |   local_i_aL_L_size    ||      int           |   local_i_aL_L_size    |
 * |               |         |   local_i_aL_virtual   ||      int           |   local_i_aL_virtual   |
 * |               |         |    local_i_aL_block    ||      int           |   local_i_aL_block     |
 * |_______________|_________|________________________||____________________|________________________|
 * |_______________|_________|________________________||____________________|________________________|
 * |   (INTEGER)   |         |                        ||                    |                        |
 * |DIMENSION(:, :)|  (IN)   |  ranges_info_array     ||     const int*     |   ranges_info_array    |
 * |               |         | ranges_info_rep_size   ||      int           | ranges_info_rep_size   |
 * |_______________|_________|________________________||____________________|________________________|
 * |   (INTEGER)   |  (IN)   |        BIb_C_rec       ||   const double*    |        BIb_C_rec       |
 * |               |         |   BIb_C_rec_L_size     ||      int           |   BIb_C_rec_L_size     |
 * |               |         |   BIb_C_rec_virtual    ||      int           |   BIb_C_rec_virtual    |
 * |               |         |    BIb_C_rec_block     ||      int           |   BIb_C_rec_block      |
 * |_______________|_________|________________________||____________________|________________________|
 * 
 * double* local_i_aL           // Flattened 3D array local_i_aL[L][virtual][block]
 * int local_i_aL_L_size        // L-size
 * int local_i_aL_virtual       // virtual dimension
 * int local_i_aL_block         // block size
 * 
 * const int* ranges_info_array // 4*size_rep (flattened)
 * int ranges_info_rep_size     //  number of replication groups
 * 
 * const double* BIb_C_rec      // Flattened 3D array BIb_C_rec[L][virtual][block]
 * int BIb_C_rec_L_size
 * int BIb_C_rec_virtual
 * int BIb_C_rec_block
 *
 */
void fill_local_i_aL(
    double* local_i_aL, int local_i_aL_L_size, int local_i_aL_virtual,
    int local_i_aL_block, const int* ranges_info_array, int ranges_info_rep_size,
    const double* BIb_C_rec, int BIb_C_rec_L_size, int BIb_C_rec_virtual,
    int BIb_C_rec_block
){
    // start timer
    offload_timeset("fill_local_i_aL\0");

    for(int irep = 0; irep < ranges_info_rep_size; irep++) {
        /**
         * Fortran: (dim1, dim2m2, dim3) (colum-major)
         * 
         * C: [dim3][dim2][dim1] (row-major) 
         * flattenf like [block_size][my_B_size(ispin)][dimen_RI]
         * 
         * to access: arr[i * ranges_info_rep_size + irep]
         */
        int Lstart_pos = ranges_info_array[0 * ranges_info_rep_size + irep];
        int Lend_pos = ranges_info_array[1 * ranges_info_rep_size + irep];
        int start_point = ranges_info_array[2 * ranges_info_rep_size + irep];
        int end_point = ranges_info_array[3 * ranges_info_rep_size + irep];

        // Number of L-index copy
        // start_point:end_point
        int L_size = end_point - start_point;

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
                size_t src_idx = ((size_t)i_block * BIb_C_rec_virtual + v_pos) * local_i_aL_L_size + (start_point - 1);

                // Destination
                // local_i_aL[(i_block * local_i_aL_virtual + v_pos) * BIb_C_rec_L_size + (Lstart_pos - 1)]
                size_t dest_idx = ((size_t)i_block * local_i_aL_virtual + v_pos) * BIb_C_rec_L_size + (Lstart_pos - 1);

                // void *memcpy(void *dest, const void *src, size_t count);
                memcpy(&local_i_aL[dest_idx], &BIb_C_rec[src_idx], L_size * sizeof(double));
            }
         }
    }

    // Stop timer
    offload_timestop();
}

/**
 * 
 * ______________________________________________________________________________________________
 * |                  FORTRAN-SIDE                ||                  C-SIDE                    |
 * |______________________________________________||____________________________________________|
 * |_________________|_________|__________________||__________________|_________________________|
 * |   (VAR TYPE)    |(INTENT) |   (VAR NAME)     ||    (VAR TYPE)    |      (VAR NAME)         |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   REAL(KIND=dp) | INOUT   | Emp2_Cou         ||    double*       | Emp2_Cou                |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   REAL(KIND=dp) | INOUT   | Emp2_EX          ||    double*       | Emp2_EX                 |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   REAL(KIND=dp) | INOUT   | Emp2_S           ||    double*       | Emp2_S                  |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   REAL(KIND=dp) | INOUT   | Emp2_T           ||    double*       | Emp2_T                  |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   TYPE(3D)      | INOUT   | BIb_C(:)         ||    double***     | BIb_C                   |
 * |   (ALLOCATABLE) |         |                  ||    int*          | BIb_C_dims[L,virt,occ]  |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   TYPE(mp2_type)| IN      | mp2_env          ||    double        | mp2_memory              |
 * |   (DERIVED)     |         |                  ||    int           | user_block_size         |
 * |                 |         |                  ||    bool          | print_dgemm_info        |
 * |                 |         |                  ||    double        | scale_S                 |
 * |                 |         |                  ||    double        | scale_T                 |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   TYPE(mp_para) | IN      | para_env         ||    cp_mpi_comm_t | para_env_comm           |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   TYPE(mp_para) | IN      | para_env_sub     ||    cp_mpi_comm_t | para_env_sub_comm       |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   INTEGER       | IN      | color_sub        ||    int           | color_sub               |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   TYPE(gd_d1)   | INOUT   | gd_array         ||    const int*    | gd_array_sizes          |
 * |   (DERIVED)     |         |                  ||    int           | gd_array_sizes_size     |
 * |                 |         |                  ||    int           | maxsize_gd_array        |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   TYPE(gd_d1)   | INOUT   | gd_B_virtual(:)  ||    const int*    | gd_B_virtual_sizes      |
 * |   (DERIVED)     |         |                  ||    int           | gd_B_virtual_sizes_size |
 * |                 |         |                  ||    int           | maxsize_gd_B_virtual    |
 * |                 |         |                  ||    int           | maxval_gd_B_virtual     |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   INTEGER(:)    | IN      | homo             ||    const int*    | homo                    |
 * |                 |         |                  ||    int           | homo_size               |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   INTEGER       | IN      | nmo              ||    int           | nmo                     |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   REAL(KIND=dp) | IN      | Eigenval(:, :)   ||    const double* | Eigenval                |
 * |   (2D array)    |         |                  ||    (flattened)   |                         |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   INTEGER       | IN      | dimen_RI         ||    int           | dimen_RI                |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   INTEGER       | IN      | unit_nr          ||    int           | unit_nr                 |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   LOGICAL       | IN      | calc_forces      ||    bool          | calc_forces             |
 * |_________________|_________|__________________||__________________|_________________________|
 * |   LOGICAL       | IN      | calc_ex          ||    bool          | calc_ex                 |
 * |_________________|_________|__________________||__________________|_________________________|
 * 
 * // Energy outputs
 * double* Emp2_Cou,    // Coulomb MP2 energy
 * double* Emp2_EX,     // Exchange MP2 energy
 * double* Emp2_S,      // Same-spin MP2 energy
 * double* Emp2_T,      // Triplet MP2 energy
 * 
 * // Integral data
 * double*** BIb_C,     // B matrix: (dimen_RI x virtual x occupied)
 * int* BIb_C_dims,     // [L_size, virtual, occupied] dimensions
 * 
 * // Environment
 * double mp2_memory,   // mp2_env%mp2_memory
 * int user_block_size, // mp2_env%ri_mp2%block_size
 * bool print_dgemm_info, // mp2_env%ri_mp2%print_dgemm_info
 * double scale_S,      // mp2_env%scale_S
 * double scale_T,      // mp2_env%scale_T
 * 
 * // MPI communicators
 * cp_mpi_comm_t para_env_comm,
 * cp_mpi_comm_t para_env_sub_comm,
 * 
 * // Process info
 * int color_sub,       // Process color within subgroup
 * 
 * // Group distribution
 * const int* gd_array_sizes,
 * int gd_array_sizes_size,
 * int maxsize_gd_array,
 * 
 * // Virtual distribution
 * const int* gd_B_virtual_sizes,
 * int gd_B_virtual_sizes_size,
 * int maxsize_gd_B_virtual,
 * int maxval_gd_B_virtual,
 * 
 * // Orbitals
 * const int* homo,     // Occupied orbitals [2]
 * int homo_size,       // Number of spins
 * const int* virtual_arr, // Virtual orbitals [2]
 * int virtual_size,
 * int maxval_virtual,
 * 
 * // Energies and other data
 * const double* Eigenval, // Eigenvalues (nmo x nspins)
 * int nmo,             // Total MOs
 * int dimen_RI,        // RI basis size
 * int unit_nr,         // Output unit number
 * bool calc_forces,    // Whether forces are computed
 * bool calc_ex         // Whether exchange is computed
 */
void mp2_ri_gpw_compute_en(
    double* Emp2_Cou, double* Emp2_EX, double* Emp2_S,
    double* Emp2_T, double*** BIb_C, int* BIb_C_dims,
    double mp2_memory, int user_block_size, bool print_dgemm_info,
    double scale_S, double scale_T, cp_mpi_comm_t para_env_comm,
    cp_mpi_comm_t para_env_sub_comm,  int color_sub, const int* gd_array_sizes,
    int gd_array_sizes_size, int maxsize_gd_array, const int* gd_B_virtual_sizes,
    int gd_B_virtual_sizes_size, int maxsize_gd_B_virtual, int maxval_gd_B_virtual,
    const int* homo, int homo_size, const int* virtual_arr,
    int virtual_size, int maxval_virtual, const double* Eigenval,
    int nmo, int dimen_RI, int unit_nr,
    bool calc_forces, bool calc_ex
) {
    // Start timer
    offload_timeset("mp2_ri_gpw_compute_en\0");
    
    int nspins = homo_size;

    // ALLOCATE (virtual(nspins))
    // virtual(:) = nmo - homo(:)
    int* virtual = (int*)malloc(nspins * sizeof(int));
    for (int s = 0; s < nspins; s++) {
        virtual[s] = nmo - homo[s];
    }
    
    // ALLOCATE (my_B_size(nspins), my_B_virtual_start(nspins), my_B_virtual_end(nspins))
    int* my_B_size = (int*)malloc(nspins * sizeof(int));
    int* my_B_virtual_start = (int*)malloc(nspins * sizeof(int));
    int* my_B_virtual_end = (int*)malloc(nspins * sizeof(int));

    int* gd_B_virtual_start = (int*)malloc(gd_B_virtual_sizes_size * sizeof(int));
    int* gd_B_virtual_end = (int*)malloc(gd_B_virtual_sizes_size * sizeof(int));
    
    int para_env_size = cp_mpi_comm_size(para_env_comm);
    int para_env_sub_rank = cp_mpi_comm_rank(para_env_sub_comm);
    int para_env_sub_size = cp_mpi_comm_size(para_env_sub_comm);
    int ngroup = para_env_size / para_env_sub_size;
    int my_group_L_start = 1;
    int my_group_L_end = gd_array_sizes[color_sub];
    int my_group_L_size = my_group_L_end - my_group_L_start + 1;

    int integ_group_size = 0;
    int ngroup_out = 0;
    int num_integ_group = 0;

    // Call gemm_c_apu
    gemm_ctx_t *ctx = gemm_ctx_create(GEMM_PU_HOST, GEMM_LIB_BLAS);
    
    // Calculate derived values for get_integ_group_size
    int max_homo = 0;
    for (int i = 0; i < homo_size; i++) {
        if (homo[i] > max_homo) max_homo = homo[i];
    }
    int sum_homo_virtual = 0;
    for (int i = 0; i < homo_size; i++) {
        sum_homo_virtual += homo[i] * virtual[i];
    }
    int product_homo = 1;
    for (int i = 0; i < homo_size; i++) {
        product_homo *= homo[i];
    }
    
    bool calc_group_size = true;
    
    c_mp2_ri_get_integ_group_size(
        &integ_group_size, &ngroup_out, &num_integ_group,
        ngroup, 0, 0,
        mp2_memory, 0,
        homo, homo_size, virtual, virtual_size,
        dimen_RI, calc_forces, unit_nr,
        gd_array_sizes, gd_array_sizes_size,
        gd_B_virtual_sizes, gd_B_virtual_sizes_size,
        maxsize_gd_array, maxsize_gd_B_virtual,
        maxval_gd_B_virtual, maxval_virtual,
        max_homo, sum_homo_virtual, product_homo,
        nspins, calc_group_size,
        para_env_comm
    );

    int comm_exchange_out = 0;
    int comm_rep_out = 0;
    int* new_sizes_array = NULL;
    int* ranges_info_array = NULL;
    int* integ_group_pos2color_sub = NULL;
    int* sizes_array_orig = NULL;
    int my_group_L_size_orig = 0;
    int my_new_group_L_size = 0;
    
    // ranges_info_array dimensions: (4 x rep_size x exchange_size)
    int comm_rep_size = para_env_size / integ_group_size;
    int comm_exchange_size = integ_group_size;
    int ranges_info_dim1 = 4;
    int ranges_info_dim2 = comm_rep_size;
    int ranges_info_dim3 = comm_exchange_size;

    ranges_info_array = (int*)calloc(4 * comm_rep_size * comm_exchange_size, sizeof(int));
    integ_group_pos2color_sub = (int*)calloc(comm_exchange_size, sizeof(int));
    
    c_mp2_ri_create_group(
        &comm_exchange_out, &comm_rep_out,
        (int*)gd_array_sizes, gd_array_sizes_size,
        ranges_info_array, ranges_info_dim1, ranges_info_dim2, ranges_info_dim3,
        integ_group_pos2color_sub, comm_exchange_size,
        sizes_array_orig, (calc_forces) ? gd_array_sizes_size : 0,
        &my_group_L_size, &my_group_L_size_orig, &my_new_group_L_size,
        my_group_L_start, my_group_L_end,
        0, 0,
        color_sub, integ_group_size, num_integ_group,
        calc_forces, my_group_L_size, my_group_L_size_orig
    );

    cp_mpi_comm_t comm_exchange_c = comm_exchange_out;
    int tag = 42;

    int comm_exchange_rank = 0;
    int comm_exchange_size_real = comm_exchange_size_real;

    double my_Emp2_Cou = 0.0;
    double my_Emp2_EX = 0.0;
    double my_Emp2_S = 0.0;
    double my_Emp2_T = 0.0;
    
    for (int jspin = 1; jspin < nspins; jspin++) {
        int j = jspin - 1;
        int current_L_size = BIb_C_dims[3*j + 0];
        int current_virtual = BIb_C_dims[3*j + 1];
        int current_occupied = BIb_C_dims[3*j + 2];
        
        c_replicate_iaK_2intgroup(
            &BIb_C[j], &current_L_size, &current_virtual, &current_occupied,
            comm_exchange_out, comm_rep_out,
            homo[j], gd_array_sizes, gd_array_sizes_size,
            my_B_size[j], my_group_L_size,
            ranges_info_array, 4, comm_rep_size, comm_exchange_size
        );
        
        for (int ispin = 1; ispin < jspin; ispin++) {
            int i = ispin - 1;
            
            // Determine spin cases
            bool my_open_shell_ss = (nspins == 2) && (ispin == jspin);
            bool my_alpha_beta_case = (ispin != jspin);
            bool my_beta_beta_case = my_open_shell_ss && (ispin == 2);
            
            double amp_fac = scale_S + scale_T;
            if (my_alpha_beta_case || my_open_shell_ss) {
                amp_fac = scale_T;
            }
            
            // Allocate local arrays (no block) 
            double* local_ab = NULL;
            double* t_ab = NULL;
            double* local_ba = NULL;

            double* P_ij = NULL;
            double* P_ab = NULL;
            double* Gamma_P_ia = NULL;
            bool P_ij_allocated = false;
            bool P_ab_allocated = false;
            bool Gamma_P_ia_allocated = false;
            
            c_mp2_ri_allocate_no_blk(
                &local_ab, &t_ab, &local_ba,
                homo, virtual, my_B_size,
                my_group_L_size,
                calc_forces, ispin, jspin,
                &P_ij, &P_ab, &Gamma_P_ia,
                &P_ij_allocated, &P_ab_allocated, &Gamma_P_ia_allocated
            );
            
            // Get block size and allocate buffer
            int block_size = 0;
            int ngroup_out2 = 0;
            double* buffer_1D = NULL;
            
            // Pass only the needed spin-specific arrays
            int spin_homo[1] = {homo[i]};
            int spin_virtual[1] = {virtual[i]};
            
            c_mp2_ri_get_block_size(
                &block_size, &ngroup_out2, &buffer_1D,
                0.0, user_block_size,
                para_env_comm, para_env_sub_comm,
                gd_array_sizes, gd_array_sizes_size, maxsize_gd_array,
                gd_B_virtual_sizes, gd_B_virtual_sizes_size,
                maxsize_gd_B_virtual, maxval_gd_B_virtual,
                spin_homo, 1,
                spin_virtual, 1, maxval_virtual,
                dimen_RI, unit_nr,
                num_integ_group,
                my_open_shell_ss, calc_forces
            );
            
            // Communication pattern
            int total_ij_pairs = 0;
            int* ij_map = NULL;
            int my_ij_pairs = 0;
            
            c_mp2_ri_communication(
                my_alpha_beta_case,
                homo[i], homo[j],
                block_size,
                ngroup,
                color_sub,
                my_open_shell_ss,
                unit_nr,
                &total_ij_pairs,
                &ij_map,
                &my_ij_pairs
            );
            
            // Gather my_ij_pairs from all processes in exchange communicator
            int* num_ij_pairs = (int*)malloc(comm_exchange_size * sizeof(int));

            //======================== Current point
            cp_mpi_allgather_int(my_ij_pairs, 1, num_ij_pairs,
                          1, para_env_comm);
            
            int max_ij_pairs = my_ij_pairs;
            for (int p = 0; p < comm_exchange_size; p++) {
                if (num_ij_pairs[p] > max_ij_pairs) {
                    max_ij_pairs = num_ij_pairs[p];
                }
            }
            
            // Allocate block arrays
            double* local_i_aL = NULL;
            double* local_j_aL = NULL;
            double* Y_i_aP = NULL;
            double* Y_j_aP = NULL;
            
            c_mp2_ri_allocate_blk(
                dimen_RI, my_B_size, block_size,
                ispin, jspin,
                &local_i_aL, &local_j_aL,
                &Y_i_aP, &Y_j_aP
            );
            
            // Loop over ij pairs (the main computational loop)
            // Handle 2
            offload_timeset("mp2_ri_gpw_compute_en_RI_loop\0");

            for (int ij_index = 1; ij_index < max_ij_pairs; ij_index++) {

                if (ij_index <= my_ij_pairs) {
                    // Get i, j, and block_size for this pair
                    int ij_counter = (ij_index - (color_sub > 0 ? 1 : 0)) * ngroup + color_sub;
                    // In real code: get from ij_map
                    int my_i = ij_map[0 * total_ij_pairs + ij_counter - 1];
                    int my_j = ij_map[1 * total_ij_pairs + ij_counter - 1];
                    int my_block_size = ij_map[2 * total_ij_pairs + ij_counter - 1];

                    // fill local_i_aL and local_j_aL
                    // call fill_local_i_aL

                    int L_size = gd_array_sizes[comm_exchange_rank];
                    // const int* ranges_info_array;
                    int ranges_info_rep_size = comm_rep_size;

                    fill_local_i_aL(
                        local_i_aL,                    // Destination
                        dimen_RI,                      // local_i_aL_L_size
                        my_B_size[i],                  // local_i_aL_virtual
                        my_block_size,                 // local_i_aL_block
                        ranges_info_array,             // ranges_info_array
                        ranges_info_rep_size,
                        BIb_C[i],                      // Source: BIb_C_rec
                        L_size,                        // BIb_C_rec_L_size
                        my_B_size[i],                  // BIb_C_rec_virtual
                        my_block_size                  // BIb_C_rec_block
                    );

                    fill_local_i_aL(
                        local_j_aL,                    // Destination
                        dimen_RI,                      // local_j_aL_L_size
                        my_B_size[j],                  // local_j_aL_virtual
                        my_block_size,                 // local_j_aL_block
                        ranges_info_array,             // ranges_info_array
                        ranges_info_rep_size,
                        BIb_C[j],                      // Source: BIb_C_rec
                        L_size,                        // BIb_C_rec_L_size
                        my_B_size[j],                  // BIb_C_rec_virtual
                        my_block_size                  // BIb_C_rec_block
                    );

                    // Handle 3
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
                            int send_i = ij_map[0 * total_ij_pairs + ij_counter_send - 1];
                            int send_j = ij_map[1 * total_ij_pairs + ij_counter_send - 1];
                            
                            // Occupied i: send and receive data
                            // Fortran BI_C_rec(1:rec_L_size, 1:my_B_size(ispin), 1:my_block_size)
                            // C: Flattened as [block][virtual][L]
                            // index: (i_block * virtual + a) * rec_L_size + (L - 1)

                            size_t rec_size_i = (size_t)rec_L_size * my_B_size[i] * my_block_size;
                            double* BI_C_rec_i = buffer_1D;
                            // BI_C_rec = 0.0_dp memset can work?
                            // === CHECK
                            memset(BI_C_rec_i, 0, rec_L_size * sizeof(double));

                            // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                            //                        proc_send, BI_C_rec, proc_receive, tag)
                            size_t send_size_i = (size_t)my_group_L_size * my_B_size[i] * my_block_size;
                            // size_t offser_i = ((size_t)(send_i - 1) * my_B_size[i]);
                            double* send_buffer_i = &BIb_C[i][((size_t)(send_i - 1) * my_B_size[i]) * my_group_L_size];

                            /*========= VERSION 1
                            cp_mpi_sendrecv_double(
                                const double *sendbuf,
                                const int sendcount,
                                const int dest,
                                const int sendtag,
                                double *recvbuf,
                                const int recvcount,
                                const int source,
                                const int recvtag,
                                const cp_mpi_comm_t comm
                            ); */

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
                                my_B_size[i],                  // local_i_aL_virtual
                                my_block_size,                 // local_i_aL_block
                                ranges_info_array,             // ranges_info_array
                                comm_rep_size,                 // ranges_info_rep_size
                                BI_C_rec_i,                    // Source: BIb_C_rec
                                rec_L_size,                    // BIb_C_rec_L_size
                                my_B_size[i],                  // BIb_C_rec_virtual
                                my_block_size                  // BIb_C_rec_block
                            );

                            // Occupied j: send and receive data
                            size_t rec_size_j = (size_t)rec_L_size * my_B_size[j] * my_block_size;
                            double* BI_C_rec_j = buffer_1D + rec_size_i; // Start of receive vuffer for j
                            // BI_C_rec = 0.0_dp memset can work?
                            memset(BI_C_rec_j, 0, rec_size_j * sizeof(double));

                            // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                            //                        proc_send, BI_C_rec, proc_receive, tag)
                            size_t send_size_j = (size_t)my_group_L_size * my_B_size[j] * my_block_size;
                            // size_t offser_j = ((size_t)(send_j - 1) * my_B_size[j]);
                            double* send_buffer_j = &BIb_C[j][((size_t)(send_j - 1) * my_B_size[j]) * my_group_L_size];

                            /**============== VERSION 1
                            cp_mpi_sendrecv_double(
                                const double *sendbuf,
                                const int sendcount,
                                const int dest,
                                const int sendtag,
                                double *recvbuf,
                                const int recvcount,
                                const int source,
                                const int recvtag,
                                const cp_mpi_comm_t comm
                            ); */
                            cp_mpi_sendrecv_double(
                                send_buffer_j,
                                (int)send_size_j,
                                proc_send,
                                tag,
                                BI_C_rec_j,
                                (int)rec_size_j,
                                proc_receive,
                                tag,
                                comm_exchange_c
                            );

                            fill_local_i_aL(
                                local_j_aL,                    // Destination
                                dimen_RI,                      // local_i_aL_L_size
                                my_B_size[j],                  // local_i_aL_virtual
                                my_block_size,                 // local_i_aL_block
                                ranges_info_array,             // ranges_info_array
                                comm_rep_size,                 // ranges_info_rep_size
                                BI_C_rec_j,                    // Source: BIb_C_rec
                                rec_L_size,                    // BIb_C_rec_L_size
                                my_B_size[j],                  // BIb_C_rec_virtual
                                my_block_size                  // BIb_C_rec_block
                            );
                        }
                        else {
                            // No work to do - we only receive data
                            // OCCUPIED i: Receive data only

                            size_t rec_size_i = (size_t)rec_L_size * my_B_size[i] * my_block_size;
                            double* BI_C_rec_i = buffer_1D;
                            memset(BI_C_rec_i, 0, rec_size_i * sizeof(double));

                            double dummy_send = 0.0;
                            
                            // Send: dummy_send to proc_send (not actually used)
                            // Receive: BI_C_rec_i from proc_receive
                            // ===========IMPLEMENT cp_mpi_recv_double AND REPLACE THIS CALL

                            /**
                            cp_mpi_sendrecv_double(
                                &dummy_send,                // Dummy send buffer
                                0,                          // Send count = 0 (nothing to send)
                                proc_send,                  // Destination
                                tag,                        // Send tag
                                BI_C_rec_i,                 // Receive buffer
                                (int)rec_size_i,            // Receive count
                                proc_receive,               // Source
                                tag,                        // Receive tag
                                comm_exchange_c             // Communicator
                            );
                            cp_mpi_recv_double(
                                double *recvbuf,
                                const int recvcount,
                                const int source,
                                const int tag,
                                const cp_mpi_comm_t comm
                            );
                             */

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
                                my_B_size[i],                  // local_i_aL_virtual
                                my_block_size,                 // local_i_aL_block
                                ranges_info_array,             // ranges_info_array
                                comm_rep_size,                 // ranges_info_rep_size
                                BI_C_rec_i,                    // Source: BIb_C_rec
                                rec_L_size,                    // BIb_C_rec_L_size
                                my_B_size[i],                  // BIb_C_rec_virtual
                                my_block_size                  // BIb_C_rec_block
                            );
                            
                            // OCCUPIED j: Receive data only
                            size_t rec_size_j = (size_t)rec_L_size * my_B_size[j] * my_block_size;
                            double* BI_C_rec_j = buffer_1D + rec_size_i;
                            memset(BI_C_rec_j, 0, rec_size_j * sizeof(double));
                            
                            // Send: dummy_send to proc_send (not actually used)
                            // Receive: BI_C_rec_j from proc_receive
                            // ===========IMPLEMENT cp_mpi_recv_double AND REPLACE THIS CALL

                            /**
                            cp_mpi_sendrecv_double(
                                &dummy_send,                // Dummy send buffer
                                0,                          // Send count = 0 (nothing to send)
                                proc_send,                  // Destination
                                tag,                        // Send tag
                                BI_C_rec_j,                 // Receive buffer
                                (int)rec_size_j,            // Receive count
                                proc_receive,               // Source
                                tag,                        // Receive tag
                                comm_exchange_c             // Communicator
                            );
                            cp_mpi_recv_double(
                                double *recvbuf,
                                const int recvcount,
                                const int source,
                                const int tag,
                                const cp_mpi_comm_t comm
                            );
                             */

                            cp_mpi_recv_double(
                                BI_C_rec_j,                 // Receive buffer
                                (int)rec_size_j,            // Receive count
                                proc_receive,               // Source (shoul be proc_send?)
                                tag,                        // Receive tag
                                comm_exchange_c             // Communicator
                            );
                            
                            // Fill local_j_aL with received data
                            // local_j_aL(:, :, 1:my_block_size) = BI_C_rec_j(:, 1:my_B_size(jspin), 1:my_block_size)
                            fill_local_i_aL(
                                local_j_aL,                    // Destination
                                dimen_RI,                      // local_j_aL_L_size
                                my_B_size[j],                  // local_j_aL_virtual
                                my_block_size,                 // local_j_aL_block
                                ranges_info_array,             // ranges_info_array
                                comm_rep_size,                 // ranges_info_rep_size
                                BI_C_rec_j,                    // Source: BIb_C_rec
                                rec_L_size,                    // BIb_C_rec_L_size
                                my_B_size[j],                  // BIb_C_rec_virtual
                                my_block_size                  // BIb_C_rec_block
                            );
                        }
                    }
                    // Handle 3
                    offload_timestop();

                    // loop over the block elements
                    for (int iiB = 1; iiB < my_block_size; iiB++) {
                        for (int jjB = 1; jjB < my_block_size; jjB++) {
                            // ====== EXPASION BLOCK
                            offload_timeset("mp2_ri_gpw_compute_en_RI_expasion\0");
                            memset(local_ab, 0, (size_t)my_B_size[i] * my_B_size[i] * sizeof(double));
                            // Use pointer to replace ASSOCIATE block
                            double* my_local_i_aL = &local_i_aL[(size_t)(iiB - 1) * my_B_size[i] * dimen_RI];
                            double* my_local_j_aL = &local_j_aL[(size_t)(jjB - 1) * my_B_size[j] * dimen_RI];

                            // DGEMM: local_ab = my_local_i_aL^T * my_local_j_aL
                            // A: my_local_i_aL (dimen_RI x my_B_size[i])
                            // B: my_local_j_aL (dimen_RI x my_B_size[j])
                            // Result: local_ab (my_B_size[i] x my_B_size[j])
                            gemm_ctx_dgemm(
                                ctx, 'T', 'N',
                                my_B_size[i], my_B_size[j], dimen_RI,
                                1.0, my_local_i_aL, dimen_RI,
                                my_local_j_aL, dimen_RI,
                                0.0, &local_ab[0], my_B_size[i]
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
                                    (int)(dimen_RI * my_B_size[i]),     // Send count
                                    proc_send,                          // Destination
                                    tag,                                // Send tag
                                    external_i_aL,                      // Receive buffer
                                    (int)ext_size,                      // Receive count
                                    proc_receive,                       // Source
                                    tag,                                // Receive tag
                                    para_env_sub_comm                   // Communicator
                                );
                                
                                // DGEMM: local_ab += external_i_aL^T * my_local_j_aL
                                // external_i_aL: (dimen_RI x rec_B_size)
                                // my_local_j_aL: (dimen_RI x my_B_size[j])
                                /**
                                 * void gemm_ctx_dgemm(
                                        gemm_ctx_t *ctx,    // GEMM context
                                        char transa,        // Transpose A? 'T' or 'N'
                                        char transb,        // Transpose B? 'T' or 'N'
                                        int m,              // Rows of C
                                        int n,              // Columns of C
                                        int k,              // Inner dimension
                                        double alpha,       // Scaling for A*B
                                        const double *A,    // Matrix A
                                        int lda,            // Leading dimension of A
                                        const double *B,    // Matrix B
                                        int ldb,            // Leading dimension of B
                                        double beta,        // Scaling for C
                                        double *C,          // Matrix C (output)
                                        int ldc             // Leading dimension of C
                                    );
                                 */
                                gemm_ctx_dgemm(
                                    ctx, 'T', 'N',
                                    rec_B_size,
                                    my_B_size[j],
                                    dimen_RI,
                                    1.0,
                                    external_i_aL,
                                    dimen_RI,
                                    my_local_j_aL,
                                    dimen_RI,
                                    1.0,
                                    &local_ab[(rec_B_virtual_start - 1) * my_B_size[j]],
                                    my_B_size[i]
                                );
                            }

                            offlad_timeset("mp2_ri_gpw_compute_en_RI_ener\0");
                            // Calculate Coulomb only MP2
                            double sym_fac = (my_i == my_j) ? 1.0 : 2.0;
                            if (my_alpha_beta_case) sym_fac = 0.5;

                            // DO b = 1, my_B_size(jspin)
                            for (int b = 0; b < my_B_size[j]; b++) {
                                int b_global = b + my_B_virtual_start[j] - 1;

                                // DO a = 1, virtual(ispin)
                                for (int a = 0; a < virtual[i]; a++) {
                                    double integral = local_ab[a * my_B_size[j] + b];
                                    double divi_part = Eigenval[(homo[i] + a) * nspins + i] + 
                                        Eigenval[(homo[j] + b_global) * nspins + j] -
                                        Eigenval[(my_i + iiB - 1) * nspins + i] -
                                        Eigenval[(my_j + jjB - 1) * nspins + j];
                                    my_Emp2_Cou -= sym_fac * 2.0 * integral * integral / divi_part;
                                }
                            }
                            offload_timestop();
                        }
                    }

                } else {
                    int my_block_size = 1;
                    offlad_timeset("mp2_ri_gpw_compute_en_RI_comm\0");
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
                            int send_i = ij_map[0 * total_ij_pairs + ij_counter_send - 1];
                            int send_j = ij_map[1 * total_ij_pairs + ij_counter_send - 1];
                            
                            // Occupied i: send and receive data
                            // Fortran BI_C_rec(1:rec_L_size, 1:my_B_size(ispin), 1:my_block_size)
                            // C: Flattened as [block][virtual][L]
                            // index: (i_block * virtual + a) * rec_L_size + (L - 1)

                            size_t rec_size_i = (size_t)rec_L_size * my_B_size[i] * my_block_size;
                            double* BI_C_rec_i = buffer_1D;
                            memset(BI_C_rec_i, 0, rec_L_size * sizeof(double));

                            // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                            //                        proc_send, BI_C_rec, proc_receive, tag)
                            size_t send_size_i = (size_t)my_group_L_size * my_B_size[i] * my_block_size;
                            // size_t offser_i = ((size_t)(send_i - 1) * my_B_size[i]);
                            double* send_buffer_i = &BIb_C[i][((size_t)(send_i - 1) * my_B_size[i]) * my_group_L_size];

                            cp_mpi_send_double(
                                send_buffer_i,
                                (int)send_size_i,
                                proc_send,
                                tag,
                                comm_exchange_c
                            );

                            // Occupied j: send and receive data
                            size_t rec_size_j = (size_t)rec_L_size * my_B_size[j] * my_block_size;
                            double* BI_C_rec_j = buffer_1D + rec_size_i; // Start of receive vuffer for j
                            // BI_C_rec = 0.0_dp memset can work?
                            memset(BI_C_rec_j, 0, rec_size_j * sizeof(double));

                            // CALL comm_exchange%sendrecv(BIb_C(ispin)%array(:, :, send_i:send_i + my_block_size - 1), &
                            //                        proc_send, BI_C_rec, proc_receive, tag)
                            size_t send_size_j = (size_t)my_group_L_size * my_B_size[j] * my_block_size;
                            // size_t offser_j = ((size_t)(send_j - 1) * my_B_size[j]);
                            double* send_buffer_j = &BIb_C[j][((size_t)(send_j - 1) * my_B_size[j]) * my_group_L_size];

                            cp_mpi_send_double(
                                send_buffer_j,
                                (int)send_size_j,
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

            free(local_ab);
            if (t_ab) free(t_ab);
            if (local_ba) free(local_ba);
            free(buffer_1D);
            free(ij_map);
            free(num_ij_pairs);
            free(local_i_aL);
            free(local_j_aL);
            if (Y_i_aP) free(Y_i_aP);
            if (Y_j_aP) free(Y_j_aP);
        }
    }
    free(virtual);
    free(my_B_size);
    free(my_B_virtual_start);
    free(my_B_virtual_end);
    free(ranges_info_array);
    free(integ_group_pos2color_sub);
    if (sizes_array_orig) free(sizes_array_orig);
    
    // Return energies
    /*
    cp_mpi_sum_double(my_Emp2_Cou, const int count,
                       const cp_mpi_comm_t comm);
           
    * cp_mpi_comm para_env_comm
    */
    cp_mpi_sum_double(&my_Emp2_Cou, 1, para_env_comm);
    cp_mpi_sum_double(&my_Emp2_EX, 1, para_env_comm);
    *Emp2_Cou += my_Emp2_Cou;
    *Emp2_EX += my_Emp2_EX; // Emp2_EX + my_Emp2_EX;
    *Emp2_S += my_Emp2_S;
    *Emp2_T += my_Emp2_T;

    /**
     * ! ============ REPLACE WITH cp_mpi_comm_free
      CALL comm_exchange%free()
      CALL comm_rep%free()
     */
    cp_mpi_comm_free(comm_exchange_c);

    /**
     * !================= USE gemm_c_api
      ! release memory allocated by local_gemm when run on GPU. local_gemm_ctx is null on cpu only runs
      CALL mp2_env%local_gemm_ctx%destroy()
      CALL timestop(handle)
     */
    gemm_ctx_destroy(ctx);
    // Timer stop
    offload_timestop();
}