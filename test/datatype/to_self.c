/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <inttypes.h>

#include <limits.h>
#include <knem_io.h>
#include <sys/uio.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "opal/datatype/opal_datatype.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/include/opal/sys/cma.h"

#define IOV_MAX sysconf(_SC_IOV_MAX)

#if 0 && OPEN_MPI
extern void ompi_datatype_dump( MPI_Datatype ddt );
#define MPI_DDT_DUMP(ddt) ompi_datatype_dump( (ddt) )
#else
#define MPI_DDT_DUMP(ddt)
#endif  /* OPEN_MPI */

#define CL sysconf (_SC_LEVEL1_DCACHE_LINESIZE)

#define L1size sysconf(_SC_LEVEL1_DCACHE_SIZE)
#define L2size sysconf(_SC_LEVEL2_CACHE_SIZE)
#define L3size sysconf(_SC_LEVEL2_CACHE_SIZE)
static void cache_flush(){
    char *cache = (char*)calloc(L1size+L2size+L3size, sizeof(char));
    free(cache);
}

static struct iovec *get_iov( ompi_datatype_t *ddt ){
    return ddt->super.iov;
}

static int get_iovcnt( ompi_datatype_t *ddt )
{
    return ddt->super.iovcnt;
}

static MPI_Datatype
create_random_indexed( int count, int seed )
{
    MPI_Datatype ddt;
    int indices[count], block[count];

    srand(seed);
    indices[0] = 0;
    block[0] = rand() % 64;
    for( int i = 1; i < count; i++ ){
        indices[i] = i * 64 + rand() % 64;
        if( indices[i] % 64 != 0 ){
            block[i] = rand() % (indices[i] % 64);
        } else {
            block[i] = rand() % 64;
        }
    }

    MPI_Type_indexed( count, block, indices, MPI_CHAR, &ddt );
    MPI_Type_commit( &ddt );

    return ddt;
}

static MPI_Datatype
create_diagonal( int count )
{
    MPI_Datatype ddt;
    int indices[count], block[count];
    
    for( int i = 0; i < count; i++ ){
        indices[i] = i + i * count;
        block[i] = 1;
    }

    MPI_Type_indexed( count, block, indices, MPI_DOUBLE, &ddt );
    MPI_Type_commit( &ddt );

    return ddt;
}

static MPI_Datatype
create_upper_triangle( int count )
{
    MPI_Datatype ddt;
    int indices[count], block[count];

    for( int i = 0; i < count; i++ ){
        indices[i] = i + i * count;
        block[i] = count - i;
    }

    MPI_Type_indexed( count, block, indices, MPI_DOUBLE, &ddt );
    MPI_Type_commit( &ddt );

    return ddt;
}

static MPI_Datatype
create_lower_triangle( int count )
{
    MPI_Datatype ddt;
    int indices[count], block[count];

    for( int i = 0; i < count; i++ ){
        indices[i] = i * count;
        block[i] = i;
    }

    MPI_Type_indexed( count, block, indices, MPI_DOUBLE, &ddt );
    MPI_Type_commit( &ddt );

    return ddt;
}

static MPI_Datatype
create_merged_contig_with_gaps(int count)  /* count of the basic datatype */
{
    int array_of_blocklengths[] = {1, 1, 1};
    MPI_Aint array_of_displacements[] = {0, 8, 16};
    MPI_Datatype array_of_types[] = {MPI_DOUBLE, MPI_LONG, MPI_CHAR};
    MPI_Datatype type;

    MPI_Type_create_struct(3, array_of_blocklengths,
                           array_of_displacements, array_of_types,
                           &type);
    if( 1 < count ) {
        MPI_Datatype temp = type;
        MPI_Type_contiguous(count, temp, &type);
    }
    MPI_Type_commit(&type);
    MPI_DDT_DUMP( type );
    return type;
}

/* Create a non-contiguous resized datatype */
struct structure {
    double not_transfered;
    double transfered_1;
    double transfered_2;
};

static MPI_Datatype
create_struct_constant_gap_resized_ddt( int number,  /* IGNORED: number of repetitions */
                                        int contig_size,  /* IGNORED: number of elements in a contiguous chunk */
                                        int gap_size )    /* IGNORED: number of elements in a gap */
{
    struct structure data[1];
    MPI_Datatype struct_type, temp_type;
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    int blocklens[2] = {1, 1};
    MPI_Aint disps[3];

    MPI_Get_address(&data[0].transfered_1, &disps[0]);
    MPI_Get_address(&data[0].transfered_2, &disps[1]);
    MPI_Get_address(&data[0], &disps[2]);
    disps[1] -= disps[2]; /*  8 */
    disps[0] -= disps[2]; /* 16 */

    MPI_Type_create_struct(2, blocklens, disps, types, &temp_type);
    MPI_Type_create_resized(temp_type, 0, sizeof(data[0]), &struct_type);
    MPI_Type_commit(&struct_type);
    MPI_Type_free(&temp_type);
    MPI_DDT_DUMP( struct_type );

    return struct_type;
}

/* Create a datatype similar to the one use by HPL */
static MPI_Datatype
create_indexed_constant_gap_ddt( int number,  /* number of repetitions */
                                 int contig_size,  /* number of elements in a contiguous chunk */
                                 int gap_size )    /* number of elements in a gap */
{
    MPI_Datatype dt, *types;
    int i, *bLength;
    MPI_Aint* displ;

    types = (MPI_Datatype*)malloc( sizeof(MPI_Datatype) * number );
    bLength = (int*)malloc( sizeof(int) * number );
    displ = (MPI_Aint*)malloc( sizeof(MPI_Aint) * number );

    types[0] = MPI_DOUBLE;
    bLength[0] = contig_size;
    displ[0] = 0;
    for( i = 1; i < number; i++ ) {
        types[i] = MPI_DOUBLE;
        bLength[i] = contig_size;
        displ[i] = displ[i-1] + sizeof(double) * (contig_size + gap_size);
    }
    MPI_Type_create_struct( number, bLength, displ, types, &dt );
    MPI_DDT_DUMP( dt );
    free(types);
    free(bLength);
    free(displ);
    MPI_Type_commit( &dt );
    return dt;
}

static MPI_Datatype
create_optimized_indexed_constant_gap_ddt( int number,  /* number of repetitions */
                                           int contig_size,  /* number of elements in a contiguous chunk */
                                           int gap_size )    /* number of elements in a gap */
{
    MPI_Datatype dt;

    MPI_Type_vector( number, contig_size, (contig_size + gap_size), MPI_DOUBLE, &dt );
    MPI_Type_commit( &dt );
    MPI_DDT_DUMP( dt );
    return dt;
}

typedef struct {
   int i[2];
   float f;
} internal_struct;
typedef struct {
   int v1;
   int gap1;
   internal_struct is[3];
} ddt_gap;

static MPI_Datatype
create_indexed_gap_ddt( void )
{
    ddt_gap dt[2];
    MPI_Datatype dt1, dt2, dt3;
    int bLength[2] = { 2, 1 };
    MPI_Datatype types[2] = { MPI_INT, MPI_FLOAT };
    MPI_Aint displ[2];

    MPI_Get_address( &(dt[0].is[0].i[0]), &(displ[0]) );
    MPI_Get_address( &(dt[0].is[0].f), &(displ[1]) );
    displ[1] -= displ[0];
    displ[0] -= displ[0];
    MPI_Type_create_struct( 2, bLength, displ, types, &dt1 );
    /*MPI_DDT_DUMP( dt1 );*/
    MPI_Type_contiguous( 3, dt1, &dt2 );
    /*MPI_DDT_DUMP( dt2 );*/
    bLength[0] = 1;
    bLength[1] = 1;
    MPI_Get_address( &(dt[0].v1), &(displ[0]) );
    MPI_Get_address( &(dt[0].is[0]), &(displ[1]) );
    displ[1] -= displ[0];
    displ[0] -= displ[0];
    types[0] = MPI_INT;
    types[1] = dt2;
    MPI_Type_create_struct( 2, bLength, displ, types, &dt3 );
    /*MPI_DDT_DUMP( dt3 );*/
    MPI_Type_free( &dt1 );
    MPI_Type_free( &dt2 );
    MPI_Type_contiguous( 10, dt3, &dt1 );
    MPI_DDT_DUMP( dt1 );
    MPI_Type_free( &dt3 );
    MPI_Type_commit( &dt1 );
    return dt1;
}

static MPI_Datatype
create_indexed_gap_optimized_ddt( void )
{
    MPI_Datatype dt1, dt2, dt3;
    int bLength[3];
    MPI_Datatype types[3];
    MPI_Aint displ[3];

    MPI_Type_contiguous( 40, MPI_BYTE, &dt1 );
    MPI_Type_create_resized( dt1, 0, 44, &dt2 );

    bLength[0] = 4;
    bLength[1] = 9;
    bLength[2] = 36;

    types[0] = MPI_BYTE;
    types[1] = dt2;
    types[2] = MPI_BYTE;

    displ[0] = 0;
    displ[1] = 8;
    displ[2] = 44 * 9 + 8;

    MPI_Type_create_struct( 3, bLength, displ, types, &dt3 );

    MPI_Type_free( &dt1 );
    MPI_Type_free( &dt2 );
    MPI_DDT_DUMP( dt3 );
    MPI_Type_commit( &dt3 );
    return dt3;
}


/********************************************************************
 *******************************************************************/

#define DO_CONTIG                         0x00000001
#define DO_CONSTANT_GAP                   0x00000002
#define DO_INDEXED_GAP                    0x00000004
#define DO_OPTIMIZED_INDEXED_GAP          0x00000008
#define DO_STRUCT_CONSTANT_GAP_RESIZED    0x00000010
#define DO_STRUCT_MERGED_WITH_GAP_RESIZED 0x00000020

#define DO_PACK                         0x01000000
#define DO_UNPACK                       0x02000000
#define DO_ISEND_RECV                   0x04000000
#define DO_ISEND_IRECV                  0x08000000
#define DO_IRECV_SEND                   0x10000000
#define DO_IRECV_ISEND                  0x20000000
#define DO_PINGPONG                     0x40000000

#define MIN_LENGTH   1024
#define MAX_LENGTH   10 * 1024 * 1024

#define REP 20
static int cycles  = 20;
static int trials  = 10;
static int warmups = 2;

static void print_result( int length, int trials, double* timers )
{
    double bandwidth, clock_prec, temp;
    double min_time, max_time, average, std_dev = 0.0;
    double ordered[trials];
    int t, pos, quartile_start, quartile_end;

    for( t = 0; t < trials; ordered[t] = timers[t], t++ );
    for( t = 0; t < trials-1; t++ ) {
        temp = ordered[t];
        pos = t;
        for( int i = t+1; i < trials; i++ ) {
            if( temp > ordered[i] ) {
                temp = ordered[i];
                pos = i;
            }
        }
        if( pos != t ) {
            temp = ordered[t];
            ordered[t] = ordered[pos];
            ordered[pos] = temp;
        }
    }
    quartile_start = trials - (3 * trials) / 4;
    quartile_end   = trials - (1 * trials) / 4;
    clock_prec = MPI_Wtick();
    min_time = ordered[quartile_start];
    max_time = ordered[quartile_start];
    average = ordered[quartile_start];
    for( t = quartile_start + 1; t < quartile_end; t++ ) {
        if( min_time > ordered[t] ) min_time = ordered[t];
        if( max_time < ordered[t] ) max_time = ordered[t];
        average += ordered[t];
    }
    average /= (quartile_end - quartile_start);
    for( t = quartile_start; t < quartile_end; t++ ) {
        std_dev += (ordered[t] - average) * (ordered[t] - average);
    }
    std_dev = sqrt( std_dev/(quartile_end - quartile_start) );
    
    bandwidth = (length * clock_prec) / (1024.0 * 1024.0) / (average * clock_prec);
    printf( "%8d\t%15g\t%10.4f MB/s [min %10g max %10g std %2.2f%%]\n", length, average, bandwidth,
            min_time, max_time, (100.0 * std_dev) / average );
}

static int save_to_file( MPI_Datatype sddt, void *sbuf, int scount, char *filename, int size, int extent )
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    FILE *fp;
    fp = fopen(filename, "w+");

    struct iovec *iov = get_iov( sddt );
    int iovcnt = get_iovcnt( sddt );

    int write_size = 0;

    if( size != extent ){
        for( int q = 0; q < scount; q++ ){
            for( int i = 0; i < iovcnt; i++ ){
                for( int j = 0; j < (int)(iov[i].iov_len); j++ ){
                    fputc( ((char*)sbuf + (ptrdiff_t)(iov[i].iov_base) + (ptrdiff_t)(q * extent))[j], fp );
                    write_size++;
                }
            }
        }
    } else {
        for( int i = 0; i < size*scount; i++ ){
            fputc( ((char*)sbuf)[i], fp );
            write_size++;
        }
    }
    fclose(fp);
    return 1;
}

static int compare_file( char *f1, char *f2 )
{
    FILE *fp1 = fopen(f1, "r"); 
    FILE *fp2 = fopen(f2, "r");

    if (fp1 == NULL || fp2 == NULL) 
    { 
        printf("Error : Files not open"); 
        exit(0); 
    } 

    char ch1 = getc(fp1); 
    char ch2 = getc(fp2); 

    int error = 0, pos = 0, line = 1, first = -1;

    while (ch1 != EOF && ch2 != EOF) 
    { 
        pos++;

        if (ch1 != ch2) {
            //printf("%c %c", ch1, ch2);
            if(first == -1)
                first = (int)pos;
            error++; 
        }

        ch1 = getc(fp1); 
        ch2 = getc(fp2); 
    } 

    //printf("pos %d\n", pos);
    if( first >= 0 ){
        //printf("first occur at pos %d\n", first);
    }
    return error;
} 

static void do_pingpong_test( MPI_Datatype sddt, MPI_Datatype rddt, int scount, int rcount, void* sbuf, void* rbuf )
{
    int rank, position, outsize, size;
    MPI_Aint lb, extent;

    double timers[trials];

    MPI_Type_size( sddt, &size );
    MPI_Type_get_extent( sddt, &lb, &extent );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if( rank == 0 ){
        MPI_Type_size( sddt, &outsize );
        outsize *= scount;

        /* warmup 
        for( int i = 0; i < warmups; i++ ){
            MPI_Send( sbuf, scount, sddt, 1, 0, MPI_COMM_WORLD );
            MPI_Recv( rbuf, rcount, rddt, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
*/
        for( int i = 0; i < trials; i++ ){
            cache_flush();

            timers[i] = MPI_Wtime();
            for( int j = 0; j < cycles; j++ ){
                cache_flush();
                MPI_Send( sbuf, scount, sddt, 1, 0, MPI_COMM_WORLD );
                cache_flush();
                MPI_Recv( rbuf, rcount, rddt, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            }
            timers[i] = (MPI_Wtime() - timers[i]) / 2 / cycles;
        }

//        save_to_file( rddt, (void*)rbuf, rcount, "recvfile", size, (int)extent );
  //      int err = compare_file( "sendfile", "recvfile" );
    //    if( err == 0 )
            print_result( (size_t)outsize, trials, timers );
    //    else
      //      printf("err %d\n", err);
    } else if( rank == 1 ) {
//        save_to_file( sddt, (void*)sbuf, scount, "sendfile", size, (int)extent );
//        for( int i = 0; i < warmups; i++ ){
  //          MPI_Recv( rbuf, rcount, rddt, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    //        MPI_Send( sbuf, scount, sddt, 0, 1, MPI_COMM_WORLD );
      //  }

        for( int i = 0; i < trials; i++ ){
            for( int j = 0; j < cycles; j++ ){
                MPI_Recv( rbuf, rcount, rddt, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                MPI_Send( sbuf, scount, sddt, 0, 1, MPI_COMM_WORLD );
            }
        }
    }
}

static void do_knem_contig_test( MPI_Datatype sddt, MPI_Datatype rddt, int slen, int rlen, int scount, int rcount, void *sbuf, void *rbuf )
{
    int rank, err;
    double timers[REP];
    MPI_Win win;

    MPI_Aint size, lb;
    int extent;
    MPI_Type_size( sddt, &size );
    MPI_Type_get_extent( sddt, &lb, &extent );

    if( size == 0 ){
        size = 4;
        extent = 4;
    }

    int knem_fd = open("/dev/knem", O_RDWR);

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if( rank == 0 ){
        struct knem_cmd_inline_copy icopy;

        int iovcnt = get_iovcnt( rddt ) * rcount;
        if( iovcnt == 0 )
            iovcnt = 1;
        
        struct knem_cmd_param_iovec knem_iov[iovcnt];

        if( size == extent ){
            knem_iov[0].base = (uint64_t)rbuf;
            knem_iov[0].len = rlen;

            icopy.local_iovec_array = (uintptr_t) &knem_iov[0];
            icopy.local_iovec_nr = 1;
        } else if( get_iovcnt(rddt) != 0 ) {
            struct iovec *rddt_iov = get_iov( rddt );

            for( int q = 0; q < rcount; q++ ){
                for( int j = 0; j < get_iovcnt(rddt); j++ ){
                    knem_iov[j + q * get_iovcnt(rddt)].base = (uint64_t)((char*)rbuf + (ptrdiff_t)(rddt_iov[j].iov_base) + (ptrdiff_t)(q * extent) );
                    knem_iov[j + q * get_iovcnt(rddt)].len = (uint64_t)( rddt_iov[j].iov_len );
    
                }
            }

            icopy.local_iovec_array = (uintptr_t) &knem_iov[0];
            icopy.local_iovec_nr = iovcnt;
        }

        MPI_Win_create( NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win );
        for( int i = 0; i < REP; i++ ){
            icopy.remote_offset = 0;
            icopy.write = 0;
            icopy.flags = 0;
            MPI_Recv( &(icopy.remote_cookie), sizeof(uint64_t), MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            cache_flush(); 
            timers[i] = MPI_Wtime();
            MPI_Win_fence( 0, win );
            err = ioctl(knem_fd, KNEM_CMD_INLINE_COPY, &icopy);
            MPI_Win_fence( 0, win );
            cache_flush();
            timers[i] = MPI_Wtime() - timers[i];
        }

//        save_to_file( rddt, (void*)rbuf, rcount, "recvfile", size, (int)extent );
  //      int err = compare_file( "sendfile", "recvfile" );
    //    if( err == 0 )
            print_result( size * scount, REP, timers );
    //    else
      //      printf("err %d\n", err);
    } else if( rank == 1 ){
        //save_to_file( sddt, (void*)sbuf, scount, "sendfile", size, (int)extent );
       
        int iovcnt = get_iovcnt( sddt ) * scount;
        if( iovcnt == 0 )
            iovcnt = 1;
        
        struct knem_cmd_create_region create;
        struct knem_cmd_param_iovec knem_iov[iovcnt];

        if( size == extent ){
            knem_iov[0].base = (uint64_t)sbuf;
            knem_iov[0].len = (uint64_t)slen;
            create.iovec_nr = 1;
        } else {
            struct iovec *sddt_iov = get_iov( sddt );

            for( int q = 0; q < scount; q++ ){
                for( int j = 0; j < get_iovcnt(sddt); j++ ){
                    knem_iov[j + q * get_iovcnt(rddt)].base = (uint64_t)((char*)sbuf + (ptrdiff_t)(sddt_iov[j].iov_base) + (ptrdiff_t)(q * extent) );
                    knem_iov[j + q * get_iovcnt(rddt)].len = (uint64_t)( sddt_iov[j].iov_len );
                }
            }
            
            create.iovec_nr = iovcnt;
        }

        create.iovec_array = (uintptr_t) &knem_iov[0];
        
        MPI_Win_create( sbuf, slen, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win );
        for( int i = 0; i < REP; i++ ){
            create.flags = KNEM_FLAG_SINGLEUSE;
            create.protection = PROT_READ;
            err = ioctl( knem_fd, KNEM_CMD_CREATE_REGION, &create );
            MPI_Send( &(create.cookie), sizeof(uint64_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD );
            cache_flush();    
            MPI_Win_fence( 0, win );
            MPI_Win_fence( 0, win );
        }

    
    }

}

static void do_cma_contig_test( MPI_Datatype sddt, MPI_Datatype rddt, int slen, int rlen, int scount, int rcount, void *sbuf, void *rbuf )
{
    int rank, err;
    double timers[REP];
    MPI_Win win;

    pid_t pid;
    MPI_Aint extent, lb;
    int size;
    MPI_Type_size( sddt, &size );
    MPI_Type_get_extent( sddt, &lb, &extent );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if( rank == 0 ){
        MPI_Win_create( NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win );
        
        int send_iovcnt;
        struct iovec *send_iov;

        int iovcnt = get_iovcnt( rddt );
        struct iovec *iov = malloc( sizeof(struct iovec) * rcount * iovcnt );
        struct iovec *rddt_iov = get_iov( rddt );

        for( int i = 0; i < rcount; i++ ){
            for( int j = 0; j < iovcnt; j++ ){
                iov[ i * iovcnt + j ].iov_base = (char*)rbuf + (ptrdiff_t)( rddt_iov[j].iov_base ) + (ptrdiff_t)(i * (int)extent);
                iov[ i * iovcnt + j ].iov_len = rddt_iov[j].iov_len;
            }
        }

        MPI_Recv( &pid, sizeof(pid_t), MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( &send_iovcnt, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        send_iov = malloc( sizeof(struct iovec) * send_iovcnt );

        MPI_Recv( send_iov, sizeof(struct iovec) * send_iovcnt, MPI_BYTE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        for( int i = 0; i < REP; i++ ){
            cache_flush(); 
            timers[i] = MPI_Wtime();
            MPI_Win_fence( 0, win );
            
            int s = send_iovcnt / IOV_MAX, pos = 0;
            while( s != 0 ){
                process_vm_readv( pid, &(iov[pos * IOV_MAX]), IOV_MAX, &(send_iov[pos * IOV_MAX]), IOV_MAX, 0 );
                s--;
                pos++;
            }

            process_vm_readv( pid, &(iov[s * IOV_MAX]), send_iovcnt % IOV_MAX, &(send_iov[s * IOV_MAX]), send_iovcnt % IOV_MAX, 0 );
            
            MPI_Win_fence( 0, win );
            cache_flush();
            timers[i] = MPI_Wtime() - timers[i];
        }

        save_to_file( rddt, (void*)rbuf, rcount, "recvfile", size, (int)extent );
        int err = compare_file( "sendfile", "recvfile" );
        if( err == 0 )
            print_result( size * scount, REP, timers );
        else
            printf("err %d\n", err);

        free(send_iov);
        free(iov);

    } else if( rank == 1 ){
        save_to_file( sddt, (void*)sbuf, scount, "sendfile", size, (int)extent );
        MPI_Win_create( sbuf, slen, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win );

        int iovcnt = get_iovcnt( sddt );
        struct iovec *iov = malloc( sizeof(struct iovec) * scount * iovcnt );
        struct iovec *sddt_iov = get_iov( sddt );

        for( int i = 0; i < scount; i++ ){
            for( int j = 0; j < iovcnt; j++ ){
                iov[ i * iovcnt + j ].iov_base = (char*)sbuf + (ptrdiff_t)( sddt_iov[j].iov_base ) + (ptrdiff_t)(i * (int)extent);
                iov[ i * iovcnt + j ].iov_len = sddt_iov[j].iov_len;
            }
        }

        pid = getpid();
        MPI_Send( &pid, sizeof(pid_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD );
        scount *= iovcnt;
        MPI_Send( &scount, 1, MPI_INT, 0, 1, MPI_COMM_WORLD );
        MPI_Send( iov, scount * sizeof(struct iovec), MPI_BYTE, 0, 2, MPI_COMM_WORLD );
        for( int i = 0; i < REP; i++ ){
            cache_flush();
            MPI_Win_fence( 0, win );
            MPI_Win_fence( 0, win );
        }
        free(iov);
        
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


static int pack( int cycles,
        MPI_Datatype sdt, int scount, void* sbuf,
        void* packed_buf )
{
    int position, myself, c, t, outsize;
    double timers[trials];

    MPI_Type_size( sdt, &outsize );
    outsize *= scount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
        }
    }
    
    cache_flush(); 
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
            cache_flush();
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }

    print_result( outsize, trials, timers );

    return 0;
}

static int unpack( int cycles,
                   void* packed_buf,
                   MPI_Datatype rdt, int rcount, void* rbuf )
{
    int position, myself, c, t, insize;
    double timers[trials];

    MPI_Type_size( rdt, &insize );
    insize *= rcount;

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < warmups; t++ ) {
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Unpack(packed_buf, insize, &position, rbuf, rcount, rdt, MPI_COMM_WORLD);
        }
    }

    cache_flush();
    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();
        for( c = 0; c < cycles; c++ ) {
            position = 0;
            MPI_Unpack(packed_buf, insize, &position, rbuf, rcount, rdt, MPI_COMM_WORLD);
            cache_flush();
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    
    print_result( insize, trials, timers );
    
    return 0;
}

static int do_test_for_ddt( int doop, MPI_Datatype sddt, MPI_Datatype rddt, int length, int rank )
{
    MPI_Aint lb, extent;
    char *sbuf, *rbuf;
    int i;

    MPI_Type_get_extent( sddt, &lb, &extent );
    sbuf = (char*)malloc( length );
    rbuf = (char*)malloc( length );

    if( rank == 0 )
        printf("# pingpong 500 times (length %d)\n", length);
    MPI_Barrier( MPI_COMM_WORLD );
    for( i = length / extent / 20; i <= length / extent; i += length / extent / 20 ){
        do_pingpong_test( sddt, rddt, i, i, sbuf, rbuf );
    }

    if( rank == 0 )
        printf("# cma 500 times (length %d)\n", length);
    MPI_Barrier( MPI_COMM_WORLD );
    for( i = length / extent / 20; i <= length / extent; i += length / extent / 20 ){
        do_cma_contig_test( sddt, rddt, i*extent, i*extent, i, i, sbuf, rbuf );
    }

    free( sbuf );
    free( rbuf );
    return 0;
}

int main( int argc, char* argv[] )
{
    int run_tests = 0xffff;  /* do all datatype tests by default */
    int rank, size;
    MPI_Datatype ddt;

    run_tests |= DO_PACK | DO_UNPACK | DO_CONTIG | DO_PINGPONG;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    if( rank > 1 ) {
        MPI_Finalize();
        exit(0);
    }

    if( rank == 0 )
        printf( "\n! vector datatype\n\n" );
    MPI_Type_vector( 64, 1, 8, MPI_DOUBLE, &ddt );
    MPI_Type_commit( &ddt );
    do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
    MPI_Type_free( &ddt );

    if( run_tests & DO_INDEXED_GAP ) {
        if( rank == 0 )
            printf( "\n! indexed gap\n\n" );
        ddt = create_indexed_gap_ddt();
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_OPTIMIZED_INDEXED_GAP ) {
        if( rank == 0 )
            printf( "\n! optimized indexed gap\n\n" );
        ddt = create_indexed_gap_optimized_ddt();
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_CONSTANT_GAP ) {
        if( rank == 0 )
            printf( "\n! constant indexed gap\n\n" );
        ddt = create_indexed_constant_gap_ddt( 80, 100, 1 );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_CONSTANT_GAP ) {
        if( rank == 0 )
            printf( "\n! optimized constant indexed gap\n\n" );
        ddt = create_optimized_indexed_constant_gap_ddt( 80, 100, 1 );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_STRUCT_CONSTANT_GAP_RESIZED ) {
        if( rank == 0 )
            printf( "\n! struct constant gap resized\n\n" );
        ddt = create_struct_constant_gap_resized_ddt( 0, 0, 0 );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
        MPI_Type_free( &ddt );
    }

    if( run_tests & DO_STRUCT_MERGED_WITH_GAP_RESIZED ) {
        if( rank == 0 )
            printf( "\n! struct constant gap resized\n\n" );
        ddt = create_merged_contig_with_gaps( 1 );
        MPI_DDT_DUMP( ddt );
        do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
        MPI_Type_free( &ddt );
    }

    if( rank == 0 )
        printf("\n\n! Diagnal matrix\n");
    ddt = create_diagonal(100);
    do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
    MPI_Type_free( &ddt );
    
    if( rank == 0 )
        printf("\n\n! Upper triangle matrix\n");
    ddt = create_upper_triangle(100);
    do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
    MPI_Type_free( &ddt );

    if( rank == 0 )
        printf("\n\n! Lower triangle matrix\n");
    ddt = create_lower_triangle(100);
    do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
    MPI_Type_free( &ddt );

    if( rank == 0 )
        printf("\n\n! Randomized indexed type\n");
    ddt = create_random_indexed( 16, 0 );
    do_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, rank );
    MPI_Type_free( &ddt );

    MPI_Finalize ();
    exit(0);
}

