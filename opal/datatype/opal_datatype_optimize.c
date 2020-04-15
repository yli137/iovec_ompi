/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2014      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "opal/datatype/opal_datatype.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype_internal.h"

static int32_t
opal_datatype_optimize_short( opal_datatype_t* pData,
                              size_t count,
                              dt_type_desc_t* pTypeDesc )
{
    dt_elem_desc_t* pElemDesc;
    dt_stack_t *pOrigStack, *pStack; /* pointer to the position on the stack */
    int32_t pos_desc = 0;            /* actual position in the description of the derived datatype */
    int32_t stack_pos = 0;
    int32_t nbElems = 0;
    ptrdiff_t total_disp = 0;
    ddt_elem_desc_t last = {.common.flags = 0xFFFF /* all on */, .count = 0, .disp = 0}, compress;
    ddt_elem_desc_t* current;

    pOrigStack = pStack = (dt_stack_t*)malloc( sizeof(dt_stack_t) * (pData->loops+2) );
    SAVE_STACK( pStack, -1, 0, count, 0 );

    pTypeDesc->length = 2 * pData->desc.used + 1 /* for the fake OPAL_DATATYPE_END_LOOP at the end */;
    pTypeDesc->desc = pElemDesc = (dt_elem_desc_t*)malloc( sizeof(dt_elem_desc_t) * pTypeDesc->length );
    pTypeDesc->used = 0;

    assert( OPAL_DATATYPE_END_LOOP == pData->desc.desc[pData->desc.used].elem.common.type );

    while( stack_pos >= 0 ) {
        if( OPAL_DATATYPE_END_LOOP == pData->desc.desc[pos_desc].elem.common.type ) { /* end of the current loop */
            ddt_endloop_desc_t* end_loop = &(pData->desc.desc[pos_desc].end_loop);
            if( 0 != last.count ) {
                CREATE_ELEM( pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                             last.blocklen, last.count, last.disp, last.extent );
                pElemDesc++; nbElems++;
                last.count= 0;
            }
            CREATE_LOOP_END( pElemDesc, nbElems - pStack->index + 1,  /* # of elems in this loop */
                             end_loop->first_elem_disp, end_loop->size, end_loop->common.flags );
            if( --stack_pos >= 0 ) {  /* still something to do ? */
                ddt_loop_desc_t* pStartLoop = &(pTypeDesc->desc[pStack->index - 1].loop);
                pStartLoop->items = pElemDesc->end_loop.items;
                total_disp = pStack->disp;  /* update the displacement position */
            }
            pElemDesc++; nbElems++;
            pStack--;  /* go down one position on the stack */
            pos_desc++;
            continue;
        }
        if( OPAL_DATATYPE_LOOP == pData->desc.desc[pos_desc].elem.common.type ) {
            ddt_loop_desc_t* loop = (ddt_loop_desc_t*)&(pData->desc.desc[pos_desc]);
            int index = GET_FIRST_NON_LOOP( &(pData->desc.desc[pos_desc]) );

            if( loop->common.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS ) {
                ddt_endloop_desc_t* end_loop = (ddt_endloop_desc_t*)&(pData->desc.desc[pos_desc + loop->items]);

                assert(pData->desc.desc[pos_desc + index].elem.disp == end_loop->first_elem_disp);
                compress.common.flags = loop->common.flags;
                compress.common.type =  pData->desc.desc[pos_desc + index].elem.common.type;
                compress.blocklen = pData->desc.desc[pos_desc + index].elem.blocklen;
                for( uint32_t i = index+1; i < loop->items; i++ ) {
                    current = &pData->desc.desc[pos_desc + i].elem;
                    assert(1 ==  current->count);
                    if( (current->common.type == OPAL_DATATYPE_LOOP) ||
                        compress.common.type != current->common.type ) {
                        compress.common.type = OPAL_DATATYPE_UINT1;
                        compress.blocklen = end_loop->size;
                        break;
                    }
                    compress.blocklen += current->blocklen;
                }
                compress.count = loop->loops;
                compress.extent = loop->extent;
                compress.disp = end_loop->first_elem_disp;
                if( compress.extent == (ptrdiff_t)(compress.blocklen * opal_datatype_basicDatatypes[compress.common.type]->size) ) {
                    /* The compressed element is contiguous: collapse it into a single large blocklen */
                    compress.blocklen *= compress.count;
                    compress.extent   *= compress.count;
                    compress.count     = 1;
                }
                /**
                 * The current loop has been compressed and can now be treated as if it
                 * was a data element. We can now look if it can be fused with last,
                 * as done in the fusion of 2 elements below. Let's use the same code.
                 */
                pos_desc += loop->items + 1;
                current = &compress;
                goto fuse_loops;
            }

            /**
             * If the content of the loop is not contiguous there is little we can do
             * that would not incur significant optimization cost and still be beneficial
             * in reducing the number of memcpy during pack/unpack.
             */

            if( 0 != last.count ) {  /* Generate the pending element */
                CREATE_ELEM( pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                             last.blocklen, last.count, last.disp, last.extent );
                pElemDesc++; nbElems++;
                last.count       = 0;
                last.common.type = OPAL_DATATYPE_LOOP;
            }

            /* Can we unroll the loop? */
            if( (loop->items <= 3) && (loop->loops <= 2) ) {
                ptrdiff_t elem_displ = 0;
                for( uint32_t i = 0; i < loop->loops; i++ ) {
                    for( uint32_t j = 0; j < (loop->items - 1); j++ ) {
                        current = &pData->desc.desc[pos_desc + index + j].elem;
                        CREATE_ELEM( pElemDesc, current->common.type, current->common.flags,
                                     current->blocklen, current->count, current->disp + elem_displ, current->extent );
                        pElemDesc++; nbElems++;
                    }
                    elem_displ += loop->extent;
                }
                pos_desc += loop->items + 1;
                goto complete_loop;
            }

            CREATE_LOOP_START( pElemDesc, loop->loops, loop->items, loop->extent, loop->common.flags );
            pElemDesc++; nbElems++;
            PUSH_STACK( pStack, stack_pos, nbElems, OPAL_DATATYPE_LOOP, loop->loops, total_disp );
            pos_desc++;
            DDT_DUMP_STACK( pStack, stack_pos, pData->desc.desc, "advance loops" );

        complete_loop:
            total_disp = pStack->disp;  /* update the displacement */
            continue;
        }
        while( pData->desc.desc[pos_desc].elem.common.flags & OPAL_DATATYPE_FLAG_DATA ) {  /* go over all basic datatype elements */
            current = &pData->desc.desc[pos_desc].elem;
            pos_desc++;  /* point to the next element as current points to the current one */

          fuse_loops:
            if( 0 == last.count ) {  /* first data of the datatype */
                last = *current;
                continue;  /* next data */
            } else {  /* can we merge it in order to decrease count */
                if( (ptrdiff_t)last.blocklen * (ptrdiff_t)opal_datatype_basicDatatypes[last.common.type]->size == last.extent ) {
                    last.extent *= last.count;
                    last.blocklen *= last.count;
                    last.count = 1;
                }
            }

            /* are the two elements compatible: aka they have very similar values and they
             * can be merged together by increasing the count, and/or changing the extent.
             */
            if( (last.blocklen * opal_datatype_basicDatatypes[last.common.type]->size) ==
                (current->blocklen * opal_datatype_basicDatatypes[current->common.type]->size) ) {
                ddt_elem_desc_t save = last;  /* safekeep the type and blocklen */
                if( last.common.type != current->common.type ) {
                    last.blocklen    *= opal_datatype_basicDatatypes[last.common.type]->size;
                    last.common.type  = OPAL_DATATYPE_UINT1;
                }

                if( (last.extent * (ptrdiff_t)last.count + last.disp) == current->disp ) {
                    if( 1 == current->count ) {
                        last.count++;
                        continue;
                    }
                    if( last.extent == current->extent ) {
                        last.count += current->count;
                        continue;
                    }
                }
                if( 1 == last.count ) {
                    /* we can ignore the extent of the element with count == 1 and merge them together if their displacements match */
                    if( 1 == current->count ) {
                        last.extent = current->disp - last.disp;
                        last.count++;
                        continue;
                    }
                    /* can we compute a matching displacement ? */
                    if( (last.disp + current->extent) == current->disp ) {
                        last.extent = current->extent;
                        last.count = current->count + last.count;
                        continue;
                    }
                }
                last.blocklen = save.blocklen;
                last.common.type = save.common.type;
                /* try other optimizations */
            }
            /* are the elements fusionable such that we can fusion the last blocklen of one with the first
             * blocklen of the other.
             */
            if( (ptrdiff_t)(last.disp + (last.count - 1) * last.extent + last.blocklen * opal_datatype_basicDatatypes[last.common.type]->size) ==
                current->disp ) {
                if( last.count != 1 ) {
                    CREATE_ELEM( pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                                 last.blocklen, last.count - 1, last.disp, last.extent );
                    pElemDesc++; nbElems++;
                    last.disp += (last.count - 1) * last.extent;
                    last.count = 1;
                }
                if( last.common.type == current->common.type ) {
                    last.blocklen += current->blocklen;
                } else {
                    last.blocklen = ((last.blocklen * opal_datatype_basicDatatypes[last.common.type]->size) +
                                     (current->blocklen * opal_datatype_basicDatatypes[current->common.type]->size));
                    last.common.type = OPAL_DATATYPE_UINT1;
                }
                last.extent += current->extent;
                if( current->count != 1 ) {
                    CREATE_ELEM( pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                                 last.blocklen, last.count, last.disp, last.extent );
                    pElemDesc++; nbElems++;
                    last = *current;
                    last.count -= 1;
                    last.disp += last.extent;
                }
                continue;
            }
            CREATE_ELEM( pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                         last.blocklen, last.count, last.disp, last.extent );
            pElemDesc++; nbElems++;
            last = *current;
        }
    }

    if( 0 != last.count ) {
        CREATE_ELEM( pElemDesc, last.common.type, OPAL_DATATYPE_FLAG_BASIC,
                     last.blocklen, last.count, last.disp, last.extent );
        pElemDesc++; nbElems++;
    }
    /* cleanup the stack */
    pTypeDesc->used = nbElems - 1;  /* except the last fake END_LOOP */
    free(pOrigStack);
    return OPAL_SUCCESS;
}

int32_t opal_datatype_commit( opal_datatype_t * pData )
{
    ddt_endloop_desc_t* pLast = &(pData->desc.desc[pData->desc.used].end_loop);
    ptrdiff_t first_elem_disp = 0;

    if( pData->flags & OPAL_DATATYPE_FLAG_COMMITTED ) return OPAL_SUCCESS;
    pData->flags |= OPAL_DATATYPE_FLAG_COMMITTED;

    /* We have to compute the displacement of the first non loop item in the
     * description.
     */
    if( 0 != pData->size ) {
        int index;
        dt_elem_desc_t* pElem = pData->desc.desc;

        index = GET_FIRST_NON_LOOP( pElem );
        assert( pElem[index].elem.common.flags & OPAL_DATATYPE_FLAG_DATA );
        first_elem_disp = pElem[index].elem.disp;
    }

    /* let's add a fake element at the end just to avoid useless comparaisons
     * in pack/unpack functions.
     */
    pLast->common.type     = OPAL_DATATYPE_END_LOOP;
    pLast->common.flags    = 0;
    pLast->items           = pData->desc.used;
    pLast->first_elem_disp = first_elem_disp;
    pLast->size            = pData->size;

    /* If there is no datatype description how can we have an optimized description ? */
    if( 0 == pData->desc.used ) {
        pData->opt_desc.length = 0;
        pData->opt_desc.desc   = NULL;
        pData->opt_desc.used   = 0;
        return OPAL_SUCCESS;
    }

    /* If the data is contiguous is useless to generate an optimized version. */
    /*if( pData->size == (pData->true_ub - pData->true_lb) ) return OPAL_SUCCESS; */

    (void)opal_datatype_optimize_short( pData, 1, &(pData->opt_desc) );
    if( 0 != pData->opt_desc.used ) {
        /* let's add a fake element at the end just to avoid useless comparaisons
         * in pack/unpack functions.
         */
        pLast = &(pData->opt_desc.desc[pData->opt_desc.used].end_loop);
        pLast->common.type     = OPAL_DATATYPE_END_LOOP;
        pLast->common.flags    = 0;
        pLast->items           = pData->opt_desc.used;
        pLast->first_elem_disp = first_elem_disp;
        pLast->size            = pData->size;
    }

    /* generate iovec */
    if( pData->iov == NULL ){
        opal_generate_iovec( pData );
        opal_datatype_compress( pData );

        ptrdiff_t disp;
        size_t length;
        uint32_t i = 0;
        char *ptr = pData->compress.storage;
        for( ptr; ptr < pData->compress.storage + pData->compress.iov_length; i++){
            if( 0 == ((uint8_t)0x01 & ptr[0]) ) {  /* last bit = 0: 8 bits */
                opal_datatype_iovec_storage_int8_t* s8 = (opal_datatype_iovec_storage_int8_t*)ptr;
                length = (size_t)s8->length >> 1;
                disp = (ptrdiff_t)s8->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int8_t);
            } else if( 0 == (0x02 & ptr[0]) ) {  /* last 2 bits = 01: 16 bits */
                opal_datatype_iovec_storage_int16_t* s16 = (opal_datatype_iovec_storage_int16_t*)ptr;
                length = (size_t)(s16->length >> 2);
                disp = (ptrdiff_t)s16->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int16_t);
            } else if( 0 == (0x04 & ptr[0]) ) {  /* last 3 bits = 011: 32 bits */
                opal_datatype_iovec_storage_int32_t* s32 = (opal_datatype_iovec_storage_int32_t*)ptr;
                length = (size_t)(s32->length >> 3);
                disp = s32->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int32_t);
            } else {  /* last 3 bits = 111: 64 bits */
                opal_datatype_iovec_storage_int64_t* s64 = (opal_datatype_iovec_storage_int64_t*)ptr;
                length = s64->length >> 3;
                disp = s64->disp;
                ptr += sizeof(opal_datatype_iovec_storage_int64_t);
            }

//            printf("i %d disp %zu len %zu\n",
  //                  i,
    //                disp, length);
        }
    }

    return OPAL_SUCCESS;
}

int32_t 
opal_datatype_compress( opal_datatype_t *pData )
{
    opal_datatype_flexible_storage_t* flexi = (opal_datatype_flexible_storage_t*)&(pData->compress);
    struct iovec *iov = pData->iov;

    uint8_t bytes = sizeof(opal_datatype_iovec_storage_int64_t);

    for( uint32_t i = 0; i < pData->iovcnt; i++ ){
        if( 0 == (0x7FFFFFFFFFFFFF80 & (intptr_t)iov[i].iov_base) ) {
            bytes = sizeof(opal_datatype_iovec_storage_int8_t);
        } else if( 0 == (0x7FFFFFFFFFFF8000 & (intptr_t)iov[i].iov_base) ) {
            bytes = sizeof(opal_datatype_iovec_storage_int16_t);
        } else if( 0 == (0x7FFFFFFF80000000 & (intptr_t)iov[i].iov_base) ) {
            bytes = sizeof(opal_datatype_iovec_storage_int32_t);
        }
        if( bytes < sizeof(opal_datatype_iovec_storage_int32_t) ) {
            if( 0 == (0xFFFFFFFFFFFFFF80 & iov[i].iov_len) ) {  /* single bit = 0 */
                /* follow the number of bits in the displacement */
            } else if( 0 == (0xFFFFFFFFFFFFC000 & iov[i].iov_len) ) {  /* 2 bits = 10 */
                bytes = sizeof(opal_datatype_iovec_storage_int16_t);
            }
        } else if( 0 != (0xFFFFFFFFE0000000 & iov[i].iov_len) ) {  /* 3 bits = 110 */
            bytes = sizeof(opal_datatype_iovec_storage_int64_t);
        }  /* otherwise 3 bits = 111 */

        bytes = sizeof(opal_datatype_iovec_storage_int32_t);
        if( (flexi->iov_pos + bytes) > flexi->iov_length ) {
            size_t new_length = (0 == flexi->iov_length ? 128 : (flexi->iov_length * 2));
            void* ptr = realloc( flexi->storage, new_length);
            if( NULL == ptr ) {  /* oops */
                return 1;
            }
            flexi->storage = ptr;
            flexi->iov_length = new_length;
        }

        switch(bytes) {
            case sizeof(opal_datatype_iovec_storage_int8_t): {
                                                                 opal_datatype_iovec_storage_int8_t* s8 = (opal_datatype_iovec_storage_int8_t*)(flexi->storage + flexi->iov_pos);
                                                                 s8->length = (uint8_t)(iov[i].iov_len) << 1;
                                                                 s8->disp = (int8_t)(intptr_t)(iov[i].iov_base);
                                                                 flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int8_t);
                                                                 break;
                                                             }
            case sizeof(opal_datatype_iovec_storage_int16_t): {
                                                                  opal_datatype_iovec_storage_int16_t* s16 = (opal_datatype_iovec_storage_int16_t*)(flexi->storage + flexi->iov_pos);
                                                                  s16->length = (uint16_t)(iov[i].iov_len) << 2 | (uint16_t)0x01;
                                                                  s16->disp = (int16_t)(intptr_t)(iov[i].iov_base);
                                                                  flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int16_t);
                                                                  break;
                                                              }
            case sizeof(opal_datatype_iovec_storage_int32_t): {
                                                                  opal_datatype_iovec_storage_int32_t* s32 = (opal_datatype_iovec_storage_int32_t*)(flexi->storage + flexi->iov_pos);
                                                                  s32->length = (uint32_t)(iov[i].iov_len) << 3 | (uint32_t)0x03;
                                                                  s32->disp = (int32_t)(intptr_t)(iov[i].iov_base);
                                                                  flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int32_t);
                                                                  break;
                                                              }
            default: {
                         opal_datatype_iovec_storage_int64_t* s64 = (opal_datatype_iovec_storage_int64_t*)(flexi->storage + flexi->iov_pos);
                         s64->length = (uint64_t)(iov[i].iov_len) << 3 | 0x07ULL;
                         s64->disp = (intptr_t)(iov[i].iov_base);
                         flexi->iov_pos += sizeof(opal_datatype_iovec_storage_int64_t);
                         break;
                     }
        }

    }

    flexi->storage = realloc( flexi->storage, flexi->iov_pos );
    flexi->iov_length = flexi->iov_pos;

    return 1;
}

int32_t
opal_generate_iovec( opal_datatype_t *pData )
{
    opal_convertor_t *local_convertor = opal_convertor_create( opal_local_arch, 0 );

    size_t max = SIZE_MAX;
    int rc;

    opal_convertor_prepare_for_send( local_convertor, pData, 1, (void*)0 );

    pData->iovcnt = 512;
    uint32_t save_iov = 0;
    uint32_t leftover_iovec = 0;
    do {
        pData->iovcnt *= 2;
        pData->iov = realloc( pData->iov, sizeof(struct iovec) * pData->iovcnt );
        leftover_iovec = pData->iovcnt - save_iov;
        rc = opal_convertor_raw( local_convertor, pData->iov + save_iov, &leftover_iovec, &max );
        leftover_iovec += save_iov;  /* for the case we leave the loop */
        save_iov = pData->iovcnt;

    } while (0 == rc);

    pData->iov = realloc( pData->iov, sizeof(struct iovec) * leftover_iovec );
    pData->iovcnt = leftover_iovec;

    return 1;
}
