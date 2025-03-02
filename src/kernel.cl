#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

kernel void count_unique_nseqs(uint n, global const uint *txt, global uint *nseq_counts, global uint * unique_nseqs, global int *nseq_slot_locks) {
    int gid = get_global_id(0);
    int glob_work_size = get_global_size(0);

    bool done = false;

    // The last n-1 threads don't do any work
    if (gid > glob_work_size - n) {
        printf("Work size is %d, thread %d bailing out", glob_work_size, gid);
	done = true;
    }

    int i = 0;

    while(i < glob_work_size) {
	bool not_done_and_got_lock = !done && (atomic_xchg(&nseq_slot_locks[i], gid) == -1);

	if (not_done_and_got_lock) {
	  printf("%d has ze lock!!!!!!", gid);
	    if (unique_nseqs[n * i] == 0) {
		for (int j = 0; j < n; ++j) {
		    unique_nseqs[n * i + j] = txt[gid + j];
		}
		nseq_counts[i] = 1;
		done = true;
	    } else {

		char matches = 0;

		for (int j = 0; j < n; ++j) {

		    if (unique_nseqs[n * i + j] == txt[gid + j]) {
			matches += 1;
		    }
		}

		if (matches == n) {
		    nseq_counts[i] += 1;
		    done = true;
		}
	    }

	    atomic_xchg(&nseq_slot_locks[i], -1);
	    i += 1;
	} else {
	   printf("%d is a lockless bastard :(", gid);
	}
    }

    return;
}
