kernel void count_unique_nseqs(uint n, global const uint *txt, global uint *nseq_counts, global uint * unique_nseqs, global uint *nseq_slot_locks) {
    int gid = get_global_id(0);
    int glob_work_size = get_global_size(0);

    // The last n-1 threads don't do any work
    if (gid > glob_work_size - n) {
        return;
    }

    int nseq_idx = -1;
    int free_slot_idx = -1;

    // Look for an existing entry for this thread's n-sequence
    for (int i = 0; i < glob_work_size; ++i) {

        // Wait for a lock if claimed
        while (atomic_cmpxchg(&nseq_slot_locks[i], 0, 1) != 0) {}

        // We know we've reached the first unoccupied n-sequence slot if its first character is 0
        if (unique_nseqs[n * i] == 0) {
            free_slot_idx = i;

            // NOTE: We don't release the lock on purpose, we will
            // hold onto it until the free slot write stage
            break;
        }

        char matches = 0;

        for (int j = 0; j < n; ++j) {

            if (txt[gid + j] == unique_nseqs[n * i + j]) {
                matches += 1;
            } else {
                break;
            }
        }

        // release lock
        atomic_xchg(&nseq_slot_locks[i], 0);

        if (matches == n) {
            nseq_idx = i;
            break;
        }
    }

    // Skip the write into unique_nseqs if we found a slot occupied by our n-sequence
    if (nseq_idx != -1) {
        // No need to lock because of atomic addition
        atomic_add(&nseq_counts[nseq_idx], 1);
        return;
    }

    // Alternatively, a free slot should have been discovered. Write
    // n-sequence chars to it.
    if (free_slot_idx != -1) {
        // Write this thread's n-sequence chars into unique_nseqs
        for (int j = 0; j < n; ++j) {
            unique_nseqs[n * free_slot_idx + j] = txt[gid + j];
        }

        // Set initial value to 1
        nseq_counts[free_slot_idx] = 1;

        // Release lock
        atomic_xchg(&nseq_slot_locks[free_slot_idx], 0);

        return;
    }

    // This section should not be reachable. Indicate a problem with a
    // magic value in the unused last element
    nseq_counts[glob_work_size - 1] = 0xdeadbeef;

    if (free_slot_idx != -1) {
        // Release the lock so that a thread reaching here does not
        // block other threads forever which makes deadbeef useless.
        atomic_xchg(&nseq_slot_locks[free_slot_idx], 0);
    }

    return;
}
