// Find the most common n-sequence in an array of strings
//
// ASSUMPTIONS:
// n - n < txt_size
// txts - size of txt_size * get_global_size(0), strings aligned to txt_size, padded with 0s
// txt_size - corresponds with the longest string length
// nseq_counts - size of get_global_size(0) * txt_size, zeroed out 
// unique_nseqs - size of get_global_size(0) * n * txt_size, zeroed out
// most_common_nseqs - size of get_global_size(0) * n, zeroed out
kernel void most_common_nseq(const uint n, global const uint *txts, const uint n_txts, const uint txt_size, global uint *nseq_counts, global uint * unique_nseqs, global uint *most_common_nseqs) {
    int gid = get_global_id(0);
    global const uint *my_txt = &txts[gid * txt_size];
    global uint *my_nseq_counts = &nseq_counts[gid * txt_size];
    global uint *my_unique_nseqs = &unique_nseqs[gid * txt_size];
    global uint *my_most_common_nseq = &most_common_nseqs[gid * n];

    if (gid >= n_txts) {
	return;
    }

    // Go through each n-sequence in my_txt
    for (uint txt_idx = 0; txt_idx < txt_size - n + 1; ++txt_idx) {

        // We've reached the padding, bail out
        if (my_txt[txt_idx + n - 1] == 0) {
            break;
        }

        global uint *txt_nseq = &my_txt[txt_idx];

        // Go through each n-sequence in my_unique_nseqs
        for (uint nseqs_idx = 0; nseqs_idx < txt_size; ++nseqs_idx) {

            global uint *unique_nseq = &my_unique_nseqs[n * nseqs_idx];

            bool slot_found = true; // Tracks whether the third loop found a my_unique_nseqs slot and wrote to it

            // Go through each individual char for n-sequence comparison
            for (uint nseq_char_idx = 0; nseq_char_idx < n; ++nseq_char_idx) {

                // Check if this my_unique_nseqs char is wrong for this n-sequence
                if (unique_nseq[nseq_char_idx] != txt_nseq[nseq_char_idx] ) {
                    if (unique_nseq[nseq_char_idx] == 0) {
                        // We're at the end of known n-sequences, populate this one
                        unique_nseq[nseq_char_idx] = txt_nseq[nseq_char_idx];
                    } else {
                        slot_found = false;
                        break;
                    }
                }
            }

            if (slot_found) {
                my_nseq_counts[nseqs_idx] += 1;

                // The relevant unique_nseqs entry was either found or
                // created, move on to next n-sequence in my_txt
		break;
            }
        }

    }

    // my_most_common_nseq == NULL means we haven't found an n-sequence that repeats
    global uint *most_common_unique_nseq = 0;

    // Only more than 1 appearance qualifies as non-trivially most common
    uint max_val = 1;

    for (uint nseqs_idx = 0; nseqs_idx < txt_size; ++nseqs_idx) {
        global uint *unique_nseq = &my_unique_nseqs[n * nseqs_idx];

        if (unique_nseq[0] == 0) {
            // Bail out if we've reached the last defined unique n-sequence
            break;
        }

        if (my_nseq_counts[nseqs_idx] > max_val) {
            max_val = my_nseq_counts[nseqs_idx];
            most_common_unique_nseq = unique_nseq;
        }
    }

    if (most_common_unique_nseq != 0) {
        for (uint nseq_char_idx = 0; nseq_char_idx < n; ++nseq_char_idx) {
            my_most_common_nseq[nseq_char_idx] = most_common_unique_nseq[nseq_char_idx];
        }
    }

    return;
}
