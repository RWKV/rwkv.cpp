/**
 * Show model info
*/

#include "ggml.h"
#include "rwkv.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    struct rwkv_init_opts opts;
    opts.n_threads = 1;
    opts.print_weights_info = true;
    rwkv_init_from_file_ex(argv[1], opts);
}
