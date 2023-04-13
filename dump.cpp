#include "ggml.h"
#include "rwkv.h"


void print_shape() {

}

int main() {
    auto ctx = rwkv_init_from_file("RWKV-4-Pile-169M-20220807-8023-q4_1.bin", 1);
    // print_shape("token_index", ctx->token_index);
    // print_shape("state", ctx->state);
    // print_shape("logits", ctx->logits);
    // print_shape(ctx->state_parts[ctx->model->n_layer * 5]

    // ggml_graph_dump_dot(ctx->graph, NULL, "graph.dot");
}
