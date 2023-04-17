#include "ggml.h"
#include "rwkv.h"


void print_shape() {

}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    auto ctx = rwkv_init_from_file(argv[1], 1);
    // print_shape("token_index", ctx->token_index);
    // print_shape("state", ctx->state);
    // print_shape("logits", ctx->logits);
    // print_shape(ctx->state_parts[ctx->model->n_layer * 5]
    // ggml_graph_dump_dot(ctx->graph, NULL, "graph.dot");
}
