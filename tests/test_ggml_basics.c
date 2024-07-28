// Tests that ggml basics work.
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <ggml.h>

#include "assertions.inc"

#define SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

void test_simple_computation(void) {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(x, 0, -10.0F);
    SET_ELEMENT_F32(x, 1, 0.0F);
    SET_ELEMENT_F32(x, 2, 2.5F);
    SET_ELEMENT_F32(x, 3, 5.0F);

    struct ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(y, 0, 1.0F);
    SET_ELEMENT_F32(y, 1, 2.0F);
    SET_ELEMENT_F32(y, 2, 3.0F);
    SET_ELEMENT_F32(y, 3, 4.0F);

    struct ggml_tensor * sum = ggml_add(ctx, x, y);

    // Allocation on heap instead of stack avoids SegFault when GGML_MAX_NODES is set to a large value.
    struct ggml_cgraph * graph = (struct ggml_cgraph *) calloc(1, sizeof(struct ggml_cgraph));
    graph->size = GGML_DEFAULT_GRAPH_SIZE;
    graph->n_nodes = 0;
    graph->n_leafs = 0;
    graph->nodes = (struct ggml_tensor **) calloc(1, GGML_DEFAULT_GRAPH_SIZE * sizeof(struct ggml_tensor *));
    graph->leafs = (struct ggml_tensor **) calloc(1, GGML_DEFAULT_GRAPH_SIZE * sizeof(struct ggml_tensor *));
    size_t hash_size = GGML_DEFAULT_GRAPH_SIZE * 2 + 1;
    graph->visited_hash_table.size = hash_size;
    graph->visited_hash_table.keys = (struct ggml_tensor **) calloc(1, hash_size * sizeof(struct ggml_tensor *));
    graph->order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;

    ggml_build_forward_expand(graph, sum);
    struct ggml_cplan plan = ggml_graph_plan(graph, 2);
    ggml_graph_compute(graph, &plan);

    free(graph->nodes);
    free(graph->leafs);
    free(graph->visited_hash_table.keys);
    free(graph);

    ASSERT_ELEMENT_F32(sum, 0, -9.0F);
    ASSERT_ELEMENT_F32(sum, 1, 2.0F);
    ASSERT_ELEMENT_F32(sum, 2, 5.5F);
    ASSERT_ELEMENT_F32(sum, 3, 9.0F);

    ggml_print_objects(ctx);

    ggml_free(ctx);
}

// RWKV model loading code depends on this behavior.
void test_computation_on_tensors_from_different_contexts(void) {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_context * ctx1 = ggml_init(params);
    struct ggml_context * ctx2 = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(x, 0, -10.0F);
    SET_ELEMENT_F32(x, 1, 0.0F);

    struct ggml_tensor * y = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(y, 0, 1.0F);
    SET_ELEMENT_F32(y, 1, 2.0F);

    struct ggml_tensor * sum = ggml_add(ctx2, x, y);

    // Allocation on heap instead of stack avoids SegFault when GGML_MAX_NODES is set to a large value.
    struct ggml_cgraph * graph = (struct ggml_cgraph *) calloc(1, sizeof(struct ggml_cgraph));
    graph->size = GGML_DEFAULT_GRAPH_SIZE;
    graph->n_nodes = 0;
    graph->n_leafs = 0;
    graph->nodes = (struct ggml_tensor **) calloc(1, GGML_DEFAULT_GRAPH_SIZE * sizeof(struct ggml_tensor *));
    graph->leafs = (struct ggml_tensor **) calloc(1, GGML_DEFAULT_GRAPH_SIZE * sizeof(struct ggml_tensor *));
    size_t hash_size = GGML_DEFAULT_GRAPH_SIZE * 2 + 1;
    graph->visited_hash_table.size = hash_size;
    graph->visited_hash_table.keys = (struct ggml_tensor **) calloc(1, hash_size * sizeof(struct ggml_tensor *));
    graph->order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;
    ggml_build_forward_expand(graph, sum);
    struct ggml_cplan plan = ggml_graph_plan(graph, 2);
    ggml_graph_compute(graph, &plan);

    free(graph->nodes);
    free(graph->leafs);
    free(graph->visited_hash_table.keys);
    free(graph);

    ASSERT_ELEMENT_F32(sum, 0, -9.0F);
    ASSERT_ELEMENT_F32(sum, 1, 2.0F);

    ggml_free(ctx0);
    ggml_free(ctx1);
    ggml_free(ctx2);
}

int main(void) {
    test_simple_computation();

    test_computation_on_tensors_from_different_contexts();

    return 0;
}
