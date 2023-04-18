# ---

set(TEST_TARGET test_ggml_basics_work)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
