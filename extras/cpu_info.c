#include <stdio.h>

#include <rwkv.h>

int main(void) {
    printf("%s", rwkv_get_system_info_string());

    return 0;
}
