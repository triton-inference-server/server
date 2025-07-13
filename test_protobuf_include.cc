#include <google/protobuf/port_def.inc>

int main() {
    // Test if PROTOBUF_NAMESPACE_OPEN is defined
    #ifdef PROTOBUF_NAMESPACE_OPEN
        return 0;
    #else
        return 1;
    #endif
}