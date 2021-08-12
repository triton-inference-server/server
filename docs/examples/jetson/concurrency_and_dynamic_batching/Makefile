TARGET=people_detection
GCC=g++
GCC_PARMS+=-I../../server -I/usr/include/opencv4 -I../../core/include/ -I/usr/local/cuda/targets/aarch64-linux/include
GCC_PARMS+=-I/opt/tritonserver/include -D TRITON_ENABLE_GPU -D TRITON_MIN_COMPUTE_CAPABILITY=5.3

GCC_LIBS=-L/opt/tritonserver/lib -L/usr/lib -L/usr/local/cuda/targets/aarch64-linux/lib
GCC_LIBS+=-lpthread -ltritonserver -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn -lcudart

all: $(TARGET) 


%.o: %.cc
	$(GCC) $(GCC_PARMS) -c -g -o $@ $^

$(TARGET): $(TARGET).o
	$(GCC) $^ $(GCC_LIBS) -o $@ 

clean:
	rm -f $(TARGET).o $(TARGET)

.PHONY: all clean
