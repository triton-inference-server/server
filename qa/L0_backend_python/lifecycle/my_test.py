import queue
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def grpc_strict_error():
    model_name = "execute_error"
    shape = [2, 2]
    number_of_requests = 3
    user_data = UserData()
    triton_server_url = "localhost:8001"  # Replace with your Triton server address

    try:
        triton_client = grpcclient.InferenceServerClient(triton_server_url)
        metadata = {"grpc_strict": "true"}

        triton_client.start_stream(
            callback=partial(callback, user_data), headers=metadata
        )

        input_datas = []
        for i in range(number_of_requests):
            input_data = np.random.randn(*shape).astype(np.float32)
            input_datas.append(input_data)
            inputs = [
                grpcclient.InferInput(
                    "IN", input_data.shape, np_to_triton_dtype(input_data.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data)
            triton_client.async_stream_infer(model_name=model_name, inputs=inputs)
            result = user_data._completed_requests.get()
            print(f"Request {i + 1} result:")
            print(type(result))
            if type(result) == InferenceServerException:
                print(result.status())

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        triton_client.stop_stream()


if __name__ == "__main__":
    grpc_strict_error()
