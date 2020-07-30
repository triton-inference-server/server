#include "grpc_client.h"

namespace nvic = nvidia::inferenceserver::client;

int
main(int argc, char* argv[])
{
  std::unique_ptr<nvic::InferenceServerGrpcClient> client;
  nvic::InferenceServerGrpcClient::Create(&client, "localhost:8001");
  bool live;
  client->IsServerLive(&live);

  if (live == 0) {
    return 0;
  }
  return 1;
}
