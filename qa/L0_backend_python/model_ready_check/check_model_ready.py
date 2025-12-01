import unittest
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

class ModelReadyTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "identity_fp32"
        self.url_http = "localhost:8000"
        self.url_grpc = "localhost:8001"
        self.client_http = httpclient.InferenceServerClient(url=self.url_http)
        self.client_grpc = grpcclient.InferenceServerClient(url=self.url_grpc)

    def test_model_ready(self):
        print(f"\nTesting if model '{self.model_name}' is READY ...")
        
        # Check HTTP
        try:
            is_ready = self.client_http.is_model_ready(self.model_name)
            self.assertTrue(is_ready, f"[HTTP] Model {self.model_name} should be READY but is NOT")
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertTrue(is_ready, f"[gRPC] Model {self.model_name} should be READY but is NOT")
        except Exception as e:
            self.fail(f"[gRPC] Unexpected error: {str(e)}")

    def test_model_not_ready(self):
        print(f"\nTesting if model '{self.model_name}' is NOT READY ...")
        
        # Check HTTP
        try:
            is_ready = self.client_http.is_model_ready(self.model_name)
            self.assertFalse(is_ready, f"[HTTP] Model {self.model_name} should be NOT READY but is READY.")
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertFalse(is_ready, f"[gRPC] Model {self.model_name} should be NOT READY but is READY")
        except Exception as e:
            self.fail(f"[gRPC] Unexpected error: {str(e)}")

if __name__ == "__main__":
    unittest.main()
