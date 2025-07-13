#!/usr/bin/env python3
"""
🚀 Example client for Apple Silicon LLM Pipeline
Demonstrates how to use the optimized inference pipeline in production
"""

import asyncio
import aiohttp
import websockets
import json
import time
import numpy as np
from typing import List, Dict

class LLMPipelineClient:
    """Client for Apple Silicon LLM Pipeline"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws/infer"
        
    async def infer_single(self, text: str, backend: str = "auto", model_size: str = "small") -> Dict:
        """Single inference request"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": text,
                "backend": backend,
                "model_size": model_size,
                "max_length": 512
            }
            
            async with session.post(f"{self.base_url}/infer", json=payload) as resp:
                return await resp.json()
    
    async def infer_batch(self, texts: List[str], backend: str = "auto") -> List[Dict]:
        """Batch inference requests"""
        tasks = []
        for text in texts:
            task = self.infer_single(text, backend)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def stream_infer(self, texts: List[str], backend: str = "auto"):
        """Streaming inference"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": texts,
                "backend": backend,
                "model_size": "small"
            }
            
            async with session.post(f"{self.base_url}/infer/stream", json=payload) as resp:
                async for line in resp.content:
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        yield data
    
    async def websocket_infer(self, texts: List[str], backend: str = "auto"):
        """Real-time WebSocket inference"""
        async with websockets.connect(self.ws_url) as websocket:
            for text in texts:
                # Send request
                request = {
                    "text": text,
                    "backend": backend,
                    "model_size": "small"
                }
                await websocket.send(json.dumps(request))
                
                # Receive response
                response = await websocket.recv()
                yield json.loads(response)
    
    async def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/stats") as resp:
                return await resp.json()
    
    async def health_check(self) -> Dict:
        """Check pipeline health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as resp:
                return await resp.json()

async def demo_basic_usage():
    """Demonstrate basic usage patterns"""
    print("🔍 Demo: Basic Usage Patterns")
    print("=" * 50)
    
    client = LLMPipelineClient()
    
    # Health check
    try:
        health = await client.health_check()
        print(f"✅ Pipeline Status: {health['status']}")
        print(f"📊 Backends Available: {health['backends_available']}")
    except:
        print("❌ Pipeline not running. Start with: python llm_inference_pipeline.py")
        return
    
    # Single inference
    print("\n🔥 Single Inference (ANE Optimized):")
    start_time = time.time()
    result = await client.infer_single(
        "Apple Silicon provides incredible performance for AI workloads.",
        backend="ane"
    )
    end_time = time.time()
    print(f"  Inference Time: {result['inference_time_ms']:.2f}ms")
    print(f"  Backend Used: {result['backend_used']}")
    print(f"  Tokens/Second: {result['tokens_per_second']:.1f}")
    print(f"  Total Time: {(end_time - start_time) * 1000:.2f}ms")
    
    # Batch inference
    print("\n📦 Batch Inference (Multiple texts):")
    texts = [
        "Neural Engine acceleration enables real-time AI processing.",
        "Metal Performance Shaders optimize GPU compute workloads.", 
        "CoreML framework provides seamless model deployment.",
        "Unified Memory Architecture reduces data transfer overhead."
    ]
    
    start_time = time.time()
    batch_results = await client.infer_batch(texts, backend="auto")
    end_time = time.time()
    
    total_time = (end_time - start_time) * 1000
    avg_latency = np.mean([r['inference_time_ms'] for r in batch_results])
    
    print(f"  Batch Size: {len(texts)}")
    print(f"  Total Time: {total_time:.2f}ms")
    print(f"  Average Latency: {avg_latency:.2f}ms")
    print(f"  Throughput: {len(texts) / (total_time / 1000):.1f} requests/sec")
    
    # Backend comparison
    print("\n⚡ Backend Comparison:")
    test_text = "Comparing Apple Silicon backend performance across ANE, Metal, and CPU."
    
    backends = ["ane", "metal", "cpu"]
    for backend in backends:
        try:
            result = await client.infer_single(test_text, backend=backend)
            print(f"  {backend.upper()}: {result['inference_time_ms']:.2f}ms ({result['tokens_per_second']:.1f} tok/s)")
        except Exception as e:
            print(f"  {backend.upper()}: Not available - {e}")

async def demo_streaming():
    """Demonstrate streaming inference"""
    print("\n🌊 Demo: Streaming Inference")
    print("=" * 50)
    
    client = LLMPipelineClient()
    
    texts = [
        "Real-time text processing with Apple Silicon optimization.",
        "Streaming inference enables immediate response generation.",
        "Low-latency processing perfect for interactive applications."
    ]
    
    print("📡 Streaming results:")
    async for result in client.stream_infer(texts, backend="ane"):
        print(f"  Batch {result['batch_index'] + 1}/{result['total_batches']}: "
              f"{result['inference_time_ms']:.2f}ms "
              f"({result['progress']*100:.1f}% complete)")

async def demo_websocket():
    """Demonstrate WebSocket real-time inference"""
    print("\n🔌 Demo: WebSocket Real-time Inference")
    print("=" * 50)
    
    client = LLMPipelineClient()
    
    texts = [
        "WebSocket enables real-time bidirectional communication.",
        "Perfect for interactive AI applications and chatbots.",
        "Ultra-low latency with Apple Neural Engine acceleration."
    ]
    
    print("⚡ Real-time results:")
    async for result in client.websocket_infer(texts, backend="ane"):
        print(f"  WebSocket Response: {result['inference_time_ms']:.2f}ms "
              f"using {result['backend_used']} backend")

async def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("\n📊 Demo: Performance Monitoring")
    print("=" * 50)
    
    client = LLMPipelineClient()
    
    # Run some requests to generate stats
    print("🔄 Generating workload...")
    for i in range(10):
        await client.infer_single(f"Performance test request #{i+1}", backend="ane")
    
    # Get stats
    stats = await client.get_stats()
    
    print(f"\n📈 Pipeline Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Average Latency: {stats['avg_latency']:.2f}ms")
    
    print(f"\n🎯 Backend Usage:")
    for backend, count in stats['backend_usage'].items():
        percentage = (count / stats['total_requests']) * 100 if stats['total_requests'] > 0 else 0
        print(f"  {backend.upper()}: {count} requests ({percentage:.1f}%)")
    
    print(f"\n⚡ Backend Performance:")
    for backend, perf in stats['backend_performance'].items():
        print(f"  {backend.upper()}: {perf['test_latency_ms']:.2f}ms ({perf['compute_units']})")

async def main():
    """Run all demos"""
    print("🍎 Apple Silicon LLM Pipeline Client Demo")
    print("=" * 60)
    print("🚀 This demo showcases production-ready LLM inference")
    print("   optimized for Apple Silicon hardware acceleration")
    print()
    
    # Check if pipeline is running
    client = LLMPipelineClient()
    try:
        await client.health_check()
    except:
        print("❌ Error: Pipeline server not running!")
        print("   Start the server first: python llm_inference_pipeline.py")
        return
    
    # Run demos
    await demo_basic_usage()
    await demo_streaming()
    await demo_websocket()
    await demo_performance_monitoring()
    
    print("\n🎉 Demo Complete!")
    print("\n💡 Production Integration Tips:")
    print("  • Use ANE backend for lowest latency (2-3ms)")
    print("  • Use Metal backend for high throughput batching")
    print("  • Implement auto-scaling based on request volume")
    print("  • Monitor backend utilization for optimal performance")
    print("  • Cache model outputs for repeated queries")
    print("  • Use streaming for long-running text processing")

if __name__ == "__main__":
    asyncio.run(main())