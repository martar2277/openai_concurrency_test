#!/usr/bin/env python3
"""
OpenAI API Concurrency Test
Tests sequential vs concurrent API requests to OpenAI using the same API key.
"""

import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Test configuration
NUM_REQUESTS = 10
MODEL = "gpt-3.5-turbo"  # Using gpt-3.5-turbo for faster/cheaper tests
MAX_TOKENS = 100

# Test prompts - 10 different prompts to ensure varied responses
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is 15 multiplied by 23?",
    "Name three programming languages.",
    "What is the largest planet in our solar system?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Name two primary colors.",
    "What year did World War II end?",
    "What is the chemical symbol for gold?"
]


def make_single_request(prompt_index):
    """
    Make a single request to OpenAI API.
    Returns: (prompt_index, response_text, duration_seconds)
    """
    prompt = TEST_PROMPTS[prompt_index]
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7
        )

        duration = time.time() - start_time
        response_text = response.choices[0].message.content.strip()

        return {
            "index": prompt_index,
            "prompt": prompt,
            "response": response_text,
            "duration": round(duration, 2),
            "success": True
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "index": prompt_index,
            "prompt": prompt,
            "response": None,
            "error": str(e),
            "duration": round(duration, 2),
            "success": False
        }


def test_sequential():
    """
    Test 1: Sequential requests (one after another)
    """
    print("=" * 80)
    print("TEST 1: SEQUENTIAL REQUESTS (One After Another)")
    print("=" * 80)
    print(f"Sending {NUM_REQUESTS} requests sequentially...\n")

    results = []
    overall_start = time.time()

    for i in range(NUM_REQUESTS):
        print(f"Request {i+1}/{NUM_REQUESTS}: Sending...", end=" ", flush=True)
        result = make_single_request(i)

        if result["success"]:
            print(f"✓ Completed in {result['duration']}s")
        else:
            print(f"✗ Failed in {result['duration']}s - {result['error']}")

        results.append(result)

    overall_duration = time.time() - overall_start

    # Print summary
    print("\n" + "-" * 80)
    print("SEQUENTIAL TEST SUMMARY")
    print("-" * 80)

    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]

    print(f"Total requests: {NUM_REQUESTS}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"\nTotal time: {round(overall_duration, 2)}s")

    if successful_requests:
        individual_times = [r["duration"] for r in successful_requests]
        print(f"Average individual response time: {round(sum(individual_times) / len(individual_times), 2)}s")
        print(f"Min response time: {round(min(individual_times), 2)}s")
        print(f"Max response time: {round(max(individual_times), 2)}s")

    return {
        "mode": "sequential",
        "total_duration": round(overall_duration, 2),
        "results": results,
        "successful": len(successful_requests),
        "failed": len(failed_requests)
    }


def test_concurrent():
    """
    Test 2: Concurrent requests (all at the same time using threading)
    """
    print("\n\n")
    print("=" * 80)
    print("TEST 2: CONCURRENT REQUESTS (All At The Same Time)")
    print("=" * 80)
    print(f"Sending {NUM_REQUESTS} requests concurrently...\n")

    results = []
    overall_start = time.time()

    # Use ThreadPoolExecutor to send all requests concurrently
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        # Submit all requests at once
        futures = {executor.submit(make_single_request, i): i for i in range(NUM_REQUESTS)}

        # Collect results as they complete
        for future in as_completed(futures):
            request_index = futures[future]
            result = future.result()
            results.append(result)

            if result["success"]:
                print(f"Request {result['index']+1} ✓ Completed in {result['duration']}s")
            else:
                print(f"Request {result['index']+1} ✗ Failed in {result['duration']}s - {result.get('error', 'Unknown error')}")

    overall_duration = time.time() - overall_start

    # Sort results by index for consistent reporting
    results.sort(key=lambda x: x["index"])

    # Print summary
    print("\n" + "-" * 80)
    print("CONCURRENT TEST SUMMARY")
    print("-" * 80)

    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]

    print(f"Total requests: {NUM_REQUESTS}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"\nTotal time: {round(overall_duration, 2)}s")

    if successful_requests:
        individual_times = [r["duration"] for r in successful_requests]
        print(f"Average individual response time: {round(sum(individual_times) / len(individual_times), 2)}s")
        print(f"Min response time: {round(min(individual_times), 2)}s")
        print(f"Max response time: {round(max(individual_times), 2)}s")

    return {
        "mode": "concurrent",
        "total_duration": round(overall_duration, 2),
        "results": results,
        "successful": len(successful_requests),
        "failed": len(failed_requests)
    }


def compare_results(sequential_data, concurrent_data):
    """
    Compare sequential vs concurrent results
    """
    print("\n\n")
    print("=" * 80)
    print("COMPARISON: SEQUENTIAL vs CONCURRENT")
    print("=" * 80)

    seq_time = sequential_data["total_duration"]
    conc_time = concurrent_data["total_duration"]
    time_saved = seq_time - conc_time
    speedup = seq_time / conc_time if conc_time > 0 else 0

    print(f"\nSequential total time:  {seq_time}s")
    print(f"Concurrent total time:  {conc_time}s")
    print(f"Time saved:             {round(time_saved, 2)}s")
    print(f"Speedup:                {round(speedup, 2)}x faster")
    print(f"\nEfficiency:             {round((time_saved / seq_time) * 100, 1)}% time reduction")

    print("\n" + "-" * 80)
    print("CONCLUSION")
    print("-" * 80)

    if speedup > 1.5:
        print(f"✓ Concurrent requests are SIGNIFICANTLY faster ({round(speedup, 2)}x speedup)")
        print("✓ One API key CAN handle multiple simultaneous requests")
        print("✓ The bottleneck is NOT the API key, but the server architecture")
    elif speedup > 1.1:
        print(f"✓ Concurrent requests are faster ({round(speedup, 2)}x speedup)")
        print("✓ Multiple simultaneous requests are possible with one API key")
    else:
        print("⚠ Results are similar - may be hitting rate limits or network constraints")

    return {
        "sequential_time": seq_time,
        "concurrent_time": conc_time,
        "time_saved": round(time_saved, 2),
        "speedup": round(speedup, 2)
    }


def save_results(sequential_data, concurrent_data, comparison_data):
    """
    Save detailed results to JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_requests": NUM_REQUESTS,
            "model": MODEL,
            "max_tokens": MAX_TOKENS
        },
        "sequential": sequential_data,
        "concurrent": concurrent_data,
        "comparison": comparison_data
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n📄 Detailed results saved to: {filename}")


def main():
    """
    Main test execution
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "OpenAI API Concurrency Test" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\nConfiguration:")
    print(f"  • Number of requests: {NUM_REQUESTS}")
    print(f"  • Model: {MODEL}")
    print(f"  • Max tokens per request: {MAX_TOKENS}")
    print(f"  • API Key: {'✓ Found' if os.getenv('OPENAI_API_KEY') else '✗ Not found'}")
    print()

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return

    # Run tests
    try:
        sequential_data = test_sequential()
        concurrent_data = test_concurrent()
        comparison_data = compare_results(sequential_data, concurrent_data)

        # Save results
        save_results(sequential_data, concurrent_data, comparison_data)

        print("\n✓ Test completed successfully!\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
