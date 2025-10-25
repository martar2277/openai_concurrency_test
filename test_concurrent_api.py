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
    error_type = None
    error_code = None

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
            "success": True,
            "error_type": None,
            "error_code": None
        }

    except Exception as e:
        duration = time.time() - start_time

        # Classify error types
        error_str = str(e)
        if "rate_limit" in error_str.lower() or "429" in error_str:
            error_type = "RATE_LIMIT"
            error_code = 429
        elif "timeout" in error_str.lower():
            error_type = "TIMEOUT"
        elif "503" in error_str or "service unavailable" in error_str.lower():
            error_type = "SERVICE_UNAVAILABLE"
            error_code = 503
        elif "500" in error_str or "internal server" in error_str.lower():
            error_type = "SERVER_ERROR"
            error_code = 500
        elif "401" in error_str or "unauthorized" in error_str.lower():
            error_type = "AUTH_ERROR"
            error_code = 401
        elif "connection" in error_str.lower():
            error_type = "CONNECTION_ERROR"
        else:
            error_type = "OTHER"

        return {
            "index": prompt_index,
            "prompt": prompt,
            "response": None,
            "error": error_str,
            "error_type": error_type,
            "error_code": error_code,
            "duration": round(duration, 2),
            "success": False
        }


def analyze_errors(results):
    """
    Analyze and categorize errors from results
    """
    failed_requests = [r for r in results if not r["success"]]

    if not failed_requests:
        return None

    error_breakdown = {}
    for result in failed_requests:
        error_type = result.get("error_type", "UNKNOWN")
        if error_type not in error_breakdown:
            error_breakdown[error_type] = {
                "count": 0,
                "examples": [],
                "avg_duration": []
            }
        error_breakdown[error_type]["count"] += 1
        error_breakdown[error_type]["avg_duration"].append(result["duration"])
        if len(error_breakdown[error_type]["examples"]) < 2:  # Store up to 2 examples
            error_breakdown[error_type]["examples"].append({
                "index": result["index"],
                "error": result.get("error", "Unknown"),
                "duration": result["duration"]
            })

    # Calculate average durations
    for error_type in error_breakdown:
        durations = error_breakdown[error_type]["avg_duration"]
        error_breakdown[error_type]["avg_duration"] = round(sum(durations) / len(durations), 2)

    return error_breakdown


def print_error_analysis(error_breakdown):
    """
    Print detailed error analysis
    """
    if not error_breakdown:
        return

    print("\n" + "!" * 80)
    print("ERROR ANALYSIS")
    print("!" * 80)

    for error_type, data in error_breakdown.items():
        print(f"\n{error_type}: {data['count']} occurrence(s)")
        print(f"  Average duration: {data['avg_duration']}s")
        print(f"  Examples:")
        for ex in data["examples"]:
            print(f"    - Request {ex['index'] + 1}: {ex['error'][:100]}...")


def print_performance_warnings(results, total_duration):
    """
    Print warnings about performance degradation
    """
    successful_requests = [r for r in results if r["success"]]

    if not successful_requests:
        return

    individual_times = [r["duration"] for r in successful_requests]
    avg_time = sum(individual_times) / len(individual_times)
    max_time = max(individual_times)

    warnings = []

    if avg_time > 10:
        warnings.append(f"⚠ CRITICAL: Average response time is {round(avg_time, 2)}s (> 10s - users will likely abandon)")
    elif avg_time > 5:
        warnings.append(f"⚠ WARNING: Average response time is {round(avg_time, 2)}s (> 5s - noticeable delay)")
    elif avg_time > 2:
        warnings.append(f"⚠ NOTE: Average response time is {round(avg_time, 2)}s (> 2s - acceptable but not instant)")

    if max_time > 15:
        warnings.append(f"⚠ CRITICAL: Maximum response time was {round(max_time, 2)}s (> 15s - extremely slow)")
    elif max_time > 10:
        warnings.append(f"⚠ WARNING: Maximum response time was {round(max_time, 2)}s (> 10s - very slow)")

    # Check for variance
    if len(individual_times) > 1:
        variance = max(individual_times) - min(individual_times)
        if variance > 5:
            warnings.append(f"⚠ High variance in response times: {round(variance, 2)}s difference between fastest and slowest")

    if warnings:
        print("\n" + "!" * 80)
        print("PERFORMANCE WARNINGS")
        print("!" * 80)
        for warning in warnings:
            print(warning)


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
            error_type = result.get("error_type", "UNKNOWN")
            print(f"✗ Failed ({error_type}) in {result['duration']}s")

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

    # Error analysis
    error_breakdown = analyze_errors(results)
    if error_breakdown:
        print_error_analysis(error_breakdown)

    # Performance warnings
    print_performance_warnings(results, overall_duration)

    return {
        "mode": "sequential",
        "total_duration": round(overall_duration, 2),
        "results": results,
        "successful": len(successful_requests),
        "failed": len(failed_requests),
        "error_breakdown": error_breakdown
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
    completion_times = []  # Track when each request completes

    # Use ThreadPoolExecutor to send all requests concurrently
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        # Submit all requests at once
        futures = {executor.submit(make_single_request, i): i for i in range(NUM_REQUESTS)}

        # Collect results as they complete
        for future in as_completed(futures):
            request_index = futures[future]
            result = future.result()
            completion_time = time.time() - overall_start
            completion_times.append(completion_time)
            results.append(result)

            if result["success"]:
                print(f"Request {result['index']+1} ✓ Completed in {result['duration']}s (at T+{round(completion_time, 2)}s)")
            else:
                error_type = result.get("error_type", "UNKNOWN")
                print(f"Request {result['index']+1} ✗ Failed ({error_type}) in {result['duration']}s (at T+{round(completion_time, 2)}s)")

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

    # Concurrency metrics
    if len(completion_times) > 1:
        first_completion = min(completion_times)
        last_completion = max(completion_times)
        print(f"\nConcurrency metrics:")
        print(f"  First request completed: {round(first_completion, 2)}s")
        print(f"  Last request completed: {round(last_completion, 2)}s")
        print(f"  Completion window: {round(last_completion - first_completion, 2)}s")

    # Error analysis
    error_breakdown = analyze_errors(results)
    if error_breakdown:
        print_error_analysis(error_breakdown)

    # Performance warnings
    print_performance_warnings(results, overall_duration)

    return {
        "mode": "concurrent",
        "total_duration": round(overall_duration, 2),
        "results": results,
        "successful": len(successful_requests),
        "failed": len(failed_requests),
        "error_breakdown": error_breakdown,
        "completion_times": completion_times
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


def save_responses_to_text(sequential_data, concurrent_data, timestamp):
    """
    Save responses to separate text files for manual examination
    """
    # Save sequential responses
    seq_filename = f"responses_sequential_{timestamp}.txt"
    with open(seq_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SEQUENTIAL RESPONSES\n")
        f.write("=" * 80 + "\n\n")

        for result in sequential_data["results"]:
            if result["success"]:
                f.write(f"Request {result['index'] + 1}:\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Response: {result['response']}\n")
                f.write(f"Duration: {result['duration']}s\n")
                f.write("-" * 80 + "\n\n")
            else:
                f.write(f"Request {result['index'] + 1}: FAILED\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                f.write("-" * 80 + "\n\n")

    # Save concurrent responses
    conc_filename = f"responses_concurrent_{timestamp}.txt"
    with open(conc_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CONCURRENT RESPONSES\n")
        f.write("=" * 80 + "\n\n")

        for result in concurrent_data["results"]:
            if result["success"]:
                f.write(f"Request {result['index'] + 1}:\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Response: {result['response']}\n")
                f.write(f"Duration: {result['duration']}s\n")
                f.write("-" * 80 + "\n\n")
            else:
                f.write(f"Request {result['index'] + 1}: FAILED\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                f.write("-" * 80 + "\n\n")

    return seq_filename, conc_filename


def save_diagnostics(sequential_data, concurrent_data, comparison_data, timestamp):
    """
    Save diagnostic report with error analysis and performance metrics
    """
    diag_filename = f"diagnostics_{timestamp}.txt"

    with open(diag_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Test Date: {datetime.now().isoformat()}\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Requests: {NUM_REQUESTS}\n")
        f.write(f"  - Model: {MODEL}\n")
        f.write(f"  - Max Tokens: {MAX_TOKENS}\n\n")

        # Sequential diagnostics
        f.write("-" * 80 + "\n")
        f.write("SEQUENTIAL TEST DIAGNOSTICS\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Success Rate: {sequential_data['successful']}/{NUM_REQUESTS} ")
        f.write(f"({round(sequential_data['successful']/NUM_REQUESTS*100, 1)}%)\n")
        f.write(f"Total Duration: {sequential_data['total_duration']}s\n\n")

        if sequential_data.get('error_breakdown'):
            f.write("Errors:\n")
            for error_type, data in sequential_data['error_breakdown'].items():
                f.write(f"  - {error_type}: {data['count']} occurrence(s)\n")
                f.write(f"    Avg duration: {data['avg_duration']}s\n")
                for ex in data['examples']:
                    f.write(f"    Example: {ex['error'][:150]}\n")
                f.write("\n")
        else:
            f.write("No errors detected.\n\n")

        # Concurrent diagnostics
        f.write("-" * 80 + "\n")
        f.write("CONCURRENT TEST DIAGNOSTICS\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Success Rate: {concurrent_data['successful']}/{NUM_REQUESTS} ")
        f.write(f"({round(concurrent_data['successful']/NUM_REQUESTS*100, 1)}%)\n")
        f.write(f"Total Duration: {concurrent_data['total_duration']}s\n\n")

        if concurrent_data.get('error_breakdown'):
            f.write("Errors:\n")
            for error_type, data in concurrent_data['error_breakdown'].items():
                f.write(f"  - {error_type}: {data['count']} occurrence(s)\n")
                f.write(f"    Avg duration: {data['avg_duration']}s\n")
                for ex in data['examples']:
                    f.write(f"    Example: {ex['error'][:150]}\n")
                f.write("\n")
        else:
            f.write("No errors detected.\n\n")

        # Performance analysis
        f.write("-" * 80 + "\n")
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        seq_successful = [r for r in sequential_data['results'] if r['success']]
        conc_successful = [r for r in concurrent_data['results'] if r['success']]

        if seq_successful:
            seq_times = [r['duration'] for r in seq_successful]
            f.write(f"Sequential avg response time: {round(sum(seq_times)/len(seq_times), 2)}s\n")
            f.write(f"Sequential max response time: {round(max(seq_times), 2)}s\n\n")

        if conc_successful:
            conc_times = [r['duration'] for r in conc_successful]
            f.write(f"Concurrent avg response time: {round(sum(conc_times)/len(conc_times), 2)}s\n")
            f.write(f"Concurrent max response time: {round(max(conc_times), 2)}s\n\n")

        f.write(f"Speedup: {comparison_data['speedup']}x\n")
        f.write(f"Time saved: {comparison_data['time_saved']}s\n\n")

        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")

        if conc_successful:
            avg_conc_time = sum([r['duration'] for r in conc_successful]) / len(conc_successful)
            if avg_conc_time < 2:
                f.write("✓ EXCELLENT: Response times are under 2 seconds - ideal for interactive use\n")
            elif avg_conc_time < 5:
                f.write("✓ GOOD: Response times are acceptable for most users\n")
            elif avg_conc_time < 10:
                f.write("⚠ MODERATE: Response times may cause user frustration\n")
            else:
                f.write("✗ POOR: Response times are too slow for interactive dialogue\n")

        if concurrent_data.get('error_breakdown'):
            rate_limit_count = concurrent_data['error_breakdown'].get('RATE_LIMIT', {}).get('count', 0)
            if rate_limit_count > 0:
                f.write(f"\n⚠ Rate limits detected ({rate_limit_count} errors)\n")
                f.write(f"  Consider reducing concurrency or upgrading API tier\n")
                f.write(f"  Current load: {NUM_REQUESTS} concurrent requests\n")

        f.write("\n")

    return diag_filename


def save_results(sequential_data, concurrent_data, comparison_data):
    """
    Save detailed results to JSON file and responses to text files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_filename = f"test_results_{timestamp}.json"
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

    with open(json_filename, 'w') as f:
        json.dump(output, f, indent=2)

    # Save text responses for manual examination
    seq_file, conc_file = save_responses_to_text(sequential_data, concurrent_data, timestamp)

    # Save diagnostics report
    diag_file = save_diagnostics(sequential_data, concurrent_data, comparison_data, timestamp)

    print(f"\n📄 Detailed results saved to: {json_filename}")
    print(f"📄 Sequential responses saved to: {seq_file}")
    print(f"📄 Concurrent responses saved to: {conc_file}")
    print(f"📄 Diagnostic report saved to: {diag_file}")


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
