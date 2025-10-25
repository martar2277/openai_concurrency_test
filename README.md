# OpenAI API Concurrency Test

This test demonstrates whether one OpenAI API key can handle multiple simultaneous requests or if requests must be processed sequentially.

## What This Test Does

The script runs two tests with the **same API key**:

1. **Sequential Test**: Sends 10 requests one after another
   - Request 1 → wait for response → Request 2 → wait → Request 3 → etc.

2. **Concurrent Test**: Sends 10 requests simultaneously using threading
   - All 10 requests sent at the same time
   - All processed in parallel

## Expected Results

If our hypothesis is correct:
- **Sequential**: ~30 seconds total (10 requests × 3s each)
- **Concurrent**: ~3 seconds total (all 10 requests processed simultaneously)
- **Speedup**: ~10x faster with concurrent requests

This proves that one API key does NOT enforce sequential processing.

## Setup

### Prerequisites
- Python 3.7+
- OpenAI API key

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key (if not already set)
export OPENAI_API_KEY='your-api-key-here'
```

### In GitHub Codespaces

If your API key is already stored in Codespace secrets:

```bash
# Just install and run
pip install -r requirements.txt
python test_concurrent_api.py
```

## Usage

```bash
python test_concurrent_api.py
```

## What You'll See

The script will:

1. Display configuration (number of requests, model, etc.)
2. Run sequential test with live progress
3. Run concurrent test with live progress
4. Show detailed comparison
5. Save results to a JSON file with timestamp

### Sample Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OpenAI API Concurrency Test                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Configuration:
  • Number of requests: 10
  • Model: gpt-3.5-turbo
  • Max tokens per request: 100
  • API Key: ✓ Found

================================================================================
TEST 1: SEQUENTIAL REQUESTS (One After Another)
================================================================================
Sending 10 requests sequentially...

Request 1/10: Sending... ✓ Completed in 2.34s
Request 2/10: Sending... ✓ Completed in 2.51s
...

--------------------------------------------------------------------------------
SEQUENTIAL TEST SUMMARY
--------------------------------------------------------------------------------
Total requests: 10
Successful: 10
Failed: 0

Total time: 24.56s
Average individual response time: 2.46s

================================================================================
TEST 2: CONCURRENT REQUESTS (All At The Same Time)
================================================================================
Sending 10 requests concurrently...

Request 1 ✓ Completed in 2.45s
Request 3 ✓ Completed in 2.52s
Request 2 ✓ Completed in 2.58s
...

--------------------------------------------------------------------------------
CONCURRENT TEST SUMMARY
--------------------------------------------------------------------------------
Total requests: 10
Successful: 10
Failed: 0

Total time: 2.98s
Average individual response time: 2.51s

================================================================================
COMPARISON: SEQUENTIAL vs CONCURRENT
================================================================================

Sequential total time:  24.56s
Concurrent total time:  2.98s
Time saved:             21.58s
Speedup:                8.24x faster

Efficiency:             87.9% time reduction

--------------------------------------------------------------------------------
CONCLUSION
--------------------------------------------------------------------------------
✓ Concurrent requests are SIGNIFICANTLY faster (8.24x speedup)
✓ One API key CAN handle multiple simultaneous requests
✓ The bottleneck is NOT the API key, but the server architecture

📄 Detailed results saved to: test_results_20250125_143022.json
```

## Output Files

Each test run creates a JSON file with detailed results:

- Timestamp of test
- Configuration used
- Individual request timings
- Complete responses
- Comparison metrics

Example: `test_results_20250125_143022.json`

## Configuration

Edit these constants in `test_concurrent_api.py` to customize:

```python
NUM_REQUESTS = 10              # Number of requests to send
MODEL = "gpt-3.5-turbo"        # OpenAI model to use
MAX_TOKENS = 100               # Max tokens per response
```

## Notes

- Uses `gpt-3.5-turbo` by default (faster and cheaper than GPT-4)
- Each request uses a different prompt to ensure varied responses
- ThreadPoolExecutor is used for concurrent requests (not async/await)
- Results are saved with timestamps for comparison across runs

## Troubleshooting

### "OPENAI_API_KEY not found"
```bash
export OPENAI_API_KEY='sk-...'
```

### Rate limit errors
If you hit OpenAI rate limits, you'll see error messages in the output. This actually proves the point - you can send concurrent requests until you hit the rate limit, not just one at a time.

### Import errors
```bash
pip install --upgrade openai
```

## Purpose

This test script proves that:
1. One API key can handle multiple simultaneous requests
2. Concurrent requests are significantly faster than sequential
3. The bottleneck in web applications is the server architecture (single-threaded Flask), not the OpenAI API key

## License

This is a test script for educational purposes.
