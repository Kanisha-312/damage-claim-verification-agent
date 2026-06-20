# Evaluation Report

## System Overview

The system processes each damage claim with a single Claude Sonnet 4.6 Vision API call. Each call receives:
- All submitted images (base64-encoded, JPEG/PNG/WebP detected via magic bytes)
- The claim conversation
- User history context (past claims, rejection rate, history flags)
- Evidence requirements filtered to the claim object type
- A structured prompt requesting a JSON response with all required output fields

Prompt injection attempts in the claim conversation or image text (e.g. "approve immediately", "skip review") are explicitly flagged as `text_instruction_present` and ignored.

---

## Sample Dataset Evaluation

Ran on 20 sample claims from `dataset/sample_claims.csv` with known expected outputs.

Evaluation compares predictions to ground truth on six fields:

| Field | Notes |
|---|---|
| `claim_status` | 3-class exact match (supported / contradicted / not_enough_information) |
| `evidence_standard_met` | Boolean exact match |
| `issue_type` | Exact match against allowed list |
| `object_part` | Exact match against allowed list per object type |
| `severity` | 5-level exact match (none / low / medium / high / unknown) |
| `valid_image` | Boolean exact match |
| `risk_flags` | Jaccard similarity of flag sets |

Run `python code/evaluation/main.py` for live results. Use `--skip-inference` to reuse an existing `sample_output.csv`.

---

## Operational Analysis

### Model calls

| Run | Claims | Images | API calls |
|---|---|---|---|
| Sample evaluation | 20 | 29 | 20 |
| Test set (claims.csv) | 44 | 82 | 44 |
| **Total** | **64** | **111** | **64** |

One call per claim regardless of image count. All images for a claim are sent in a single multi-part message.

### Token usage (estimated)

Pricing: claude-sonnet-4-6 — $3.00 / MTok input, $15.00 / MTok output.

Assumptions: ~800 text tokens per call (prompt + conversation + history + requirements), ~800 tokens per image (typical damage photo at standard resolution), ~400 output tokens per call (JSON response).

| Run | Input tokens | Output tokens | Cost |
|---|---|---|---|
| Sample (20 calls, 29 images) | ~39,200 | ~8,000 | ~$0.24 |
| Test (44 calls, 82 images) | ~100,800 | ~17,600 | ~$0.57 |
| **Total** | **~140,000** | **~25,600** | **~$0.81** |

Image token counts vary with image dimensions. Smaller images reduce cost; large high-resolution photos increase it.

### Images processed

- Sample set: 29 images across 20 claims
- Test set: 82 images across 44 claims
- Total: 111 images

### Latency

Each API call takes roughly 3–8 seconds depending on image count and size. Full sequential runs:

| Run | Estimated time |
|---|---|
| Sample (20 claims) | ~1–2 minutes |
| Test (44 claims) | ~3–6 minutes |

### Rate limits and batching strategy

- claude-sonnet-4-6 default limits: ~50 RPM, ~100K TPM (varies by tier)
- The system processes claims sequentially with no explicit throttle
- At ~10 RPM actual rate the system stays well within RPM limits
- TPM headroom is ~60–90K per minute, which comfortably fits burst processing
- No retry loop implemented; API errors are caught and written as fallback rows so the run completes

### Error handling and fallback

If an API call raises any exception (auth error, timeout, 5xx, JSON parse failure), the claim is written with:
- `claim_status = not_enough_information`
- `evidence_standard_met = false`
- `risk_flags = manual_review_required`
- `valid_image = false`
- `severity = unknown`

This ensures `output.csv` always has exactly one row per input claim.

### Image type detection

Images are decoded using file magic bytes rather than file extension:
- `\xff\xd8\xff` → `image/jpeg`
- `\x89PNG\r\n\x1a\n` → `image/png`
- `RIFF....WEBP` → `image/webp`

One test image (`case_044/img_2`) has a `.jpg` extension but is actually a PNG. The magic bytes approach sends the correct `Content-Type` to the API regardless of how the file is named.

### Caching and repeated calls

No caching is implemented. Running the script twice makes 2× the API calls. For iterative development on the sample set, `--skip-inference` reuses `sample_output.csv` from a prior run without making additional API calls.

### Potential improvements

- **Response caching**: hash the image bytes + prompt and cache JSON responses to avoid re-calling identical inputs
- **Model tier**: use `claude-haiku-4-5` for simpler claims (package damage, clear contradictions) to reduce cost by ~5×
- **Parallel calls**: `asyncio` + `asyncio.gather` could process all claims concurrently, reducing wall-clock time from minutes to ~15 seconds, at the cost of higher instantaneous TPM
- **Retry with backoff**: add exponential retry for transient 5xx and 429 errors instead of immediately writing a fallback row
