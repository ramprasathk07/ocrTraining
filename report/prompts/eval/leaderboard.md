# Prompt leaderboard

Newest first. Prompt texts are in registry.json.

## Prompt versions

| version | task | first seen | prompt |
|---|---|---|---|
| `p-8261c4fec7` | crop_ocr | 2026-11-19 | `<image>\nFree OCR.` |
| `p-00a105b332` | page_parse | 2026-11-19 | `<image>\n<\|grounding\|>Convert the document to markdown.` |

## Results

| date | run | split | checkpoint | source | prompt | n | metrics |
|---|---|---|---|---|---|---|---|
| 2026-11-19T16:40 | eval_test_data | test | start | printed_words | `p-8261c4fec7` | 200 | cer=922.42%, wer=575.50%, exact_match=0.00% |
| 2026-11-19T16:40 | eval_test_data | test | start | printed_words_1024 | `p-8261c4fec7` | 200 | cer=626.11%, wer=650.00%, exact_match=0.00% |
| 2026-11-19T16:40 | eval_test_data | test | start | golden_regions | `p-8261c4fec7` | 200 | cer=122.36%, wer=155.06%, exact_match=6.00% |
| 2026-11-19T16:40 | eval_test_data | test | start | golden_pages | `p-00a105b332` | 49 | page_cer=157.97%, layout_ap50=37.23%, mean_iou=41.21%, reading_order=98.08%, valid_output=77.55% |
