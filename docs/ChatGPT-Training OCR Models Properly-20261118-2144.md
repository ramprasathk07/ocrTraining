# Training OCR Models Properly

**User:** Anonymous  
**Created:** 11/14/2026 14:27:35  
**Updated:** 11/18/2026 19:27:26  
**Exported:** 11/18/2026 21:44:00  
**Link:** [https://chatgpt.com/g/g-p-6afb1ab9497c8191acbdc3a36a28a450-ocr/c/6af8227f-11ec-83ec-bf83-c015cd5c5457](https://chatgpt.com/g/g-p-6afb1ab9497c8191acbdc3a36a28a450-ocr/c/6af8227f-11ec-83ec-bf83-c015cd5c5457)  

## Prompt:
11/14/2026, 2:27:35 PM

how to train an new ocr model from stratch should i train with image and jsons like added below or MD file? how large scale ocr models are trained exactly like liquidfm models,deepseek ocr or unlimited-OCR. 

{
  "source": "F:\\gitRepos\\LiquidFM\\data\\images\\renders\\0ac41af42196a7f9\\0004.jpg",
  "model": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0",
  "elapsed_seconds": 8.89,
  "finish_reason": "stop",
  "usage": {
    "completion_tokens": 1186,
    "prompt_tokens": 1903,
    "total_tokens": 3089,
    "completion_tokens_details": null,
    "prompt_tokens_details": {
      "audio_tokens": null,
      "cached_tokens": 0
    }
  },
  "raw_content": "{\n  \"document_type\": \"Complaint Form\",\n  \"languages\": [\n    \"Hindi\",\n    \"English\"\n  ],\n  \"fields\": [\n    {\n      \"field_name\": \"Reason for delay in reporting by the compliant informant\",\n      \"value\": \"कार्यान्वयन के लिए प्रतिक्रिया देने के लिए कोई कार्य नहीं किया गया है।\"\n    },\n    {\n      \"field_name\": \"Particulars of items of interest\",\n      \"value\": \"संपत्ति की विवरण\"\n    },\n    {\n      \"field_name\": \"Description\",\n      \"value\": \"विवरण\"\n    },\n    {\n      \"field_name\": \"S.No. (संख्या)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Total Value of items in Rs./\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Inquest Report/U.D. Case No., If any (मुख्य संख्या रिपोर्ट/ मुख्य मामला संख्या)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"U.D. Case No. (मुख्य मामला संख्या)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"First Information Contents (प्रमुख जानकारी)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Action taken : since the above information reveals commission of offences w/s as mentioned at Item No. 2 :\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Registered the case and took up the investigation (प्रमुख मामला को संदर्भित किया और जांच की गई)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Directed (Name of I.O.) (नाम के अधिकारी)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Rank (पद)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"No. (संख्या)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"to take up the investigation (जांच करने के लिए संदर्भित किया गया)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Refused investigation due to (अनुचित कारण)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"F.I.R. read over to the complainant/informant, admitted to be correctly recorded and a copy given free of cost.\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Signature/Thumb impression of the Complainant/Informant\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Signature of Officer in Charge of Police Station\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Name (नाम)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"Rank (पद)\",\n      \"value\": \"\"\n    },\n    {\n      \"field_name\": \"No. (संख्या)\",\n      \"value\": \"\"\n    }\n  ],\n  \"uncertain_fields\": [\n    \"संख्या (संख्या)\",\n    \"मुख्य मामला संख्या (मुख्य मामला संख्या)\",\n    \"प्रमुख जानकारी (प्रमुख जानकारी)\",\n    \"प्रमुख मामला को संदर्भित किया और जांच की गई (प्रमुख मामला को संदर्भित किया और जांच की गई)\"\n  ]\n}",
  "extraction": {
    "document_type": "Complaint Form",
    "languages": [
      "Hindi",
      "English"
    ],
    "fields": [
      {
        "field_name": "Reason for delay in reporting by the compliant informant",
        "value": "कार्यान्वयन के लिए प्रतिक्रिया देने के लिए कोई कार्य नहीं किया गया है।"
      },
      {
        "field_name": "Particulars of items of interest",
        "value": "संपत्ति की विवरण"
      },
      {
        "field_name": "Description",
        "value": "विवरण"
      },
      {
        "field_name": "S.No. (संख्या)",
        "value": ""
      },
      {
        "field_name": "Total Value of items in Rs./",
        "value": ""
      },
      {
        "field_name": "Inquest Report/U.D. Case No., If any (मुख्य संख्या रिपोर्ट/ मुख्य मामला संख्या)",
        "value": ""
      },
      {
        "field_name": "U.D. Case No. (मुख्य मामला संख्या)",
        "value": ""
      },
      {
        "field_name": "First Information Contents (प्रमुख जानकारी)",
        "value": ""
      },
      {
        "field_name": "Action taken : since the above information reveals commission of offences w/s as mentioned at Item No. 2 :",
        "value": ""
      },
      {
        "field_name": "Registered the case and took up the investigation (प्रमुख मामला को संदर्भित किया और जांच की गई)",
        "value": ""
      },
      {
        "field_name": "Directed (Name of I.O.) (नाम के अधिकारी)",
        "value": ""
      },
      {
        "field_name": "Rank (पद)",
        "value": ""
      },
      {
        "field_name": "No. (संख्या)",
        "value": ""
      },
      {
        "field_name": "to take up the investigation (जांच करने के लिए संदर्भित किया गया)",
        "value": ""
      },
      {
        "field_name": "Refused investigation due to (अनुचित कारण)",
        "value": ""
      },
      {
        "field_name": "F.I.R. read over to the complainant/informant, admitted to be correctly recorded and a copy given free of cost.",
        "value": ""
      },
      {
        "field_name": "Signature/Thumb impression of the Complainant/Informant",
        "value": ""
      },
      {
        "field_name": "Signature of Officer in Charge of Police Station",
        "value": ""
      },
      {
        "field_name": "Name (नाम)",
        "value": ""
      },
      {
        "field_name": "Rank (पद)",
        "value": ""
      },
      {
        "field_name": "No. (संख्या)",
        "value": ""
      }
    ],
    "uncertain_fields": [
      "संख्या (संख्या)",
      "मुख्य मामला संख्या (मुख्य मामला संख्या)",
      "प्रमुख जानकारी (प्रमुख जानकारी)",
      "प्रमुख मामला को संदर्भित किया और जांच की गई (प्रमुख मामला को संदर्भित किया और जांच की गई)"
    ]
  }
}

## Response:
11/14/2026, 2:31:14 PM · gpt-6-t-mini

I’ll break this into the data format, the actual training pipeline used by modern OCR/VLM systems, and a practical recipe for your Hindi–English form dataset. I’ll also distinguish **OCR transcription** from **document extraction**, because the JSON example you shared is mostly the latter.

> **Designing OCR training data**
>
> I’m treating this as document understanding rather than plain OCR. The target should be canonical, validated JSON paired with each image; Markdown can serve as an auxiliary, human-readable representation, not the primary label.
> 
> I’d separate raw text recognition, layout extraction, and field mapping, then train or fine-tune a pretrained vision-language model rather than starting from scratch. The model’s generated examples should be reviewed and corrected before reuse.
> 
> A strong record includes the image, task instruction, normalized JSON target, optional transcript, reading order, bounding boxes, confidence, and metadata. Keep missing fields explicit and avoid embedding logs or inconsistent serialization in targets.
>
> **Designed OCR training data**
>
> 
>
> **Refining model details**
>
> I’m verifying the exact architecture and training distinctions before recommending a pipeline. The key distinction remains: OCR transcription, layout understanding, and structured extraction need different supervision, even when one model ultimately performs all three.
>
> **Structuring OCR supervision**
>
> I’m recommending JSONL or Parquet manifests referencing images, with one canonical target per example. I’d omit logs and duplicate fields, while preserving transcription, reading order, regions, and structured extraction labels when available.
> 
> Markdown can be an auxiliary target for readable layout reconstruction, but fixed-schema JSON is better for reliable downstream extraction.
>
> **Comparing training strategies**
>
> I’m distinguishing full pretraining from practical adaptation. Large OCR systems combine massive image-text corpora, OCR annotations, layout grounding, instruction tuning, and preference refinement; I’d fine-tune an existing vision-language model first, especially for Hindi.
>
> **Refined model details, structured OCR supervision, and compared training strategies**
>
> 
>
> **Separating training objectives**
>
> I’m organizing the system around transcription, layout, and field extraction as related but distinct tasks. I’d derive multiple supervised examples from each image, using faithful text labels alongside structured JSON and optional Markdown.
>
> **Building the dataset**
>
> I’m recommending one image per record with a canonical, validated JSON target, plus optional transcription and layout annotations. I’d omit duplicated fields, model metadata, timing, and guessed uncertainty; use explicit nulls for unreadable values.
>
> **Expanding annotation coverage**
>
> I’m adding realistic page variation: high-resolution scans, crops, skew, blur, compression, tables, stamps, mixed scripts, and synthetic documents with exact labels. I’d validate weak labels, preserve bounding boxes, and prioritize difficult examples through iterative review.
>
> **Choosing the training scale**
>
> I’m distinguishing compact OCR pretraining from foundation-model pretraining. A specialized detector-recognizer can be trained from scratch, but a large vision-language OCR model generally requires an existing language model, vision encoder, and massive multimodal data.
>
> **Mapping large-scale recipes**
>
> I’m comparing public reports cautiously: large OCR systems use separate visual encoders, language decoders, OCR/layout mixtures, synthetic data, staged pretraining, and instruction tuning. Reported compute and data volumes are far beyond ordinary workstation-scale training.
>
> **Planning the training pipeline**
>
> I’m prioritizing canonical JSON targets, auxiliary Markdown, faithful transcription, layout boxes, and Hindi–English coverage. I’ll recommend staged adaptation, document-level splitting, and metrics for text, structure, fields, and geometry.
>
> **Designing the dataset**
>
> I’m converging on image-plus-JSONL supervision: keep images as files, store paths and targets in JSONL or Parquet, and serialize each canonical target consistently for multimodal training.
> 
> Markdown can be an auxiliary target for readable page reconstruction, but I wouldn’t use it as the primary annotation format. Structured JSON is better for reliable fields, validation, and downstream applications.
> 
> I’d separate three tasks:
> 
> 1. **Faithful OCR:** reproduce printed text exactly, preserving script, punctuation, line order, and spelling.
> 2. **Layout understanding:** return blocks, reading order, block types, and normalized bounding boxes or quadrilaterals.
> 3. **Semantic extraction:** map recognized content into a fixed schema, using `null` for missing or unreadable values.
> 
> For Hindi–English documents, I’d preserve Unicode carefully, including Devanagari matras, conjuncts, nukta characters, and mixed-script text. I wouldn’t silently transliterate or “correct” the transcription.
> 
> A practical record could contain:
> 
> ```json
> {
>   "id": "page_0004_extract",
>   "image": "images/page_0004.jpg",
>   "task": "extract_json",
>   "prompt": "Read the document and return the requested fields as valid JSON.",
>   "target": {
>     "document_type": "incident_report",
>     "language": ["hi", "en"],
>     "fields": {
>       "reason_for_delay": {
>         "value": null,
>         "status": "illegible"
>       },
>       "date": {
>         "value": "2025-02-14",
>         "status": "present"
>       }
>     }
>   }
> }
> ```
> 
> For OCR and layout, I’d use separate records derived from the same image:
> 
> ```json
> {
>   "id": "page_0004_blocks",
>   "image": "images/page_0004.jpg",
>   "task": "ocr_blocks",
>   "prompt": "Transcribe the page faithfully and return reading-order blocks.",
>   "target": {
>     "blocks": [
>       {
>         "text": "Visible text here",
>         "bbox": [85, 120, 910, 185],
>         "reading_order": 0,
>         "language": "hi"
>       }
>     ]
>   }
> }
> ```
> 
> The loader can convert `target` into one canonical JSON string for the assistant response. I wouldn’t store both `raw_content` and a duplicated extraction object unless they represent genuinely different annotations. I’d also exclude timing, model names, token usage, and generation logs from training targets.
> 
> For unreadable content, I’d distinguish `blank`, `illegible`, and `not_present`; I wouldn’t train on a teacher model’s guess as if it were ground truth. Generated labels should be reviewed, especially for Hindi.
> 
> I’m treating the public large-model recipes as guidance rather than exact blueprints: they combine visual encoders, language decoders, document OCR, layout grounding, synthetic data, staged pretraining, and instruction tuning. A specialized recognizer can be trained from scratch, but reproducing a full-scale vision-language OCR model requires vastly more data and compute than ordinary fine-tuning.
>
> **Standardizing targets**
>
> I’m standardizing field states explicitly:
> 
> `blank` means the printed field exists but is empty; `illegible` means content exists but cannot be read; `not_present` means the field does not exist; `present` requires a value.
> 
> This prevents empty strings and separate uncertainty lists from teaching the model to guess.
> 
> I’d use a staged system:
> 
> 1. **Layout stage:** detect regions, fields, tables, and reading order.
> 2. **Recognition stage:** transcribe text faithfully, including Hindi–English mixed script.
> 3. **Extraction stage:** map recognized content into your fixed JSON schema.
> 4. **Refinement stage:** review difficult examples and retrain on verified corrections.
> 
> Since you’re already exploring a layout detector, I’d keep layout detection modular rather than forcing a newly initialized vision-language model to learn detection, recognition, and semantic extraction simultaneously. memoryciteturn0memory0
> 
> For a pretrained vision-language model, begin with supervised fine-tuning on image-plus-target records. Freeze most of the visual encoder initially, train the projector and language layers, then selectively unfreeze vision components if validation results justify it. Use teacher-generated labels only as provisional data; verified human corrections should carry more weight.
> 
> For a genuinely new OCR system, a detector-recognizer architecture is more realistic than recreating a large multimodal foundation model. Train text-region detection, cropped-text recognition, layout analysis, and JSON extraction as coordinated components. A full vision-language model from random initialization requires enormous multimodal data, pretrained language capacity, and distributed compute.
> 
> The public examples illustrate this scale difference. One large OCR system combines a visual encoder, a language decoder, tens of millions of document pages, multilingual OCR data, synthetic documents, layout annotations, and staged training. Another system builds on an existing OCR checkpoint and concentrates on decoder or attention changes rather than starting everything from zero. These are useful architectural patterns, not small-workstation recipes. citeturn598986view2turn598986view3
> 
> I’d evaluate each capability separately:
> 
> - Character error rate and word error rate for transcription
> - Exact string match for critical fields
> - JSON validity and schema compliance
> - Field-level precision, recall, and F1
> - Reading-order accuracy
> - Bounding-box overlap for layout
> - Table-specific structural metrics when tables matter
> 
> I’d split data by original document, form template, or source PDF before creating crops or augmented versions. Otherwise, near-duplicate pages can leak between training and validation and make results look much better than real deployment performance.
> 
> One important caution: the specific small vision-language checkpoint being considered does not publicly list Hindi among its supported languages, so I wouldn’t assume strong Devanagari performance merely because it handles several other languages. I’d benchmark Hindi–English pages directly and consider a multilingual base with explicit Devanagari coverage if results are weak. citeturn598986view0
>
> **Planned dataset design and training pipeline with standardized targets**
>
> 
>
> **Searched 12 websites**
>
> 
>
> Worked for 5m 11s

# Training an OCR model from scratch: the right approach

My recommendation is to use **image + structured JSON/JSONL annotations as your primary training dataset**, not Markdown alone.

However, there is an important distinction: the example you've shared is not simply an OCR dataset. It is a **document understanding and information extraction dataset**\.

Your example asks a model to\:

1. Read Hindi and English text from a document.
2. Identify the document type.
3. Understand field names and their associated values.
4. Preserve empty fields.
5. Detect uncertain or illegible information.
6. Return a structured JSON response.

That is considerably more sophisticated than conventional OCR, which primarily converts image pixels into text.

## 1\. JSON vs Markdown: which should you train with?

Both are useful, but they teach the model different behaviours.

| Format | Best use | My recommendation |
|--------|--------|--------|
| Image + transcription JSON | Text recognition and ground truth | Essential |
| Image + bounding-box JSON | Text detection and spatial understanding | Essential for layout OCR |
| Image + Markdown | Reading order and document reconstruction | Useful auxiliary task |
| Image + extraction JSON | Semantic field extraction | Essential for your use case |
| Image + HTML | Tables and structured document reconstruction | Optional |

I would not choose between JSON and Markdown as mutually exclusive formats. I would train different tasks using the same source document.

For example, a single FIR document could generate three training samples\:

**Task A - Raw OCR**

Input: Document image.

Output\:

```text
FIRST INFORMATION REPORT

Police Station: ...
District: ...
Date: ...

Reason for delay in reporting:
कार्यान्वयन के लिए...
```

**Task B - Document reconstruction**

Input: Document image.

Output\:

```markdown
# First Information Report

| Field | Value |
|---|---|
| Police Station | |
| District | |
| Date | |
| Reason for delay | ... |
```

**Task C - Structured extraction**

Input: Document image.

Output\:

```json
{
  "document_type": "FIR",
  "languages": ["Hindi", "English"],
  "fields": [
    {
      "field_name": "Police Station",
      "value": ""
    },
    {
      "field_name": "District",
      "value": ""
    },
    {
      "field_name": "Reason for delay",
      "value": "..."
    }
  ]
}
```

The same image can supervise all three tasks. This is much more useful than training a model to produce only one output format.

**Important\:** Your labels must be ground truth, not just outputs copied from LFM2.5-VL. A teacher model can generate draft annotations, but those outputs need validation. Otherwise, you are teaching your new model to reproduce the teacher's OCR mistakes.

---

## 2\. How Liquid AI, DeepSeek-OCR and Unlimited-OCR actually approach training

These systems are interesting because they use somewhat different approaches to the same problem.

### A. Liquid AI - LFM2.5-VL
[![LFM2-VL: Efficient Vision-Language Models | Liquid AI](https://images.openai.com/static-rsc-4/y06ULOSIyp_1BT67WEzqayxQond3XRBuCRvXso3Fs-llE8YTOHn9C5OaCFreHFyhx6bvImoCiFzfA3ESkBTiAh0pbrSoag9L_rVhvSB5mPcTx1gPSynIy2jW9aYoMAxJv3xs4Ksh9nXLAoKlVPmRrJbMneTjITzSH5WghlO_fPQjrs1A0r0c4j6H1qaA5waB?purpose=fullsize)](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models)

The LFM2.5-VL-1.6B model you are using has two major components\:

| Component | Architecture |
|--------|--------|
| Language backbone | LFM2.5-1.2B-Base |
| Vision encoder | SigLIP2 NaFlex, 400M parameters |
| Total | 1.6B parameters |

It handles images using native resolution processing and tiling. Larger images are split into 512 × 512 patches, with an additional thumbnail providing global context. \[[LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com); [README.md](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/README.md?code=true&utm_source=chatgpt.com)\]

Its publicly disclosed training approach includes\:

1. Pretraining the language and vision representations.
2. Combining image-caption, OCR, grounding and general instruction data.
3. Supervised fine-tuning.
4. Knowledge distillation from larger teacher models.
5. Further preference/reward-based optimization.

Liquid AI's August 2026 announcement for its newer 3B model reports approximately 34 trillion pretraining tokens and four times the vision pretraining tokens of its predecessor. These figures are specific to that release, not a disclosed training budget for your 1.6B checkpoint. \[[LFM2.5-VL-3B: A Better and Faster Vision-Language Model for the Edge \| Blog](https://www.liquid.ai/blog/lfm2-5-vl-3b?utm_source=chatgpt.com); [LFM2 Technical Report](https://arxiv.org/abs/2511.23404?utm_source=chatgpt.com)\]

**The important lesson\:** They aren't training a model exclusively on image-to-JSON pairs. They build general vision and language capabilities first, then train the model on OCR and document-specific tasks.

One detail relevant to your Hindi documents: the LFM2.5-VL-1.6B model card lists eight languages, but Hindi is not among them. You should therefore measure its Devanagari accuracy independently rather than assume its multilingual capability includes strong Hindi OCR. \[[LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com)\]

### B. DeepSeek-OCR
[![Modern OCR for the Large Language & Vision Model Era |](https://images.openai.com/static-rsc-4/XHGF1wVDlc9ctd3jwJx5v06ObORI8JVoK3OU-upZzDRcMf6sapwQ09JXd6ol7-cRDyKsY5-_MSJQ3IpKhuDrwVSFgHidEeJxMMDdF54C_VaErpnEQuOjrMyJUXeK_q3iwqhV7vMnkT1nxXJXIzt2F-7x2tBUepU48SQImDJDMk2KT9MvsJj_r9iMxYT7GksV?purpose=fullsize)](https://roots-automation.github.io/roots-labs/post/ocr/)

This is probably the most interesting architecture to study if you want to build an OCR-focused VLM.

DeepSeek-OCR consists of\:

| Component | Architecture | Parameters |
|--------|--------|--------|
| Visual perception | SAM-base | 80M |
| Visual knowledge | CLIP-large | 300M |
| Token compressor | Two convolutional layers | Part of encoder |
| Decoder | DeepSeek-3B-MoE | 3B total, approximately 570M active |

The vision encoder compresses visual tokens by 16× before passing them to the language decoder.

For instance, a 1024 × 1024 image initially produces 4,096 patch tokens. The compression module reduces this to 256 vision tokens. \[[DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com); [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)\]

But the really interesting part is its data engine.

**How DeepSeek constructs its OCR dataset**

The paper describes approximately 30 million document pages collected from the internet, covering around 100 languages.

They construct different levels of supervision\:

| Dataset component | Approximate scale | Purpose |
|--------|--------|--------|
| Document pages | 30M | General document OCR |
| Fine Chinese annotations | 2M pages | Detailed layout and recognition |
| Fine English annotations | 2M pages | Detailed layout and recognition |
| Minority-language model-generated labels | 600K samples | Improve language coverage |
| Word-level image-text pairs | 3M | Recognition without full-page layout |
| Chinese scene OCR | 10M samples | Natural image text |
| English scene OCR | 10M samples | Natural image text |
| Synthetic charts | 10M images | Chart-to-HTML parsing |
| Chemical formulas | 5M pairs | Formula recognition |

They use tools such as PyMuPDF, PP-DocLayout, MinerU, GOT-OCR2.0 and PaddleOCR to construct annotations. \[[DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com); [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)\]

Notice that they don't depend exclusively on human annotations. They use existing OCR models to generate labels, then use more detailed annotation pipelines for higher-quality samples.

**Their disclosed training mixture\:**

## DeepSeek-OCR training mixture

**OCR data**

# 70%

**General vision data**

# 20%

**Text-only data**

# 10%

Percentages reported in the DeepSeek-OCR technical paper.

Their training has two main stages\:

- **Stage 1: Encoder training.** Train DeepEncoder independently using OCR and general image-text data.
- **Stage 2: End-to-end training.** Combine the trained encoder with the DeepSeek MoE decoder and train the multimodal model. Some vision components are frozen while others are updated.

Their reported training hardware for the second stage was 20 nodes, each with eight A100 40GB GPUs: 160 GPUs in total. \[[DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com); [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)\]

This is large-scale foundation-model training, not ordinary OCR fine-tuning.

### C. Unlimited-OCR
[![Unlimited OCR: How Baidu Solves Long Document OCR](https://images.openai.com/static-rsc-4/TrFdJw3JWN-YHkV18Io8AEDed3i4R2_la6CMEQuD2eYxfDLEtJ3MEIfBE6t_5iClEIFP__iDr-jf44KoDr8oX8yoaFF1uITWrqaWn5KMZYbazxaTUEUK4CIP6zmm9LiZPdo7sFriUHULvpFGmZxbohxVo8R8bcHfH1TqYuYXrYw141QATrb4woZb8rMN3U5a?purpose=fullsize)](https://www.labellerr.com/blog/baidu-unlimited-ocr/)

Unlimited-OCR takes a different approach.

It starts from DeepSeek-OCR rather than building a new vision encoder and language model from random initialization.

Its main innovation is **Reference Sliding Window Attention (R-SWA)**\.

The problem with conventional autoregressive decoding is that the KV cache keeps growing as more text is generated.

Unlimited-OCR changes the attention mechanism so that each generated token can attend to\:

- All original visual and prompt tokens.
- A fixed window of preceding generated tokens.

This keeps the decode-side KV cache bounded while allowing long documents to be transcribed. \[[Unlimited OCR Works](https://arxiv.org/abs/2606.23050?utm_source=chatgpt.com); [Unlimited OCR Works](https://arxiv.org/abs/2606.23050?utm_source=chatgpt.com)\]

Its publicly described training recipe is particularly useful\:

| Parameter | Unlimited-OCR |
|--------|--------|
| Starting checkpoint | DeepSeek-OCR |
| OCR samples | Approximately 2M |
| Single-page / multi-page | 90% / 10% |
| Multi-page synthetic documents | 2-50 pages |
| Context length | 32K tokens |
| Training steps | 4,000 |
| Global batch size | 256 |
| GPUs | 8 × 16 A800 |
| Initial learning rate | 1e-4 |
| Vision encoder | Frozen |
| Main training target | Language decoder |

Its annotations concatenate document element coordinates and text. Coordinates are normalized to 0-1000. Multi-page examples use page separators. \[[Unlimited OCR Works](https://arxiv.org/abs/2606.23050?utm_source=chatgpt.com)\]

This is a crucial distinction\:

**DeepSeek-OCR builds an OCR-focused multimodal model. Unlimited-OCR improves the long-document decoding capabilities of an existing OCR model.**

You don't need to replicate either architecture to build a useful domain-specific OCR model.

---

## 3\. The architecture I would build

For your Hindi-English forms, FIRs, government records and other structured documents, I would build a document AI architecture with three supervised capabilities.

## Proposed OCR architecture

**Document image / PDF**

Original high-resolution pages

**Document preprocessing**

DPI normalization · deskewing · orientation · cropping

**Visual encoder**

ViT / SigLIP / custom visual backbone

High-resolution patch and feature extraction

**Multimodal feature fusion / projector**

Connect visual features to text decoder

**Autoregressive text decoder**

OCR and document understanding

**OCR transcription**

Exact text, lines, reading order

**Layout understanding**

Boxes, tables, paragraphs, fields

**Structured extraction**

FIR fields, entities, values, uncertainty and validated JSON

I would keep the layout detector as a separate component initially. Since you are also exploring DocLayout-YOLO, you can use that as an independent layout model rather than forcing your first VLM to learn detection, recognition and semantic extraction simultaneously.

## 4\. How I would design your actual training dataset

I would maintain one master dataset with multiple tasks per document.

The important point is that **the image is the input, and the annotations are the supervision**\.

I would use the following directory structure\:

```text
ocr-dataset/
│
├── images/
│   ├── fir/
│   │   ├── fir_0001.png
│   │   ├── fir_0002.png
│   │   └── fir_0003.png
│   │
│   ├── invoices/
│   ├── receipts/
│   └── handwritten/
│
├── annotations/
│   ├── ocr_blocks.jsonl
│   ├── markdown.jsonl
│   ├── extraction.jsonl
│   └── layout.jsonl
│
├── splits/
│   ├── train.json
│   ├── validation.json
│   └── test.json
│
└── metadata/
    ├── document_types.json
    └── languages.json
```

### Recommended annotation schema

**A. OCR recognition and layout**

This is where you teach the model to recognize actual text and spatial relationships.

```json
{
  "id": "fir_0004",
  "image": "images/fir/fir_0004.png",
  "task": "ocr_blocks",
  "prompt": "Transcribe all visible text in reading order.",
  "target": {
    "blocks": [
      {
        "type": "heading",
        "bbox": [120, 40, 850, 110],
        "text": "FIRST INFORMATION REPORT",
        "reading_order": 0,
        "language": "en"
      },
      {
        "type": "field_label",
        "bbox": [100, 150, 450, 200],
        "text": "पुलिस स्टेशन",
        "reading_order": 1,
        "language": "hi"
      }
    ]
  }
}
```

Coordinates are normalized to 0-1000 here, following the kind of spatial representation used by Unlimited-OCR.

**B. Document extraction**

This teaches semantic understanding.

```json
{
  "id": "fir_0004",
  "image": "images/fir/fir_0004.png",
  "task": "extract_json",
  "prompt": "Extract all fields without inventing missing values.",
  "target": {
    "document_type": "FIR",
    "languages": ["Hindi", "English"],
    "fields": [
      {
        "field_name": "Police Station",
        "value": "",
        "status": "blank"
      },
      {
        "field_name": "District",
        "value": "Trichy",
        "status": "present"
      },
      {
        "field_name": "Complainant Name",
        "value": null,
        "status": "illegible"
      }
    ]
  }
}
```

I strongly recommend maintaining the distinction between\:

- `present`\: text exists and is readable.
- `blank`\: the printed field is empty.
- `illegible`\: text exists but cannot be confidently read.
- `not_present`\: the field does not exist in this document.

This will help prevent hallucinated values.

**C. Markdown reconstruction**

Store another annotation for the same document\:

```json
{
  "id": "fir_0004",
  "image": "images/fir/fir_0004.png",
  "task": "document_to_markdown",
  "target": "# First Information Report\n\n..."
}
```

This teaches the model to reconstruct document structure and reading order.

### One important correction to your existing JSON

Your current record contains\:

```json
{
  "raw_content": "{ ... }",
  "extraction": {
    "...": "..."
  }
}
```

I would change this.

You have essentially duplicated the prediction twice. Also, `model`, `usage`, `elapsed_seconds` and `finish_reason` are inference metadata, not OCR ground truth.

Keep those in a separate experiment log.

For the training dataset, maintain just\:

- Image reference.
- Task/instruction.
- Verified target.
- Optional annotation metadata.

Store targets as structured objects in your annotation files, then serialize the assistant target consistently when constructing the model's training conversations.

---

## 5\. How to build a large-scale OCR data engine

This is probably more important than the model architecture.

Look at what DeepSeek did: they didn't manually annotate 30 million pages. They built a data engine using existing extraction tools, automated annotation, synthetic generation and quality refinement.

You can implement a smaller version.

## Your OCR data engine

1. **Real document collection**
   
   Collect PDFs, scans, government forms, invoices, court records, receipts and handwritten documents.
2. **PDF ground-truth extraction**
   
   Use PyMuPDF to extract native PDF text where available. Retain page coordinates and verify the reading order.
3. **Synthetic document generation**
   
   Generate forms using known text, multiple fonts, different layouts, Hindi-English combinations, tables and empty fields. Render them as images, retaining the exact original text and coordinates.
4. **Teacher model annotation**
   
   Use PaddleOCR, Surya, LFM, DeepSeek-OCR and other suitable models to produce candidate annotations.
5. **Annotation validation**
   
   Find disagreements between models, manually correct important errors and reject unreliable annotations.
6. **Dataset construction**
   
   Generate OCR, layout, Markdown and extraction training examples from the validated documents.
7. **Active learning**
   
   Evaluate the trained model, collect documents where it fails and send those examples back into the annotation pipeline.

For your specific Hindi-English use case, synthetic data should include Devanagari conjuncts, matras, nukta, different numeral representations, mixed-script words, stamps, skew, blur and low-quality scans.

Don't generate only perfectly clean PDFs. Real-world document degradation is a major part of the problem.

Also, split the dataset by original document and document template before generating crops or synthetic variants. Otherwise, nearly identical pages can leak into training and test sets.

---

## 6\. What training strategy should you actually follow?

There are two possible projects here.

### Option A: Train a conventional OCR model from scratch

This is feasible with a much smaller compute budget.

Architecture\:

- Text detection: DBNet or a YOLO-based detector.
- Text recognition: CRNN/CTC or a Transformer recognizer.
- Layout understanding: document layout detector.
- Structured extraction: separate language model.

You will need bounding boxes or polygons for detection, cropped text images with exact transcripts for recognition, and structured labels for extraction.

This is a good choice if your objective is a specialized Hindi-English OCR engine.

### Option B: Build an end-to-end OCR VLM

This is the direction of DeepSeek-OCR and LFM2.5-VL.

You need\:

| Component | Suggested approach |
|--------|--------|
| Vision encoder | Pretrained ViT/SigLIP or SAM-derived encoder |
| Visual compression | Patch merging, convolutional downsampling or resampler |
| Projector | Connect visual embeddings to decoder embeddings |
| Decoder | Pretrained multilingual language model |
| Training | Multimodal SFT, followed by optional preference optimization |

The training stages would be\:

## Suggested VLM training stages

**Stage 1 - Vision-language alignment**

Teach the vision encoder and decoder to associate document images with text.

(Image-text pretraining)

**Stage 2 - OCR specialization**

Train on page transcription, word recognition, bounding boxes, reading order, tables and formulas.

(OCR supervised training)

**Stage 3 - Document understanding**

Train on Hindi-English forms, document classification, structured extraction and JSON generation.

(Multimodal SFT)

**Stage 4 - Quality optimization**

Improve exact transcription, uncertainty handling and field extraction using preference data and targeted hard examples.

(Optional preference optimization)

For your available GPU environment, I would begin with a pretrained model and LoRA/QLoRA, then selectively unfreeze vision layers when you have evidence that the encoder is the bottleneck.

Do not train a randomly initialized 1.6B VLM just because you want to call it a model trained from scratch. You would first have to solve general language and visual representation learning.

There is also an important practical point: **use the original Hugging Face model checkpoint for training, not the GGUF Q8 quantized inference artifact you used in your example.** Liquid AI provides its native checkpoint for fine-tuning and recommends SFT approaches using TRL or Unsloth. \[[LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com); [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com)\]

---

## 7\. How should you measure whether it is actually learning OCR?

Don't evaluate everything with just exact JSON matching.

| Task | Metric |
|--------|--------|
| Text recognition | Character Error Rate (CER) |
| Word recognition | Word Error Rate (WER) |
| Text detection | Precision, recall, mAP |
| Spatial grounding | Bounding-box IoU |
| Reading order | Ordering accuracy |
| JSON extraction | Field-level precision, recall, F1 |
| JSON structure | Schema validity |
| Document parsing | Table and layout structure accuracy |

For Hindi, measure CER separately from English. Also evaluate mixed-language fields separately.

A particularly important metric for your system is the false-population rate: how often does the model fill a field that is actually blank?

A model that produces beautiful JSON but invents missing information is not a good document OCR model.

---

## 8\. My recommended implementation roadmap

Based on what you're trying to achieve, I would approach this as a research project rather than immediately attempting to recreate DeepSeek-OCR.

# Your OCR research roadmap

(Phase 1)

**Build the data engine**

Establish image storage, JSONL annotations, PDF extraction, synthetic page generation and quality checks.

Deliverable: a reproducible image-to-annotations dataset.

---

(Phase 2)

**Build the OCR baseline**

Train/fine-tune the layout detector and text recognizer. Establish CER, WER and layout benchmarks.

Deliverable: a measurable multilingual OCR baseline.

---

(Phase 3)

**Build the OCR VLM**

Start with a multilingual pretrained VLM, perform multimodal SFT on your three tasks and compare frozen versus trainable vision encoders.

Deliverable: an end-to-end document parsing model.

---

(Phase 4)

**Optimize the architecture**

Experiment with visual token compression, high-resolution processing, efficient attention and multi-page inference.

Deliverable: a smaller and faster OCR model.

---

(Phase 5)

**Production evaluation**

Benchmark on unseen document templates and degraded real-world scans. Validate structured output and uncertainty.

Deliverable: a reliable document AI system.

## 9\. Research papers and implementations worth studying

These are the references I would prioritize\:

| Work | Why study it |
|--------|--------|
| [DeepSeek-OCR](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com) | Encoder compression, OCR data engine and multimodal training |
| [Unlimited OCR](https://arxiv.org/abs/2606.23050?utm_source=chatgpt.com) | Long-context OCR and Reference Sliding Window Attention |
| [LFM2.5-VL](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com) | Compact vision-language backbone and fine-tuning |
| [LFM2 Technical Report](https://arxiv.org/abs/2511.23404?utm_source=chatgpt.com) | Foundation-model architecture and training |
| [Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2?utm_source=chatgpt.com) | Large-scale multilingual and synthetic OCR data |

---

## My final recommendation

I would make one important decision before starting\:

**Don't begin by building a 1.6B-parameter OCR model. Begin by building a high-quality OCR data engine.**

Your biggest opportunity is the Hindi-English document domain. Generic OCR models already exist, but their ability to handle mixed scripts, complex forms, poor-quality scans, tables and structured government documents needs to be independently evaluated.

For an initial implementation, I'd use\:

- **Dataset\:** Image + JSONL with multiple task-specific targets.
- **Layout\:** DocLayout-YOLO.
- **Recognition\:** Transformer-based recognizer.
- **VLM\:** A multilingual pretrained vision-language backbone.
- **Training\:** PyTorch + Hugging Face Transformers + TRL.
- **Annotation\:** PaddleOCR/Surya plus model-assisted validation.
- **Output\:** Raw OCR, Markdown and structured JSON.

The end goal should be a model that genuinely understands the document image, rather than one that simply imitates an existing model's JSON response.

## Prompt:
11/14/2026, 2:37:50 PM

lets say i want to train an ocr from stract or ffinetune a model for my language specific then how will train in stages like sft, pretraining or RL? and what what dataset type i need to use.

[Attachment: Architecture v2.md]

[Attachment: Tab 1.md]

## Response:
11/14/2026, 2:37:55 PM · gpt-6-t-mini

You’ve attached two design notes, so I’ll use those as the starting point rather than give you a generic OCR recipe. The key distinction I’ll make is between **training the visual recognizer**, **teaching a language stack to understand Indic text**, and **post-training the model to follow OCR/document tasks**—because SFT and RL solve different problems at those stages.

> **Structuring the training plan**
>
> I’m grounding the explanation in the provided schedules and distinguishing pretraining from fine-tuning. I’ll clarify each stage, dataset role, sampling mix, expected capability, and where the plan depends on starting from scratch rather than an existing pretrained model.
>
> **Mapping training stages**
>
> I’m separating language-specific fine-tuning from true scratch training. The plan progresses from tokenization and OCR supervision to page-level multitask learning, schema-focused SFT, optional verifiable-reward optimization, and final distillation or quantization.
>
> **Defining dataset roles**
>
> I’m organizing data by objective: text modeling, OCR recognition, page structure, extraction, document memory, reward learning, and evaluation. I’ll also separate draft mixture estimates from universal prescriptions and emphasize clean template-level splits.
>
> **Sequencing adaptation paths**
>
> I’m distinguishing adaptation from scratch training: fine-tuning usually skips text pretraining, preserves the tokenizer, adapts visual-text alignment, then trains page-level tasks, extraction SFT, optional memory, rewards, and deployment optimization.
>
> **Producing the training curriculum**
>
> I’m consolidating the proposed scratch and fine-tuning recipes, including stage terminology, data mixtures, OCR-specific losses, optional memory, reward design, and evaluation safeguards. I’ll flag estimates as draft assumptions rather than guarantees.
>
> **Designing the curriculum**
>
> I’m ready to turn the proposed stages into a concrete scratch-versus-fine-tuning workflow, with dataset formats, training objectives, exit metrics, OCR-specific rewards, and clear warnings around unproven architecture assumptions.
>
> **Designed and sequenced the training curriculum and adaptation paths**
>
> 
>
> **Integrated architecture V2 training stages and datasets**
>
> 
>
> Worked for 5m 27s

Looking at both your uploaded documents - **Indic OCR Foundation Model: From-Scratch Training Plan** and **Architecture v2: MoE Model with Whole-PDF Memory** - I think you are approaching this in the right direction.

But I would make one important distinction before designing the training pipeline\:

**Training an OCR model for a new language and training an OCR foundation model from scratch are two different projects.**

For a language-specific model, you may not need to train a language model from scratch at all. You need to teach an existing multimodal model to recognize the visual characteristics of your script, understand its text, and produce accurate document representations.

Let's design both approaches.

# 1\. First understand the three types of training

These are the three major concepts you mentioned.

(1\. Pretraining)

**Learn general capabilities**

The model learns language patterns, visual representations, character shapes, words, reading order and document structure.

---

(2\. Supervised fine-tuning (SFT))

**Learn the tasks you want it to perform**

You teach it to transcribe documents, identify fields, reconstruct tables, produce Markdown and return structured JSON.

---

(3\. Reinforcement learning (RL))

**Improve behaviour using rewards**

You reward correct transcriptions, accurate bounding boxes, valid extraction and appropriate uncertainty handling.

One subtlety: multimodal OCR pretraining can itself use supervised image-to-text pairs and next-token prediction. The distinction is not simply whether labels are present. It is the scope and objective of the training.

| Training type | Main question |
|--------|--------|
| Language pretraining | Does the model understand the language? |
| Vision-language alignment | Can it associate glyphs and images with text? |
| Document pretraining | Can it understand complete documents and their layout? |
| SFT | Can it follow the required OCR/document instructions? |
| RL | Can it improve its decisions against measurable quality criteria? |

You do not necessarily need all five stages for every project.

# 2\. Two training pipelines

**Approach A - Fine-tune an existing VLM**

(My recommendation)

1. Evaluate the existing model on the target language.
2. Adapt the vision-language connection using OCR image-text pairs.
3. Train on complete pages and document structures.
4. SFT on language-specific OCR tasks.
5. Optional RL for measurable improvements.

**No language-model pretraining from scratch required.**

---

**Approach B - Train a new OCR foundation model**

(Research-intensive)

1. Build a language-specific tokenizer and data engine.
2. Pretrain the language stack on text corpora.
3. Align the vision encoder and language stack.
4. Pretrain on full-page OCR and document understanding.
5. Train document memory, if needed.
6. SFT on application-specific document tasks.
7. Optional RL.

**Requires substantially more data and compute.**

Your `Tab 1.md` describes Approach B: a pretrained C-RADIO vision encoder, but a new tokenizer and language stack trained from scratch. Therefore, this is technically a hybrid foundation-model training approach, not training every component from random initialization. fileciteturn1file3

---

# 3\. Approach A: Fine-tuning for a language-specific OCR model

Let's say you already have an existing 1B-3B VLM and want to make it excellent at Hindi OCR.

This is how I would train it.

## Stage 0 - Dataset preparation and baseline evaluation

Before any training, you need a dataset of images and verified transcripts.

Do not start by generating millions of examples. First build a small, high-quality evaluation set.

I would prepare these types of samples\:

| Data type | Example |
|--------|--------|
| Word images | Individual Hindi words |
| Line images | Full Hindi sentences |
| Paragraph images | Multiple lines |
| Document images | Complete FIR or invoice |
| Tables | Rows, columns, merged cells |
| Mixed scripts | English and Hindi in the same line |
| Degraded images | Blur, skew, compression and low resolution |
| Handwritten images | Actual handwritten text |

For every image, retain its exact Unicode transcription.

Example\:

```json
{
  "image": "images/hindi/line_0001.png",
  "task": "line_ocr",
  "target": {
    "text": "भारत सरकार गृह मंत्रालय",
    "language": "hi"
  }
}
```

You should also inspect your existing model's tokenizer. If a Hindi word is fragmented into an excessive number of tokens, that can increase sequence length and make learning less efficient.

Do not automatically replace the tokenizer of an existing VLM. Doing so requires resizing and retraining the corresponding embedding and output layers.

**Output\:** A clean dataset, a golden evaluation set and baseline CER/WER measurements.

## Stage 1 - Vision-language adaptation (OCR continued pretraining)

(Most important stage for language-specific OCR)

This is where the model learns to associate visual glyphs with the correct Unicode characters.

Use\:

- Synthetic word images.
- Real printed word images.
- Synthetic line images.
- Real printed line images.
- Handwritten words.
- Mixed-script samples.

The training objective is straightforward\:

$$
\mathcal{L}_{OCR} =
-\sum_{t=1}^{T}
\log P(y_t \mid y_{<t}, I)
$$

Where\:

- $I$ is the input image.
- $y_t$ is the ground-truth output token.
- $y_{<t}$ is the preceding text.

You are minimizing the difference between the generated transcription and the correct Unicode text.

**How I would train\:**

| Substage | What to train |
|--------|--------|
| 1A | Projector/neck and new task embeddings |
| 1B | Decoder with LoRA or full fine-tuning |
| 1C | Unfreeze selected vision encoder layers |
| 1D | Word → line → paragraph curriculum |

Initially freeze the vision encoder. Train the visual projection and language adapter. Then unfreeze selected vision layers using a smaller learning rate.

The exact balance depends on whether your existing encoder already represents the target script effectively.

Your attached plan uses this same basic approach: first train the neck while keeping the encoder and language stack frozen, then gradually unfreeze the language stack and move from word to line to paragraph images. fileciteturn1file0

**Expected result\:** The model can recognize individual characters, words, lines and paragraphs in the target language.

Do not move on just because the model produces fluent-looking Hindi. Measure its character-level accuracy.

## Stage 2 - Full-page OCR and document pretraining

This is where the model evolves from a text recognizer into a document OCR model.

Now increase the complexity of training examples.

Instead of\:

`Image → Hindi sentence`

You train\:

`Full document image → Structured document representation`

I would introduce eight tasks, based on your Tab 1 design.

| Task | Input | Expected output |
|--------|--------|--------|
| Page text | Full page | Plain transcription |
| Page parsing | Full page | Ordered blocks with text and coordinates |
| Layout only | Full page | Headings, tables, paragraphs and boxes |
| Spotting | Full page | Individual words with coordinates |
| Region reading | Image + bounding box | Transcription of selected region |
| Grounding | Image + phrase | Bounding box of phrase |
| Tables/formulas | Table image | HTML or LaTeX |
| Image translation | Document crop | Translated text preserving block structure |

fileciteturn1file1

The important idea is that **one page can generate multiple training examples**\.

For example, the same FIR image could be used for\:

```text
Task 1: Transcribe the entire page.

Task 2: Detect the heading and paragraphs.

Task 3: Identify the bounding box of "पुलिस स्टेशन".

Task 4: Read the text inside the specified bounding box.

Task 5: Extract the complaint details into JSON.
```

You do not need to annotate every task manually. Synthetic pages provide exact boxes, text and reading order. Layout datasets can provide box supervision even when they have no transcriptions.

### Suggested data mixture

Your existing design proposes this initial distribution for full-page training\:

## Stage 2: Full-page training mixture

The proposed token mixture in your uploaded plan, not a universal optimum.

Keep text-only replay in the mixture to avoid degrading language-generation capabilities. The 5% is your proposed starting point, not a required percentage. fileciteturn1file1

**Expected result\:** The model can recognize and structurally reconstruct entire pages.

## Stage 3 - Domain-specific SFT

Now you train the model to perform the actual task that your application needs.

This is where your original JSON example becomes especially useful.

For example\:

**User input\:** An FIR image and an extraction instruction.

**Assistant target\:**

```json
{
  "document_type": "FIR",
  "language": ["hi", "en"],
  "fields": [
    {
      "field_name": "Police Station",
      "value": "",
      "status": "blank",
      "bbox": [120, 200, 480, 245]
    },
    {
      "field_name": "Complainant Name",
      "value": "राम कुमार",
      "status": "present",
      "bbox": [120, 260, 480, 305]
    },
    {
      "field_name": "District",
      "value": null,
      "status": "illegible",
      "bbox": [120, 310, 480, 350]
    }
  ]
}
```

I would create separate SFT tasks\:

| Task | Training target |
|--------|--------|
| OCR transcription | Plain text |
| Document parsing | Ordered blocks and coordinates |
| Field extraction | Structured JSON |
| Table extraction | HTML/JSON |
| Uncertainty handling | Blank, illegible or absent |
| Document classification | Document type and language |
| PDF reconstruction | Markdown or structured document output |

Your real-world scans should be heavily represented here. Use human-verified annotations or carefully validated teacher labels.

**Training strategy\:**

- Fine-tune all relevant components using a low learning rate.
- Retain some Stage 2 examples to prevent forgetting.
- Train on actual task prompts rather than only one generic OCR instruction.
- Include blank and unreadable documents.
- Use a JSON schema validator to check structural consistency.

Your uploaded plan proposes replaying about 20% of the previous document-pretraining data, using roughly one-tenth of the previous learning rate and one to two passes. This is a reasonable experiment to start with, not a universal setting. fileciteturn1file4

(Stage 3 result)

You now have a language-specific OCR VLM that can perform real document tasks.

## Stage 4 - RL / preference optimization (optional)

This is where I would be careful.

For OCR, RL should not be your main training technique.

Why?

Suppose you have this image\:

Ground truth

**भारत सरकार**

(Correct)

**भारत सरकार**

(Incorrect)

**भारत सरकर**

A single-character error can change the meaning or accuracy of a document.

You already have the correct transcript. Supervised cross-entropy directly teaches the model which character to produce.

RL introduces sampled outputs and rewards. It is generally less direct for this kind of exact recognition.

Where RL becomes interesting is when the model needs to make more complicated decisions.

### What should you reward?

I would design a composite reward\:

$$
R = \lambda_1 R_{\text{text}}
+\lambda_2 R_{\text{layout}}
+\lambda_3 R_{\text{schema}}
+\lambda_4 R_{\text{fields}}
-\lambda_5 P_{\text{hallucination}}
$$

| Reward | How to calculate it |
|--------|--------|
| Text accuracy | Normalized character edit similarity |
| Layout accuracy | Bounding-box IoU |
| Parsing accuracy | Correct block types and reading order |
| JSON correctness | Schema validity |
| Field accuracy | Exact field/value matching or field F1 |
| Hallucination penalty | Penalize invented text in blank or absent fields |
| Uncertainty | Reward appropriate abstention on genuinely unreadable content |

You can use GRPO or PPO after SFT, provided you have reliable reward computation.

An example RL training record would contain\:

```json
{
  "image": "fir_0001.png",
  "prompt": "Extract all FIR fields.",
  "reference": {
    "Police Station": "Trichy",
    "District": "Tiruchirappalli"
  },
  "candidates": [
    {
      "output": {
        "Police Station": "Trichy",
        "District": "Tiruchirappalli"
      },
      "reward": 1.0
    },
    {
      "output": {
        "Police Station": "Trichy",
        "District": "Chennai"
      },
      "reward": 0.35
    }
  ]
}
```

The rewards here are illustrative, not a proposed scoring formula.

**One major warning\:** Never reward a model just because its JSON looks plausible or its Hindi is grammatically correct. The reward must compare its extraction with evidence in the pixels and verified ground truth.

For your model, I would prioritize RL for\:

- Choosing whether to return a value or flag uncertainty.
- Correctly handling complex tables and reading order.
- Selecting the correct region when fields are ambiguous.
- Improving document-level parsing.

I would not use RL as a substitute for OCR transcription labels.

Your `Tab 1.md` also places reinforcement after instruction tuning, and describes rule-based rewards for spotting and parsing. That is the correct order. fileciteturn1file8

---

# 4\. Approach B: Training the language stack from scratch

Now let's consider the more ambitious approach in your Indic OCR Foundation Model plan.

In this case, I would use the following stages.

## From-scratch training curriculum

1. **Stage 0 - Tokenizer and data engine**
   
   Unicode text corpora, language balance, fonts, renderer, degradation and annotation tooling.
2. **Stage 1 - Language pretraining**
   
   Monolingual text, translation pairs where relevant, Markdown, HTML, LaTeX and structured outputs.
3. **Stage 2 - Visual alignment**
   
   Word and line images, exact transcripts, visual features, and bounding-box targets.
4. **Stage 3 - Layout-aware multimodal pretraining**
   
   Full pages, page parsing, transcription, tables, spotting, grounding and region reading.
5. **Stage 4 - Document state (optional)**
   
   Multi-page documents with repeated entities and damaged text.
6. **Stage 5 - Instruction SFT**
   
   Real scans, application instructions, structured extraction and uncertainty.
7. **Stage 6 - RL (optional)**
   
   Verifiable rewards for transcription, spotting, parsing and structured extraction.

This follows the seven-stage curriculum in your uploaded plan. fileciteturn1file3

Let's look at what each stage actually needs.

### Stage 0: Tokenizer

For a language-specific model, the tokenizer matters considerably.

Suppose you are building for Hindi. You need appropriate coverage of\:

- Devanagari consonants and vowels.
- Matras.
- Conjuncts.
- Nukta characters.
- Numerals.
- Punctuation and symbols.
- Hindi-English mixed text.

For a multilingual Indic model, train a shared tokenizer using balanced language corpora.

Your proposal uses approximately 64K vocabulary entries, with dedicated layout, coordinate, language and task tokens. That's a reasonable experiment for the proposed multi-language architecture, but it should be benchmarked against smaller tokenizers for the actual target languages. fileciteturn1file3

### Stage 1: Text-only pretraining

This stage is necessary when your language decoder is randomly initialized.

**You do not train it on images yet.**

You train it on normal language text.

Your document proposes\:

| Objective | Data | Proposed share |
|--------|--------|--------|
| Next-token prediction | Monolingual Indic text | 50% |
| Translation | Indic-English sentence pairs | 35% |
| Structured output | Markdown, HTML, LaTeX and layout schemas | 15% |

fileciteturn1file0

Some potential datasets from your plan\:

- [IndicLLMSuite / Sangraha](https://huggingface.co/datasets/ai4bharat/IndicLLMSuite?utm_source=chatgpt.com) - monolingual Indic text.
- [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC?utm_source=chatgpt.com) - Indic-English parallel sentences.

These are useful because the model needs language knowledge independently of visual recognition.

One correction I'd make: if you are training strictly for one language rather than a multilingual model, you don't need to allocate 35% to translation. Prioritize high-quality monolingual text and the structured formats your OCR model must generate.

### Stage 2: Vision-language alignment

Now bring the image encoder into the pipeline.

For each text sample, generate or collect its visual representation\:

```text
Text:
न्यायालय का नाम

        ↓ Renderer

Image:
[Rendered Hindi text crop]

        ↓ Training

Output:
न्यायालय का नाम
```

Use a curriculum\:

1. Individual words.
2. Full lines.
3. Paragraphs.
4. Mixed-language crops.

For printed language recognition, you can initially use synthetic image-text pairs, but real scans must be included to prevent overfitting to rendered fonts.

Your plan identifies Mozhi, IIIT-INDIC-HW-WORDS and IndicSTR12 for this stage, alongside synthetic crops. fileciteturn1file6

### Stage 3: Document pretraining

Now start training on complete pages, rather than isolated text crops.

This is the largest multimodal training stage in your proposal.

At this point, the model learns\:

- Character recognition.
- Word-level localization.
- Paragraph reading order.
- Document layout.
- Table reconstruction.
- Field relationships.
- Multi-script document understanding.

The important distinction is that page layout should be introduced during pretraining, not added as an afterthought during final SFT.

### Stage 4: Document memory

This comes from your Architecture v2 design.

You propose three levels\:

1. **Output window\:** recent generated text.
2. **Reference\:** current and nearby page representations.
3. **Latent memory\:** compressed information retained from older pages.

You want to use the memory to improve recognition of repeated names, identifiers, and other entities across pages.

This is interesting, but it is an experimental feature, not a prerequisite for OCR.

The design proposes synthetic documents with repeated entities, some clean occurrences and some deliberately damaged occurrences. You also propose no-support examples to verify that the model does not hallucinate missing information. fileciteturn2file3

I would first prove this with a tiny model before adding it to your main VLM.

### Stage 5 and 6

These are the same SFT and optional RL stages explained in Approach A.

The difference is that your model now has its own language and document representations, rather than inheriting them from an existing VLM.

---

# 5\. Exactly what datasets do you need?

Here is the dataset inventory I would build for your project.

| Dataset type | Required annotation | Training stage |
|--------|--------|--------|
| Monolingual text | Unicode text | Language pretraining |
| Translation corpus | Source and translated text | Language pretraining |
| Synthetic word images | Image + exact transcript | Visual alignment |
| Real word images | Image + exact transcript | Visual alignment |
| Line images | Image + transcript + optional word boxes | Visual alignment |
| Handwriting | Image + transcript | Visual alignment |
| Full document pages | Image + block text + coordinates | Document pretraining |
| Layout images | Image + block classes + coordinates | Document pretraining |
| Tables | Image + cell structure + text | Document pretraining |
| Forms | Image + key-value pairs + source boxes | SFT |
| Multi-page documents | Images + repeated entities + damaged occurrences | Memory training |
| RL examples | Input + reference + verifiable reward | RL |

### Datasets from your proposed ecosystem

These are the resources identified in your Tab 1 document\:

**Language corpora**

- [IndicLLMSuite / Sangraha](https://huggingface.co/datasets/ai4bharat/IndicLLMSuite?utm_source=chatgpt.com) - language text.
- [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC?utm_source=chatgpt.com) - translation pairs.

---

**Recognition datasets**

- [Mozhi](https://cvit.iiit.ac.in/usodi/tdocrmil.php?utm_source=chatgpt.com) - printed words and lines.
- [IIIT-INDIC-HW-WORDS](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words?utm_source=chatgpt.com) - handwritten words.
- [IndicSTR12](https://ar5iv.labs.arxiv.org/html/2403.08007?utm_source=chatgpt.com) - scene text.

---

**Document layout datasets**

- [IndicDLP](https://huggingface.co/datasets/IndicDLP/IndicDLP-dataset?utm_source=chatgpt.com) - document layout.
- [Sarvam Indic OCR Benchmark](https://huggingface.co/datasets/sarvamai/indic-ocr-bench?utm_source=chatgpt.com) - evaluation.
- [olmOCR-mix](https://arxiv.org/abs/2502.18443?utm_source=chatgpt.com) - English PDF page transcriptions.
- DocLayNet, PubTables-1M and FinTabNet - layout and table training.

These datasets don't all provide the same supervision. In particular, your notes correctly distinguish IndicDLP's layout annotations from text transcription labels. Do not assume a layout annotation is also a transcription. Also, check the licences of datasets whose licences are unspecified or not verified before commercial training. fileciteturn1file5

# 6\. How should you organize all these training samples?

I recommend storing everything in JSONL, with one training example per line.

Use a shared schema with a `task` field so that the same document can generate different training examples.

Here is a possible structure\:

```text
indic-ocr/
│
├── data/
│   ├── text/
│   │   ├── hindi.txt
│   │   └── english.txt
│   │
│   ├── crops/
│   │   ├── words/
│   │   ├── lines/
│   │   └── paragraphs/
│   │
│   ├── pages/
│   │   ├── synthetic/
│   │   ├── real/
│   │   └── degraded/
│   │
│   └── multipage/
│
├── annotations/
│   ├── stage1_language.jsonl
│   ├── stage2_alignment.jsonl
│   ├── stage3_documents.jsonl
│   ├── stage5_sft.jsonl
│   └── stage6_rl.jsonl
│
├── configs/
│
├── models/
│
└── evaluation/
    ├── golden_train.jsonl
    ├── golden_dev.jsonl
    └── golden_test.jsonl
```

Example stage-specific records\:

**Recognition**

```json
{
  "id": "word_001",
  "image": "crops/words/word_001.png",
  "task": "recognition",
  "target": "न्यायालय"
}
```

**Page parsing**

```json
{
  "id": "page_001",
  "image": "pages/page_001.png",
  "task": "page_parse",
  "target": {
    "blocks": [
      {
        "type": "heading",
        "bbox": [100, 50, 900, 120],
        "text": "प्रथम सूचना रिपोर्ट",
        "reading_order": 0
      },
      {
        "type": "paragraph",
        "bbox": [100, 140, 900, 350],
        "text": "...",
        "reading_order": 1
      }
    ]
  }
}
```

**Extraction SFT**

```json
{
  "id": "form_001",
  "image": "pages/form_001.png",
  "task": "extract_fields",
  "instruction": "Extract the fields from this form.",
  "target": {
    "document_type": "FIR",
    "fields": [
      {
        "name": "Police Station",
        "value": "Trichy",
        "status": "present",
        "source_bbox": [100, 200, 450, 250]
      }
    ]
  }
}
```

You can serialize these targets into model-specific conversational training formats when preparing the dataloader.

**One important dataset rule\:** Split by original document before creating crops or augmentations. Never put a crop from a test document into your training set.

Your proposed golden-set strategy uses test, development and tuning splits, with the tune split removed when the entire set is very small. That is a sensible safeguard. fileciteturn1file8

# 7\. How to evaluate every stage

I would establish explicit exit conditions rather than train for an arbitrary number of epochs.

| Stage | Evaluation | What proves success? |
|--------|--------|--------|
| Language pretraining | Held-out perplexity and translation quality | Language generation is competent |
| Visual alignment | CER and WER | Characters are recognized accurately |
| Page pretraining | CER, layout mAP, IoU and reading order | Full document reconstruction works |
| Document memory | Cross-page recall and correction precision | Memory helps without corrupting text |
| SFT | Field F1, JSON validity and task accuracy | Application tasks work reliably |
| RL | Reward and held-out OCR metrics | Genuine improvement without regression |

For your model, I would report separate scores for each script and document type.

Also measure the percentage of blank fields that are incorrectly populated. In document AI, that can be more important than a marginal improvement in overall OCR accuracy.

# 8\. What I would actually implement first

Here is how I would prioritize the work.

## Recommended experiment sequence

(Start here)

**Experiment 1 - Existing VLM + OCR adaptation**

- Select a pretrained multilingual VLM.
- Train on word and line image-text data.
- Progress to full-page OCR.
- Add the layout and extraction tasks.
- Run SFT on verified real documents.

Measure against your current LFM and other OCR baselines.

---

**Experiment 2 - Small model from scratch**

- Build a language tokenizer.
- Train a 100M-150M language/vision prototype.
- Validate the text pretraining and visual alignment.
- Compare dense and split encoder-decoder configurations.

Do not scale before it demonstrates real improvements.

---

**Experiment 3 - Architecture research**

- Add reference attention.
- Evaluate long-document inference.
- Experiment with MoE.
- Separately test latent memory.
- Only then investigate adaptive looping.

Treat every architectural change as an ablation.

This last point is especially important for Architecture v2. Your document proposes eight small experimental rungs, from a dense baseline through split attention, MoE and eventually latent memory and looping. I agree with that philosophy. The memory and looping components are hypotheses, not established OCR improvements. fileciteturn2file2

Also, I would not start with a 3B-6B MoE. Your own design places dense models below approximately 2B and reserves MoE for larger capacity requirements. First prove that the simpler model can read your scripts accurately. fileciteturn2file1

---

# 9\. My final recommendation: which training pipeline should you choose?

# Choose your starting point

I have an existing VLM and want excellent OCR for my language.

I want to develop my own OCR foundation model with a new tokenizer and language stack.

I want a pretrained vision encoder but a language model trained from scratch.

---

**My preference would be to complete the fine-tuning route first, then use that as your baseline for the from-scratch research.** Your from-scratch model should have to beat that baseline on unseen documents, not merely demonstrate that it can learn OCR.

One final distinction: the 850M model's approximately 7,000 GPU-hour budget and 105B-token training budget in your attached plan are estimates based on assumed throughput and data volume, not established requirements for an Indic OCR model. Do not commit to that allocation before the small-model experiments. fileciteturn1file8

The next valuable step is to turn these stages into a runnable training specification: dataset manifests, task-specific JSONL schemas, model initialization, loss functions, training configurations and evaluation gates for each stage.

## Prompt:
11/14/2026, 2:56:22 PM

so for the last goal to reconstruct an proper grounded translated PDF back to the user is the goal and i was planning a model not more that 3b to facilitate the edge user and training needs.

Need one clarification where to start and also i need the OCR to do the extraction plus translation together as these ocr have a llm decoder and will be less a externel step and a good way to train.

so this reconstruction need to be done vian an external harness or can be done by inducing the doc intelligence into the model? also list some resources for me to read on undertand the importants of differenct ocr concepts and diff models like unlimited ocr, paddle and research papers on diff vision encoders particulary trained for text or we need to train one since all these encoders are trained on variety of different tasks.

nbeed a complete plan here.

## Response:
11/14/2026, 2:56:56 PM · gpt-6-t-mini

For your end goal, I’d separate **document intelligence** from **PDF rendering**, but not necessarily separate OCR from translation. The model can do recognition, field extraction, translation and layout-aware planning in one multimodal pass; a deterministic renderer should then turn its grounded output into the final PDF. I’m checking the latest model papers and your two design notes so the plan can be concrete and realistic under the 3B ceiling.

> **Comparing OCR encoders**
>
> I’m outlining OCR-focused vision encoders, including Donut, TrOCR, PARSeq, SVTR, and PaddleOCR variants. I’ll avoid false precision, distinguish recognition from document understanding, and verify newer claims before presenting them.
>
> **Compared OCR encoders**
>
> 
>
> **Analyzed OCR translation rendering architecture**
>
> 
>
> **Verifying model details**
>
> I’m checking primary sources for newer OCR architectures and capabilities, especially whether long-context processing, layout grounding, and coordinate output are genuinely supported. I’ll separate confirmed features from assumptions before recommending a design.
>
> **Verified model details**
>
> 
>
> **Shaping the architecture**
>
> I’m balancing edge deployment, multilingual OCR, translation, and layout fidelity. A dense 1.2–1.8B design with a document-specialized vision encoder seems more practical than a larger sparse model, while structured outputs preserve alignment and reading order.
>
> **Designing multilingual training**
>
> I’m planning joint OCR, translation, and document-structure tasks. The model should preserve identifiers, dates, layout, reading order, confidence, and source-to-translation alignment, using verified parallel data plus carefully rendered synthetic documents.
>
> **Planning the document pipeline**
>
> I’m prioritizing deterministic rendering and validation over asking the model to handle exact typography. The harness should preserve text, geometry, and visuals, while adaptive sizing or reflow prevents overflow without silently deleting content.
>
> **Shaped architecture, multilingual training, and document pipeline planning**
>
> 
>
> **Comparing document models**
>
> I’m narrowing the shortlist to PaddleOCR-VL, HunyuanOCR, DeepSeek-OCR2, and similar compact systems. I’ll verify multilingual coverage carefully, especially for Indic scripts, before recommending a starting checkpoint.
>
> **Refining model selection**
>
> I’m treating Indic support as a language-dependent claim rather than assuming it. I’ll compare compact document models against modular OCR pipelines, then recommend validation on the exact scripts, layouts, and translation directions required.
>
> **Balancing deployment constraints**
>
> I’m favoring a dense 1.3–2B model over a 3B mixture-of-experts design because edge devices still store every expert. I’d fine-tune a compact multilingual decoder with a document-specialized visual encoder, then quantize or distill it.
>
> **Choosing vision backbones**
>
> I’m comparing general, document-specific, and OCR-specific encoders, including NaViT, SigLIP2, C-RADIO, SVTR, TrOCR, and DiT. I’ll test multi-scale resolution carefully because Indic glyph detail can be lost through aggressive patch merging.
>
> **Compared document models and selected vision backbones within deployment constraints**
>
> 
>
> **Validating the implementation**
>
> I’m checking model papers and PDF-rendering references to ground the plan. The key split remains recognition versus document understanding, with translation and layout reconstruction handled as coordinated but separately testable stages.
>
> **Validated the implementation**
>
> 
>
> **Planning the complete system**
>
> I’m assembling the final blueprint around a grounded Document IR, a compact multilingual vision-language model, verified parallel data, and a deterministic PDF renderer. I’ll separate OCR, translation, layout, evaluation, and deployment so failures remain diagnosable.
>
> **Defining the build**
>
> I’m organizing the plan into baseline evaluation, OCR adaptation, layout parsing, translation training, joint document translation, PDF rendering, and deployment. This keeps errors diagnosable and lets the system improve without sacrificing transcription fidelity.
>
> **Planned and defined the complete system build**
>
> 
>
> **Planning the document translator**
>
> I’m evaluating compact OCR-VLMs, multilingual translation, structured document output, and PDF reconstruction. I’ll prioritize a dense edge-friendly model, page-batched processing, grounded source geometry, and explicit translation-quality validation.
>
> **Refining the architecture**
>
> I’m treating the current design as a strong starting point, but I’m simplifying it first: a dense 1–1.8B model is more edge-friendly than a 3B MoE whose inactive experts still consume storage.
>
> **Planned the document translator and refined its architecture**
>
> 
>
> **Designing the build plan**
>
> I’m turning your proposed system into a practical sequence: benchmark compact vision-language models first, preserve geometry in a document representation, train translation separately, then join OCR, translation, and PDF rendering.
>
> **Designed the build plan**
>
> 
>
> **Searched 19 websites**
>
> 
>
> Worked for 13m 41s

# 1\. The product architecture I would target

Let's define the actual product, because it affects almost every training decision.

Your input is an original PDF in one language. Your output is a new PDF in another language, with\:

- All text correctly recognized, including tables, forms and footnotes.
- Translated content, without changing its factual meaning.
- Every translated block connected to its original source location.
- Original visual structure preserved as much as possible.
- Correct fonts, spacing, line wrapping, alignment and page dimensions.
- No missing content, invented content or silently discarded fields.

This is not just OCR. It is **grounded document translation and reconstruction**\.

I would call the intermediate representation (IR) a `Document IR`\. It is the central contract between the AI model and the PDF renderer.

## 1.1 Model and renderer responsibilities

## Proposed end-to-end architecture

**Original PDF**

Pages, text, images, fonts and vector objects

**PDF ingestion and visual preprocessing**

Native PDF text extraction · page rendering · image quality · coordinates · embedded fonts

(Your OCR + Translation VLM · ≤3B)

**Multimodal Document Intelligence**

OCR recognition

Exact source text

Layout understanding

Blocks and geometry

Semantic extraction

Fields and relationships

Translation

Source-grounded target text

**Output: Document IR**

Source text + translated text + source boxes + hierarchy + confidence + layout constraints

**External reconstruction harness**

Font selection · text shaping · translation layout · overflow detection · table recreation · image placement · pagination

**Translated, grounded PDF**

Searchable, selectable and visually reconstructed

My recommendation is to train the model to produce the Document IR, and build the renderer as a separate deterministic application component.

This is not an unnecessary external AI step. The renderer does not need another LLM. It just executes the model's structured layout decisions.

## 1.2 Why I would not put PDF reconstruction entirely inside the model

There are three separate problems.

**Problem A: Understanding the page**

The model needs to identify headings, paragraphs, tables, images, signatures, field labels, relationships and coordinates.

This is a learning problem. Train the VLM.

**Problem B: Translating the content**

The model needs to translate a paragraph or table cell without losing meaning, preserving numbers, entities and technical terminology.

This is also a learning problem. Train the VLM.

**Problem C: Physically rebuilding the PDF**

The model needs to calculate font sizes, line wrapping, text baselines, box dimensions, font fallback and image positions.

These are engineering and rendering problems. They are better handled by code.

A model can suggest that a translated paragraph needs a larger box or a smaller font. But a proper renderer needs to measure the actual glyph widths and verify that the text fits.

For example, a Hindi paragraph translated to English could become 30% longer. The original text box may no longer fit.

You need an overflow solver that can\:

1. Try the original font size.
2. Reduce the font within a defined limit.
3. Adjust line spacing.
4. Expand the block where the surrounding layout permits it.
5. Reflow the page when there is insufficient space.
6. Record a layout warning when faithful preservation is impossible.

This is how I would approach the final PDF reconstruction.

**Two output modes are essential\:**

| Mode | Behaviour |
|--------|--------|
| Faithful layout | Preserve page dimensions, geometry and original visual hierarchy, with controlled adjustments |
| Reflowed translation | Preserve semantic hierarchy, tables and images, but allow different page breaks and text geometry |

Do not force both into the same rendering strategy. Some source documents simply cannot fit their translations into the original boxes without compromising readability.

# 2\. How to train OCR and translation together

I agree with your reasoning here.

A vision encoder followed by an autoregressive language decoder can recognize text, understand context and generate translations. You don't need a separate inference-time translation LLM.

However, I would not train it to produce only the translated document. That creates a serious problem: you lose the ability to distinguish what was actually printed from what the model inferred or translated.

Instead, train the decoder with **joint OCR + translation + structure supervision**\.

## 2.1 Three tasks, one shared model

I would define three model modes.

**Mode A - OCR**

Input: source page image

```json
{
  "task": "ocr",
  "source_text": "भारत सरकार गृह मंत्रालय"
}
```

---

**Mode B - Translation**

Input: source image or recognized source text

```json
{
  "task": "translate",
  "source_language": "hi",
  "target_language": "en",
  "translation": "Government of India, Ministry of Home Affairs"
}
```

---

**Mode C - Document translation**

(Your main product task)

```json
{
  "task": "document_translate",
  "source_text": "भारत सरकार गृह मंत्रालय",
  "translated_text": "Government of India, Ministry of Home Affairs",
  "bbox": [100, 40, 850, 100],
  "block_type": "heading",
  "reading_order": 0
}
```

All three modes use the same vision encoder, language decoder and shared parameters.

Their supervision is different, so I would train all three.

## 2.2 The dataset you need for joint training

This is an important addition to your existing data-engineering plan.

You need four kinds of training data\:

| Dataset | Image | Original text | Translation | Geometry |
|--------|--------|--------|--------|--------|
| OCR | Yes | Yes | No | Optional |
| Text translation | No | Yes | Yes | No |
| Synthetic document translation | Yes | Yes | Yes | Yes |
| Real document translation | Yes | Verified | Human-verified | Yes |

For instance, suppose your source document contains\:

```text
मूल राशि: ₹५०,०००
दिनांक: १४/११/२०२६
```

You want the model to return\:

```json
{
  "source_text": "मूल राशि: ₹५०,०००\nदिनांक: १४/११/२०२६",
  "translated_text": "Original Amount: ₹50,000\nDate: 14/11/2026"
}
```

You need to train explicit preservation rules for numbers, dates, identifiers, names and amounts. Translation must never silently change them.

### Where to get translation training data

For Indic languages, I would start with\:

- [IndicTrans2 paper](https://arxiv.org/abs/2305.16307) - multilingual Indic-English translation and the BPCC parallel corpus. BPCC contains approximately 230 million bitext pairs, including a human-curated component. \[[IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages](https://arxiv.org/abs/2305.16307?utm_source=chatgpt.com); [GitHub - AI4Bharat/IndicTrans2: Translation models for 22 scheduled languages of India](https://github.com/ai4bharat/IndicTrans2?utm_source=chatgpt.com)\]
- [BPCC dataset](https://huggingface.co/datasets/ai4bharat/BPCC) - parallel text data for synthetic document generation.
- [IndicTrans2 implementation](https://github.com/AI4Bharat/IndicTrans2?utm_source=chatgpt.com) - a valuable teacher and translation baseline.

Use BPCC to produce text translation pairs. Then render their source-language sentences into document images using your renderer, keeping the corresponding translated sentences as ground truth.

For real documents, curate a smaller high-quality collection with verified translations, especially for legal, financial and government terminology.

## 2.3 How I would train the joint model

Don't just concatenate every target into one enormous output and hope the model discovers how they relate.

Train a sequence of tasks\:

1. Source image → exact source transcription.
2. Source text → target translation.
3. Source image → source transcription + translated transcription.
4. Source image → structured bilingual Document IR.
5. Multi-page image input → complete structured bilingual document.

A practical joint loss is\:

$$
\mathcal{L} =
\lambda_{\text{ocr}}\mathcal{L}_{\text{ocr}}
+\lambda_{\text{mt}}\mathcal{L}_{\text{translation}}
+\lambda_{\text{layout}}\mathcal{L}_{\text{layout}}
+\lambda_{\text{joint}}\mathcal{L}_{\text{joint}}
$$

You can also add field extraction and confidence losses.

Start by giving OCR and translation substantial individual training weight. Introduce the joint document task after the model is competent at both. Otherwise, a weak recognizer can hide behind a fluent translation decoder.

A useful additional curriculum is\:

- First, translate clean source text without images.
- Then translate text generated by the OCR model, including realistic OCR noise.
- Finally, translate directly from full document images while preserving the intermediate source transcription.

This teaches the model to deal with recognition mistakes without learning to silently invent corrections.

---

# 3\. What should the Document IR contain?

This is the most important interface in the whole system.

I would use a richer representation than Markdown or plain extraction JSON.

For example\:

```json
{
  "document_id": "doc_0001",
  "source_language": "hi",
  "target_language": "en",
  "pages": [
    {
      "page_number": 1,
      "page_size": {
        "width_pt": 595.28,
        "height_pt": 841.89
      },
      "blocks": [
        {
          "id": "block_001",
          "type": "heading",
          "reading_order": 0,
          "source": {
            "text": "भारत सरकार",
            "language": "hi",
            "confidence": 0.99
          },
          "translation": {
            "text": "Government of India",
            "language": "en"
          },
          "geometry": {
            "bbox": [100, 50, 850, 120],
            "coordinate_system": "normalized_1000"
          },
          "style": {
            "role": "heading",
            "alignment": "center",
            "font_weight": "bold"
          },
          "constraints": {
            "preserve_content": true,
            "allow_resize": true
          },
          "evidence": {
            "page": 1,
            "bbox": [100, 50, 850, 120]
          }
        }
      ]
    }
  ]
}
```

Illustrative schema; production fields should be versioned and validated.

Notice something important: source content and target content are separate.

This enables\:

- Auditing translations against original evidence.
- Rebuilding the source document if needed.
- Replacing a translation without running OCR again.
- Producing bilingual documents.
- Rendering translated and original text side by side.
- Identifying which source region generated each translated block.

The model predicts content and logical layout. The harness validates geometry, typography and rendering.

For tables, extend this with table IDs, row/column indices, cell bounding boxes, merged-cell relationships and header associations. Don't flatten tables into paragraphs.

For mathematical content, preserve the formula separately using LaTeX or a suitable structured representation.

---

# 4\. Which vision encoder should you use?

This is probably the most important architectural research question for your project.

Your observation is correct: a generic vision encoder is not necessarily optimal for OCR.

But there are three different kinds of visual representation learning, and you should understand the distinction.

## 4.1 Three categories of vision encoders

**A. General-purpose vision-language encoders**

[![多模态模型分析-面壁MiniCPMV | Henry-Z](https://images.openai.com/static-rsc-4/NsOAmWVySYF62stJFKhPtndjq2mPwjbbMakNeuCRtiVzpLGihZZP6pq5bduOD-qIZp3XyvlgOKBGKcPfbeH0J-4Z03mOAp18RyKaTqz8GSprTInQ19-AtubFUbokH2p9jA1i9Dp4H1CHnztJlj_kIoI9WGS1yUBK6ldJSU3cRhGwQOwJLSYtN3ujMT9jg3uT?purpose=fullsize)](https://zgh551.github.io/2024/12/14/multi-modal-MiniCPMV/)

Examples: SigLIP / SigLIP2, CLIP, C-RADIO.

They learn semantic visual representations, image-text correspondence, and sometimes localization or dense features.

**Limitation\:** Semantic understanding does not automatically mean excellent recognition of small character strokes.

---

**B. Document-specific vision encoders**

[![\[ACM MM 2022\] DiT: 基于Transformer的文档图像自监督预训练方法 - 智源社区](https://images.openai.com/static-rsc-4/u8VXYCFBeRYpwyF9KFsW7lIL59z1u2OXxMNrVn8XCXsCzaLHzStkAoMRahF7TwS9qot3HkOBs3tAluSGNiWqYKMGAN5HMVBeRFTRKxJiR-eX9V40ooXV5kQ0F3BVAiC-1K3G1u5mFbM_6QdASkQLFz6X7LX4wfHjM7sQr85PfmjbWtOQnFEv5hOXKOG093nD?purpose=fullsize)](https://hub.baai.ac.cn/view/21616)

Example: DiT (Document Image Transformer).

They are trained on document images and learn page structure, visual regions, tables and layout features.

**Limitation\:** Document-level visual understanding is not the same as recognizing exact Unicode characters.

---

**C. Text-recognition-specific visual encoders**

[![PP-OCRv3技术报告 - PaddleOCR 文档](https://images.openai.com/static-rsc-4/wMv5uI9c6ciEwx0aZVXynE6BcDhfhjOxmJEvIiek1MppUq9ej80UhjAjzK_wJ5DSWpxCqXBIfkfBgWBQy_EIv1QHz1zTkm9YL9l6sAf4KM7A6DW8SkcVOGIunQpyEa34ty-oSKl85MTKhRLh68C-DbiRYJ9n7R2VRVe6BzcCJ3tSoXsav_oxDvYDnVwzii4H?purpose=fullsize)](https://www.paddleocr.ai/v2.9/ppocr/blog/PP-OCRv3_introduction.html)

Examples: SVTR, TrOCR visual encoder, PP-OCR recognition backbones.

They focus on character components, local strokes, character relationships and text-line representations.

**Limitation\:** They usually solve text recognition, not the full semantic and structural understanding of an entire document.

## 4.2 Papers you should understand

| Paper / architecture | What to learn from it | Relevance |
|--------|--------|--------|
| [TrOCR](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com) | Pretrained image Transformer + text Transformer for OCR | Very high |
| [SVTR](https://arxiv.org/abs/2205.00159?utm_source=chatgpt.com) | Text-specialized visual representation learning | Very high |
| [PARSeq](https://arxiv.org/abs/2207.06966?utm_source=chatgpt.com) | Combining visual recognition and autoregressive language modelling | Very high |
| [ABINet](https://arxiv.org/abs/2103.06495?utm_source=chatgpt.com) | Visual and language modelling with iterative correction | High |
| [DiT](https://arxiv.org/abs/2203.02378?utm_source=chatgpt.com) | Self-supervised document image pretraining | Very high |
| [Donut](https://arxiv.org/abs/2111.15664?utm_source=chatgpt.com) | End-to-end document understanding without a separate OCR engine | Very high |

Here's what I would take away from them.

**TrOCR\:** This is a particularly valuable paper for your research. It uses large-scale synthetic and real text-line data. Its first pretraining dataset contains 684 million text lines extracted from 2 million digital-born PDF pages. It also uses synthetic handwritten text lines and real receipt data. \[[TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com); [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com)\]

This gives you a useful idea: digital-born PDFs can become a large source of aligned visual-text training data.

**SVTR\:** Instead of using a general image representation, it explicitly models local and global character relationships. This makes it relevant to Devanagari recognition, where individual strokes and character components matter considerably. \[[SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159?utm_source=chatgpt.com); [SVTR: Scene Text Recognition with a Single Visual Model](https://ijcai.org/proceedings/2022/124?utm_source=chatgpt.com)\]

**DiT\:** It uses self-supervised masked image modelling on document images. It is useful for visual document representations, layout analysis, table detection and text detection. It is not a replacement for a text recognizer. \[[DiT: Self-supervised Pre-training for Document Image Transformer](https://arxiv.org/abs/2203.02378?utm_source=chatgpt.com); [DiT: Self-supervised Pre-training for Document Image Transformer](https://arxiv.org/abs/2203.02378?utm_source=chatgpt.com)\]

**Donut\:** It demonstrates that you can combine document image perception and structured output generation in an OCR-free architecture. Its synthetic data generation approach is particularly relevant to you. \[[OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664?utm_source=chatgpt.com); [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664?utm_source=chatgpt.com)\]

## 4.3 What about PaddleOCR-VL and HunyuanOCR encoders?

These are closer to what you want to build.

**PaddleOCR-VL**

(Strong first benchmark for Hindi)

[![Modern OCR for the Large Language & Vision Model Era |](https://images.openai.com/static-rsc-4/f4Htkou-tfx7ZyxsTYrmxLiZnk3XltU5sc-ltWHKkklAYijl8iU75P7KD-8UMzt_hrrpp6lC7Th2Mb-S80D6DICWqYlVvBdSwiTXVCKl33WLfQon6y0GtU2e6CskFvoXLneqc5A_1yMi5IFDQ1ElFU7Mr-WYdMQxUR5-m4cngONg-l0Wacukb-TskGT918A1?purpose=fullsize)](https://roots-automation.github.io/roots-labs/post/ocr/)

- NaViT-style dynamic-resolution vision encoder.
- Pretrained visual backbone.
- MLP connector.
- ERNIE-4.5-0.3B language model.
- Approximately 0.9B parameters.
- Explicit support for Hindi and other Devanagari languages.

The original PaddleOCR-VL paper describes 109 supported languages, including Hindi, Marathi, Nepali and several other Devanagari languages. \[[PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model](https://arxiv.org/abs/2510.14528?utm_source=chatgpt.com); [PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model](https://arxiv.org/abs/2510.14528?utm_source=chatgpt.com)\]

[Read PaddleOCR-VL](https://arxiv.org/abs/2510.14528)

---

**HunyuanOCR**

[![Hunyuan OCR: End-to-End OCR Expert Vision-Language Model](https://images.openai.com/static-rsc-4/j9oox0XAgd0YMhIw2YO9GL4KF3LRnfSVOZxwUQya0SDKCi4pxK_s04AiYTCwxFXW0LZRGWFYEfqv0clqySSoY-Y_hF--J-Ij5PPXSnBzfa_jcgx-9gHSbcMBqkua5PaeE5UO4zfzlIGND3lmq6N1C9i771eYNdzUuxglFZfLIwZEDcBIoy7Z6OiS4neb-hFS?purpose=fullsize)](https://hunyuanocr.org/)

- Approximately 0.4B native-resolution ViT, initialized from SigLIP2.
- Approximately 0.5B language model.
- Learnable MLP connection.
- Integrated OCR, spotting, information extraction, parsing and image translation.

This model is particularly important for your product because it demonstrates OCR and text-image translation in a compact VLM. Its paper also reports that translation remains weaker than recognition, which is a warning that you should explicitly train and benchmark the translation capability. \[[HunyuanOCR Technical Report](https://arxiv.org/abs/2511.19575?utm_source=chatgpt.com); [HunyuanOCR Technical Report](https://arxiv.org/abs/2511.19575?utm_source=chatgpt.com)\]

[Read HunyuanOCR](https://arxiv.org/abs/2511.19575)

## 4.4 Should you train your own vision encoder?

My recommendation is\:

(Do not train one from random initialization initially.)

Instead, conduct a controlled encoder experiment.

| Experiment | Vision encoder | Purpose |
|--------|--------|--------|
| A | NaViT-style dynamic resolution | Strong full-page baseline |
| B | C-RADIOv4 | General visual representation baseline |
| C | DiT | Document-specific representation baseline |
| D | SVTR-derived text encoder | Character and line recognition baseline |

These are not necessarily drop-in compatible encoders. You may need different feature projections, patch sizes or input preprocessing.

Your Architecture v2 selects C-RADIOv4-SO400M, but your Tab 1 correctly identifies the absence of an OCR benchmark for it. I would not commit to it without running the experiments. fileciteturn0file1L19-L40

### The experiment that decides your encoder

Build a fixed dataset of difficult script-specific examples\:

- Devanagari conjuncts.
- Matras and vowel signs.
- Nukta characters.
- Similar-looking characters.
- Very small printed text.
- Mixed English and Hindi.
- Low-resolution scans.
- Skewed pages.
- Stamps and overlapping text.

Train the same decoder architecture against each candidate.

Measure CER, WER, text localization and recognition accuracy at equivalent compute.

Also test visual compression at 4×, 8× and 16×. Your Architecture v2 proposes 16× reduction, but I would retain a higher-resolution path until you verify that it preserves the smallest Indic glyph details.

**My preferred direction\:** a document-capable, dynamic-resolution vision encoder with OCR-specific continued pretraining. This gives you the best combination of full-page structure and fine-grained recognition without creating an entirely new vision backbone.

---

# 5\. Which OCR models should you study and reproduce?

I would divide your research into four groups.

| Model | Main innovation | What you should learn |
|--------|--------|--------|
| PaddleOCR | Modular detection, recognition and document parsing | Production OCR pipeline |
| PaddleOCR-VL | Compact multilingual end-to-end VLM | Your primary architecture baseline |
| HunyuanOCR | Unified perception and semantic tasks | OCR + translation + extraction |
| DeepSeek-OCR | Compressed visual tokens and LLM decoding | Vision compression and large-scale OCR data |
| DeepSeek-OCR 2 | Visual causal flow | Reordering visual tokens |
| Unlimited-OCR | Reference Sliding Window Attention | Long-document generation |
| Unlimited-OCR 2 | Sparse Reference Sliding Window Attention | Long-document scaling |

### A. PaddleOCR: don't confuse it with PaddleOCR-VL

The classic PaddleOCR is a modular system.

Its major components include\:

- PP-OCR: detection and text recognition.
- PP-StructureV3: document layout, reading order, tables, formulas and charts.
- PP-ChatOCRv4: key-information extraction.

It is valuable because each component can be trained and deployed independently. \[[PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595?utm_source=chatgpt.com); [PP-StructureV3 Introduction](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html?utm_source=chatgpt.com)\]

Its newer documentation also covers PP-OCRv6, with a unified PPLCNetV4-based recognition and detection backbone. Check each model's supported languages separately rather than assuming all PaddleOCR variants cover Devanagari. \[[Usage Tutorial](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html?utm_source=chatgpt.com); [Research Papers](https://www.paddleocr.ai/main/en/papers.html?utm_source=chatgpt.com)\]

Useful reading\:

- [PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595)
- [PaddleOCR Documentation](https://www.paddleocr.ai/main/en/?utm_source=chatgpt.com)
- [PaddleOCR Research Papers](https://www.paddleocr.ai/main/en/papers.html?utm_source=chatgpt.com)

### B. DeepSeek-OCR and DeepSeek-OCR 2

Study these to understand how the visual encoder can be optimized specifically for document reading.

DeepSeek-OCR 2 is particularly interesting because it combines a SAM ViT-B visual encoder, a Qwen2 hybrid attention encoder and a DeepSeek language model. Its key innovation is visual token reordering rather than treating the document as an ordinary raster-scan sequence. \[[DeepSeek-OCR 2: Visual Causal Flow](https://arxiv.org/abs/2601.20552?utm_source=chatgpt.com); [DeepSeek-OCR-2](https://huggingface.co/docs/transformers/model_doc/deepseek_ocr2?utm_source=chatgpt.com)\]

Resources\:

- [DeepSeek-OCR](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)
- [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552)
- [DeepSeek-OCR 2 implementation](https://github.com/deepseek-ai/DeepSeek-OCR-2?utm_source=chatgpt.com)

### C. Unlimited-OCR and Unlimited-OCR 2

These are especially relevant to your whole-PDF architecture, but I would not make their attention mechanism your first implementation.

**Unlimited-OCR\:**

Regular autoregressive attention keeps increasing its KV cache as output tokens grow. Reference Sliding Window Attention retains the full visual/prompt prefix and a fixed number of recent generated tokens.

Its cache is therefore bounded with respect to generated output length\:

$$
C(T)=L_{\text{visual}}+\min(T,n)
$$

where $n$ is the output window, typically 128 tokens. \[[Unlimited OCR Works](https://arxiv.org/abs/2606.23050?utm_source=chatgpt.com); [Unlimited OCR Works](https://arxiv.org/abs/2606.23050?utm_source=chatgpt.com)\]

Paper: [Unlimited OCR Works](https://arxiv.org/abs/2606.23050)

**Unlimited-OCR 2\:**

The problem with the original approach is that visual reference context still grows with page count.

The second version introduces Sparse Reference Sliding Window Attention, using\:

- Page Gate.
- Token Sieve.
- Local sliding window.
- Fixed-budget remote visual references.

Its objective is to keep active visual context and memory bounded even as the document grows to hundreds of pages. \[[Unlimited OCR Works 2](https://arxiv.org/abs/2610.39308?utm_source=chatgpt.com); [Unlimited OCR Works 2](https://arxiv.org/abs/2610.39308?utm_source=chatgpt.com)\]

Paper: [Unlimited OCR Works 2](https://arxiv.org/abs/2610.39308)

My recommendation: start with conventional page-by-page processing and a document-level state maintained by your harness. Introduce R-SWA or SR-SWA only after multi-page context becomes a demonstrated bottleneck.

You don't need to reproduce a hundred-page attention architecture to translate a typical ten-page PDF reliably.

---

# 6\. The model I would actually build under 3B

Looking at your two architecture proposals, I would make one significant adjustment.

Your Architecture v2 proposes a 3B MoE model with approximately 0.71B active parameters per generated token. It has 412M vision parameters and a 24-layer language stack with 28 experts per layer. fileciteturn0file0L54-L90

That is an interesting research architecture, but there is an important distinction for edge deployment\:

**Active parameters determine much of the computation; total parameters determine the weight-storage requirement.**

Your 3B MoE model still has roughly 3B parameters that need to be accessible during inference. It is not equivalent to a 700M dense model in memory requirements.

For your first product, I would prefer a dense architecture.

## 6.1 Proposed architecture

(Recommended V1 research target)

# Indic Document Translation VLM

# 1.2B-1.8B

Proposed total parameter budget; not a published architecture.

**Vision encoder**

Document-capable, dynamic resolution **300-500M**

**Visual connector / resampler**

MLP / patch merging **30-60M**

**Multilingual language decoder**

Translation + OCR + reasoning **800M-1.2B**

**Additional heads and embeddings**

Confidence / auxiliary outputs **10-40M**

---

**Supported capabilities**

(OCR) (Translation) (Layout) (Spotting) (Extraction) (Tables) (Grounding)

You can later scale the decoder towards 2B or 3B if your evaluation shows that translation or reasoning is the bottleneck.

My preference\:

| Version | Model size | Purpose |
|--------|--------|--------|
| V1 | 0.9-1.2B | Edge-friendly initial implementation |
| V2 | 1.5-1.8B | Stronger translation and document understanding |
| V3 | 2-3B | Higher-quality variant, if justified |

These are proposed development targets, not guaranteed performance tiers.

For inference optimization, evaluate INT8 and 4-bit quantization, with special attention to OCR accuracy on small characters. Also benchmark real device memory, including the vision tower, activations and KV cache.

# 7\. Complete training plan

Now let's put everything together into an actual development sequence.

I would modify your existing seven-stage plan slightly because you're starting from a pretrained model rather than necessarily building the language model from scratch.

# Indic Document Translation training roadmap

Recommended order for a pretrained VLM, with optional foundation-model experiments later.

1. **Stage 0 - Data engine and evaluation**
   
   (No training initially)
   
   Build PDF ingestion, synthetic rendering, annotation, translation alignment and the golden test set.
   
   **Deliverable\:** A validated dataset and baseline benchmark.
2. **Stage 1 - OCR visual adaptation**
   
   (Vision-language continued training)
   
   Train on word, line, paragraph and mixed-script images.
   
   **Data\:** Real and synthetic image-text pairs.
   
   **Train\:** Connector, then language adapter, followed by selected vision layers.
   
   **Deliverable\:** Accurate target-language recognition.
3. **Stage 2 - Document understanding**
   
   (Multimodal continued pretraining)
   
   Train full-page transcription, layout, reading order, spotting, region reading, tables and grounding.
   
   **Data\:** Full document images with structural annotations.
   
   **Deliverable\:** Source-document reconstruction in Document IR.
4. **Stage 3 - Translation adaptation**
   
   (Text-only translation training)
   
   Train source-to-target translation using BPCC and high-quality bilingual data.
   
   **Data\:** Parallel text, back-translation and verified domain translations.
   
   **Deliverable\:** Reliable multilingual translation capability.
5. **Stage 4 - Joint OCR + translation**
   
   (Core research stage)
   
   Train the model to recognize, understand, extract and translate each document block jointly.
   
   **Data\:** Source document images with bilingual Document IR ground truth.
   
   **Deliverable\:** Image-to-translated-document generation.
6. **Stage 5 - Application SFT**
   
   Train with actual product instructions, legal forms, financial forms, government documents and other target domains.
   
   Add blank-field handling, uncertainty and schema validation.
   
   **Deliverable\:** Stable product-facing outputs.
7. **Stage 6 - Preference optimization / RL**
   
   Optional. Improve text spotting, parsing, translation faithfulness, uncertainty and layout consistency with verifiable rewards.
   
   **Deliverable\:** Improved reliability on challenging samples.
8. **Stage 7 - Long-document extension**
   
   Only when the previous stages establish that page-by-page processing is insufficient.
   
   Evaluate R-SWA, SR-SWA or your document-memory approach.
   
   **Deliverable\:** Scalable multi-page inference.
9. **Stage 8 - Edge optimization**
   
   Quantization, distillation, attention optimization, batch sizing and document-level memory management.
   
   **Deliverable\:** Deployable model within your edge resource budget.

Notice that there is no full language-model pretraining stage here.

You can skip that if you initialize from an existing multilingual decoder with sufficient language capabilities. Your original Tab 1 plan includes a full text-pretraining stage because it assumes the language stack is new. fileciteturn0file1L6-L16

If you eventually train that stack from scratch, restore that stage.

## 7.1 Dataset matrix

This is the dataset engineering work I would prioritize.

| Training stage | Dataset you need | Ground truth |
|--------|--------|--------|
| OCR adaptation | Words, lines, paragraphs | Exact Unicode transcript |
| Document understanding | Full-page PDF/image | Blocks, polygons, text, reading order |
| Layout | Full-page images | Layout classes and boxes |
| Tables | Tables and forms | Cells, spans, rows, columns |
| Translation | Parallel Indic-English text | Source and translation |
| Joint training | Original document images | Bilingual Document IR |
| SFT | Real domain documents | Verified extraction and translation |
| RL | Challenging document samples | Checkable rewards |
| Long context | Multi-page documents | Repeated entities, cross-page relationships |

I would build three distinct data generators.

**Generator 1 - OCR renderer**

Use text corpora and font libraries to generate\:

- Word crops.
- Line images.
- Paragraph images.
- Tables.
- Full pages.
- Degraded scans.

Keep exact source transcription and geometry.

**Generator 2 - Document translation renderer**

Take parallel sentences from BPCC, create structured documents, render the source language and store the translated version against every block.

This generates virtually unlimited layout variations without requiring manual page translation for every synthetic sample.

**Generator 3 - Real document annotation**

Collect real documents. Extract embedded PDF text where available and use multiple OCR/VLM teachers to create initial annotations.

Keep model-generated labels only after validation. Have humans verify the most challenging and important documents.

Your existing data plan already identifies the key building blocks: Sangraha, BPCC, Mozhi, IIIT-INDIC-HW-WORDS, IndicSTR12, IndicDLP and synthetic full-page generation. fileciteturn0file1L62-L87

## 7.2 Should you use RL?

I would use SFT as the foundation and RL as a final refinement.

There is a good precedent here: HunyuanOCR describes four pretraining/application stages followed by GRPO. Its reported results include improvements in spotting, parsing, information extraction and text-image translation. \[[HunyuanOCR Technical Report](https://arxiv.org/abs/2511.19575?utm_source=chatgpt.com); [HunyuanOCR Technical Report](https://arxiv.org/abs/2511.19575?utm_source=chatgpt.com)\]

PaddleOCR-VL-1.6 is also worth studying because it identifies poorly optimized regions, refines training data and uses progressive post-training with reinforcement learning. \[[PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training](https://arxiv.org/abs/2606.03264?utm_source=chatgpt.com)\]

For your model, I would calculate separate rewards\:

| Objective | Reward mechanism |
|--------|--------|
| OCR | Character edit similarity |
| Text spotting | Text correctness + box overlap |
| Layout | Detection and reading-order accuracy |
| Translation | Reference-based COMET + verified human evaluation |
| Extraction | Field-level F1 |
| Grounding | Correct page and source-box correspondence |
| Uncertainty | Correctly abstaining on genuinely unreadable content |

Do not optimize one combined reward without first verifying that its components actually improve.

---

# 8\. PDF reconstruction harness: how to implement it

Now to your second architectural question.

I would implement the rendering harness as an independent service or library. It should consume the Document IR and create PDFs.

You don't need an additional LLM here.

## 8.1 Recommended components

| Component | Suggested technology | Responsibility |
|--------|--------|--------|
| PDF parsing | PyMuPDF | Read PDF metadata, embedded text and positions |
| PDF rendering | PyMuPDF / MuPDF | Render pages at suitable resolutions |
| OCR model | Your VLM | Recognition and translation |
| Document validation | Pydantic / JSON Schema | Validate Document IR |
| Layout engine | Custom Python module | Resolve translated blocks and geometry |
| Font shaping | HarfBuzz | Correct complex-script shaping |
| PDF reconstruction | PyMuPDF | Compose text, images and vector elements |
| Image cleanup | OpenCV / inpainting model | Remove original text from scanned backgrounds |
| Quality validation | PyMuPDF + image rendering | Check output and visual layout |

One important detail for Indic scripts: use a shaping engine rather than simply placing Unicode characters with basic PDF text insertion. PyMuPDF's `insert_htmlbox` supports multiple fonts, mixed scripts and complex text shaping through HarfBuzz. It also supports text wrapping inside defined rectangles. \[[Text](https://pymupdf.readthedocs.io/en/latest/recipes-text.html?utm_source=chatgpt.com); [FAQ](https://pymupdf.readthedocs.io/en/latest/faq/index.html?utm_source=chatgpt.com)\]

Useful documentation: [PyMuPDF text insertion and formatting](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)\.

## 8.2 Rendering process

## Document reconstruction workflow

1. **Extract original page geometry**
   
   Page dimensions, text spans, positions, images, vector objects and font information.
2. **Receive bilingual Document IR**
   
   Source text, translation, blocks, reading order, geometry and constraints.
3. **Validate source coverage**
   
   Verify that all original blocks have been recognized and that no content has disappeared.
4. **Resolve layout**
   
   Map source boxes to the original page coordinate system. Measure the translated text using the selected font.
5. **Resolve overflow**
   
   Adjust typography or reflow content within the chosen preservation mode.
6. **Reconstruct page**
   
   Preserve relevant artwork, lines, tables, images and backgrounds. Draw the translated content.
7. **Generate final PDF**
   
   Embed the required fonts, preserve links and metadata where appropriate, and make text searchable/selectable.
8. **Render and validate**
   
   Inspect every page for clipping, missing blocks, text overlap, incorrect shaping and broken tables.

One difficult case deserves special attention.

For digitally generated PDFs, you may be able to reuse the original vector artwork and extract native text spans directly.

For scanned documents, the original text is part of the page image. Removing it without damaging surrounding lines, stamps and backgrounds is much harder. Your harness will need masking and potentially image inpainting, or a reflowed page reconstruction mode.

That is why I would support two distinct outputs\:

- **Faithful reproduction\:** Preserve the original background, page dimensions, positions and layout wherever feasible.
- **Reconstructed translation\:** Recreate the document's structure and visual style while allowing text to reflow and pages to expand.

This distinction is critical. An excellent OCR model cannot guarantee a perfect visual match when the translated language requires significantly more space.

---

# 9\. How to evaluate the complete product

Don't evaluate this project exclusively through OCRBench or OmniDocBench.

You need your own end-to-end benchmark.

| Layer | Metrics |
|--------|--------|
| Recognition | CER, WER, exact digit and identifier accuracy |
| Localization | Bounding-box IoU, detection precision/recall |
| Document structure | Reading-order accuracy, table TEDS |
| Extraction | Field precision, recall and F1 |
| Translation | COMET, human evaluation, terminology and number preservation |
| Grounding | Source-box correspondence and text coverage |
| PDF rendering | Overflow, clipping, overlaps, missing content |
| Final product | Successful translated document rate |

I would make three metrics mandatory release gates\:

**1\. Source content preservation\:** No dropped paragraphs, cells, signatures, identifiers or important fields.

**2\. Translation faithfulness\:** No invented, omitted or materially altered information. Pay special attention to numbers, dates, names and legal terminology.

**3\. Rendering quality\:** No clipped text, accidental overlaps or broken scripts.

One useful automated check is to render the translated PDF, extract its visible text again, and compare that output to the translated Document IR.

This creates an independent post-render validation loop.

---

# 10\. Complete research reading list

I've arranged these in the order I would study them, rather than simply giving you a pile of arXiv links.

## Reading group A - Understand OCR fundamentals

| Resource | What you should learn |
|--------|--------|
| [TrOCR](https://arxiv.org/abs/2109.10282?utm_source=chatgpt.com) | Recognition, pretrained visual/text Transformers, synthetic training |
| [SVTR](https://arxiv.org/abs/2205.00159?utm_source=chatgpt.com) | Visual text recognition and character-level feature learning |
| [PARSeq](https://arxiv.org/abs/2207.06966?utm_source=chatgpt.com) | Internal language models and sequence prediction |
| [ABINet](https://arxiv.org/abs/2103.06495?utm_source=chatgpt.com) | Language-aware OCR and iterative refinement |

## Reading group B - Understand Document AI

| Resource | What you should learn |
|--------|--------|
| [DiT](https://arxiv.org/abs/2203.02378?utm_source=chatgpt.com) | Document-specific visual representation |
| [Donut](https://arxiv.org/abs/2111.15664?utm_source=chatgpt.com) | OCR-free document understanding and synthetic generation |
| [LayoutLMv3](https://doi.org/10.1145/3503161.3548112?utm_source=chatgpt.com) | Joint image-text-document representations |
| [PaddleOCR 3.0](https://arxiv.org/abs/2507.05595) | Modular OCR, layout, extraction |

## Reading group C - Understand modern OCR VLMs

[![PaddlePaddle/PaddleOCR-VL · Hugging Face](https://images.openai.com/static-rsc-4/vrxuKfWbfLSj76aJFMYYGjM0Ra_3V1XLXtTjC7QUH3jwMgK52OVVL5xjIsHAL3eKnNroMF4URPPa8GWJo42UtaDEUFeepXBJDcdX_Nz197PLzGh2TlFgXCS8UaGpB5ceQe-mm_xiLV2AsyFhK6EX9z1ITL9e6yVtvyZ_izTLWG7_Ez33MII70Ta90Y5-NgSa?purpose=fullsize)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

**PaddleOCR-VL and 1.5 / 1.6**

[Original paper](https://arxiv.org/abs/2510.14528)

Dynamic resolution, data refinement and compact multilingual VLMs.

---

[![腾讯混元](https://images.openai.com/static-rsc-4/t0lQpxacAKU0eFv1jLg64DsGcrWKQ-CkUlZKRFd1Low8PE5BhdY7Ncp4qYhEw6zeRH-p8qpVirsTu03pgFm3uJ4pWQ-3qOkuYlEDbows3wYlTXKxhRdk0fAerTMym9qwBhIpXR9Lew6_Zl5WOCYeIjViaKK6_Rw7IlS_j9IvqgT7uJOVp0mvXThMdA7WoWly?purpose=fullsize)](https://hunyuan.tencent.com/vision/en?tabIndex=0)

**HunyuanOCR**

[Technical report](https://arxiv.org/abs/2511.19575)

Integrated OCR, extraction, translation and RL.

---

[![DeepSeek OCR: Smarter, Faster Context Compression for AI](https://images.openai.com/static-rsc-4/7xM2yJ9y3sDiH38rSVt8-KuVhPIxVHX3UCPm6ZzY8IL3MvbcrBt65K_SA2GT52DlDttKESl_NhXDbBgEoL5zD48q9a6bqdnB9Gnv4mYJO7DCHHyJXWvrUmYAzwkJ9nIolsL9k6GYQamDz0_Nnup_AB2GgonqOgDGJZSHHsh5uTy6eCIjLXR3ksmkst1ncxzh?purpose=fullsize)](https://www.clarifai.com/blog/deepseek-ocr/)

**DeepSeek-OCR and OCR 2**

[DeepSeek-OCR](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)

· [OCR 2](https://arxiv.org/abs/2601.20552)

Visual token compression and causal visual processing.

---

[![Unlimited OCR: How Baidu Solves Long Document OCR](https://images.openai.com/static-rsc-4/TrFdJw3JWN-YHkV18Io8AEDed3i4R2_la6CMEQuD2eYxfDLEtJ3MEIfBE6t_5iClEIFP__iDr-jf44KoDr8oX8yoaFF1uITWrqaWn5KMZYbazxaTUEUK4CIP6zmm9LiZPdo7sFriUHULvpFGmZxbohxVo8R8bcHfH1TqYuYXrYw141QATrb4woZb8rMN3U5a?purpose=fullsize)](https://www.labellerr.com/blog/baidu-unlimited-ocr/)

**Unlimited-OCR and Unlimited-OCR 2**

[Original paper](https://arxiv.org/abs/2606.23050)

· [Version 2](https://arxiv.org/abs/2610.39308)

Long-horizon document decoding and attention efficiency.

## Reading group D - Understand translation and reconstruction

| Resource | Why it matters |
|--------|--------|
| [IndicTrans2](https://arxiv.org/abs/2305.16307) | Language-specific translation |
| [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) | Parallel translation data |
| [Understand Layout and Translate Text](https://pubmed.ncbi.nlm.nih.gov/40031201/?utm_source=chatgpt.com) | Jointly modelling document layout and translation |
| [ICDAR 2025 End-to-End Document Image Translation Challenge](https://arxiv.org/abs/2603.09392?utm_source=chatgpt.com) | End-to-end document image translation and evaluation |
| [PyMuPDF text reconstruction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html) | Actual PDF rendering and text placement |

I particularly recommend the document image translation paper because it examines the issue you're trying to solve: separating translation and layout can create accumulated errors, while a model without sufficient explicit layout supervision may fail to preserve spatial structure. \[[Understand Layout and Translate Text: Unified Feature-Conductive End-to-End Document Image Translation.](https://pubmed.ncbi.nlm.nih.gov/40031201/?utm_source=chatgpt.com); [Understand Layout and Translate Text: Unified Feature-Conductive End-to-End Document Image Translation \| IEEE Journals &amp; Magazine](https://ieeexplore.ieee.org/document/10844563/?utm_source=chatgpt.com)\]

---

# 11\. My recommended project execution order

Let me bring everything together into one practical decision.

# What I would start implementing

(First priority)

**1\. Build the PDF reconstruction harness**

Create the PDF-to-Document-IR-to-PDF path with a mock bilingual Document IR. You should prove that your renderer can preserve layout and handle translated text before depending on an AI model.

---

**2\. Establish pretrained model baselines**

Compare\:

- PaddleOCR-VL 1.5 or 1.6 for Hindi/Devanagari document parsing.
- HunyuanOCR for integrated translation and OCR concepts.
- DeepSeek-OCR 2 for visual processing.
- LFM2.5-VL as your existing comparison.

Benchmark on the same documents and verify the permitted use and fine-tuning licences.

---

**3\. Build the Indic OCR data engine**

Use Mozhi, IndicDLP, BPCC, public digital PDFs, real document scans and synthetic documents.

---

**4\. Fine-tune the OCR model**

Begin with visual alignment and recognition before adding full-page tasks.

---

**5\. Train joint translation**

Introduce text translation, then image-to-bilingual-Document-IR supervision.

---

**6\. Train the full document task**

Add source grounding, uncertainty, tables, layout and extraction.

---

**7\. Optimize and deploy**

Quantize, benchmark on edge hardware, and add long-document attention only if needed.

## A final architectural decision

I would keep your two original documents, but assign them different roles\:

| Your document | Role in the project |
|--------|--------|
| Tab 1 - Indic OCR Foundation Model | Main training and data roadmap |
| Architecture v2 - MoE and whole-PDF memory | Advanced architecture research, after the baseline |

In particular, keep the latent document memory, adaptive loop, MoE and advanced attention mechanisms behind experimental gates. Your own Architecture v2 rightly acknowledges that the memory and loop have not yet been experimentally validated and proposes adding them incrementally. fileciteturn0file0L17-L22

**My final recommendation\:** Build a dense 1.2B-1.8B multilingual document VLM, teach it OCR, extraction, translation and spatial grounding together, and use an external deterministic reconstruction harness to create the PDF. You will have one AI inference model for the document's intelligence, without asking that model to perform low-level PDF composition.

That gives you a much more practical path to a genuinely useful edge-deployable document translation model than trying to reproduce DeepSeek's entire training infrastructure or Unlimited-OCR's long-context architecture from the beginning.

The first concrete engineering milestone should be a working `Document IR → translated PDF` renderer. Once that contract is stable, the OCR model has a precise target to learn.

## Prompt:
11/15/2026, 7:40:49 PM

# Selected text:

## Selection 1
I would change this.

You have essentially duplicated the prediction twice. Also, model, usage, elapsed_seconds and finish_reason are inference metadata, not OCR ground truth.

Keep those in a separate experiment log.

For the training dataset, maintain just:

Image reference.

Task/instruction.

Verified target.

Optional annotation metadata.

Store targets as structured objects in your annotation files, then serialize the assistant target consistently when constructing the model's training conversations.

## My request:
i dont understand this section

## Response:
11/15/2026, 7:41:03 PM · gpt-6-t-mini

> **Clarifying dataset structure**
>
> I’m separating three things: the verified target annotation, the model’s prediction, and experiment metadata. `raw_content` and `extraction` duplicate the target, so I’d keep one parsed annotation, use stable relative image paths, and store predictions separately from training examples.
>
> **Clarified dataset structure**
>
> 
>
> Worked for 15s

No worries\! The main idea is that **you need to separate what the model predicted from what the model is supposed to learn.**

Let's use your actual JSON example.

## 1\. What's wrong with your current JSON?

Your current structure looks like this\:

```json
{
  "source": "F:/gitRepos/LiquidFM/data/images/renders/0004.jpg",

  "model": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0",

  "elapsed_seconds": 8.89,

  "usage": {
    "completion_tokens": 1186,
    "prompt_tokens": 1903
  },

  "raw_content": "{ \"document_type\": \"Complaint Form\", ... }",

  "extraction": {
    "document_type": "Complaint Form",
    "languages": ["Hindi", "English"],
    "fields": []
  }
}
```

There are two different types of information here.

**A. Prediction information**

This tells you what happened when you ran LFM\:

- Which model generated the output?
- How long did inference take?
- How many tokens were consumed?
- What was the raw response?

**B. Training information**

This tells you what the model should learn\:

- Which image should it read?
- What task should it perform?
- What is the correct answer?

The problem is that `raw_content` and `extraction` contain essentially the same prediction.

You don't need both in the training dataset.

## 2\. How I would organize it

I'd maintain two separate JSONL files.

**1\. training\_dataset.jsonl**

Contains the image, task and verified ground truth.

This is what your training pipeline reads.

---

**2\. inference\_logs.jsonl**

Contains the model used, its raw prediction, inference time, token usage and other experiment metadata.

This is for debugging, benchmarking and tracking teacher models.

### A. Your training dataset

Suppose a human has checked the LFM output and corrected all OCR mistakes.

Your training record would look like this\:

```json
{
  "id": "complaint_0004",
  "image": "images/renders/0004.jpg",

  "task": "document_extraction",

  "instruction": "Extract all fields from this complaint form. Preserve empty fields and identify uncertain text.",

  "target": {
    "document_type": "Complaint Form",
    "languages": ["Hindi", "English"],
    "fields": [
      {
        "field_name": "Reason for delay in reporting",
        "value": "कार्यान्वयन के लिए प्रतिक्रिया देने के लिए कोई कार्य नहीं किया गया है।"
      },
      {
        "field_name": "Particulars of items",
        "value": "संपत्ति की विवरण"
      },
      {
        "field_name": "S.No.",
        "value": "",
        "status": "blank"
      }
    ],
    "uncertain_fields": [
      "First Information Contents"
    ]
  }
}
```

Notice that I've retained your original task and target structure, while simplifying the record.

The important condition is that `target` must be the verified correct answer, not an unchecked LFM prediction.

### B. Your inference log

Separately, you preserve the original model execution details\:

```json
{
  "id": "complaint_0004",
  "image": "images/renders/0004.jpg",

  "model": "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0",

  "elapsed_seconds": 8.89,

  "usage": {
    "completion_tokens": 1186,
    "prompt_tokens": 1903,
    "total_tokens": 3089
  },

  "finish_reason": "stop",

  "raw_prediction": "{...}",

  "validation_status": "pending"
}
```

This is useful when you want to compare LFM against PaddleOCR-VL, HunyuanOCR or your own model.

---

## 3\. How does this become training data?

This is the part that might be confusing.

Your model doesn't directly learn from arbitrary JSON files. Your training code reads these records and converts them into conversations.

For example\:

**Input to the model**

[![Sajjan Jindal Accused of Rape by Mumbai-based Doctor, FIR Registered](https://images.openai.com/static-rsc-4/ZmzT8Q_T9xPiYOljzfuKe8f4vw2mi9psZzG8MIuhzCDybT6yYsuSDFbGloHHbI771a33CDTH0VkRFm1-FDaoSWImHBuec7oRGyK-HzVEMo-laecKziPJVM9ORNeeWdX6MO59lbGx5RrgwG3-Xult4E0Ch5cKTtV6-BzywAFPMov1Yq7Kyu-mdvFVD5P69mB0?purpose=fullsize)](https://www.moneylife.in/article/sajjan-jindal-accused-of-rape-by-mumbaibased-doctor-fir-registered/72882.html)

**Image\:** complaint\_0004.jpg

**Instruction\:** Extract all fields from this complaint form. Preserve empty fields.

---

**Expected assistant output**

```json
{
  "document_type": "Complaint Form",
  "languages": ["Hindi", "English"],
  "fields": [
    {
      "field_name": "Reason for delay",
      "value": "कार्यान्वयन के लिए..."
    }
  ]
}
```

(This is what the model learns to generate)

Internally, your training pipeline serializes the target JSON into a sequence of tokens and calculates the next-token prediction loss.

Conceptually\:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all fields from this complaint form."},
        ],
    },
    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
]
```

The actual image formatting depends on the VLM's processor and chat template.

## 4\. Why this distinction matters for your OCR project

Imagine you collect 100,000 documents using LFM, PaddleOCR and HunyuanOCR as teacher models.

You might get\:

| Model | Prediction |
|--------|--------|
| LFM | भारत सरकार |
| PaddleOCR | भारत सरकर |
| HunyuanOCR | भारत सरकार |

Your annotation pipeline can identify disagreement and send the sample for verification.

After human verification, the training target becomes\:

```json
{
  "text": "भारत सरकार"
}
```

That is your ground truth.

You can retain all three teacher predictions in the inference logs for future analysis, but you train against the verified target.

**One more important distinction\:** You can use teacher-generated annotations without human verification for some pretraining experiments, provided you measure and control their noise. But don't silently label them as ground truth.

---

## 5\. My suggested structure for your entire project

I would standardize your dataset around this format\:

```text
ocr_data/
│
├── images/
│
├── annotations/
│   ├── recognition.jsonl
│   ├── page_parsing.jsonl
│   ├── layout.jsonl
│   ├── extraction.jsonl
│   └── translation.jsonl
│
├── predictions/
│   ├── lfm.jsonl
│   ├── paddle.jsonl
│   └── hunyuan.jsonl
│
├── validation/
│   ├── pending_review.jsonl
│   └── verified.jsonl
│
└── experiments/
    ├── inference_logs.jsonl
    └── training_logs.jsonl
```

The key idea\:

- **Images\:** What the model sees.
- **Annotations\:** What the model should learn.
- **Predictions\:** What teacher models produced.
- **Validation\:** Whether those predictions were verified.
- **Experiments\:** How the models performed.

This also makes it much easier to reuse the same document for OCR, layout reconstruction, translation, extraction and eventually joint multimodal training.

**Think of it this way: the teacher model supplies a draft answer; your verified annotation supplies the answer key.** That's the distinction I was making.

## Prompt:
11/17/2026, 11:37:27 PM

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000ca7882118ca2af5c36e2bfb8&fn=image.png&cd=attachment&ts=498616&p=fs&cid=1&sig=fe7fbdbfa014e4186041ac31610be2c5e7e6583cf983162ce5ebea388f50e99f&v=0)

![image(1).png](https://chatgpt.com/backend-api/estuary/content?id=file_000000005970821191d28a5cc16fee0e&fn=image%281%29.png&cd=attachment&ts=498616&p=fs&cid=1&sig=6a4e53036e101545581fb0ea373109fd186de3711c45a832b7cdbf32496ba706&v=0)

![image(2).png](https://chatgpt.com/backend-api/estuary/content?id=file_0000000023c082119493e7ea2dd0a683&fn=image%282%29.png&cd=attachment&ts=498616&p=fs&cid=1&sig=a4a6a13eb9af96ec7772b043b3f35a7f3c7a24a1ffc7f6a97d4dcce6d6f5f690&v=0)

# Selected text:

## Selection 1
Indic Document Translation training roadmap
Recommended order for a pretrained VLM, with optional foundation-model experiments later.
Stage 0 — Data engine and evaluation
No training initially
Build PDF ingestion, synthetic rendering, annotation, translation alignment and the golden test set.

Deliverable: A validated dataset and baseline benchmark.

Stage 1 — OCR visual adaptation
Vision-language continued training
Train on word, line, paragraph and mixed-script images.

Data: Real and synthetic image-text pairs.

Train: Connector, then language adapter, followed by selected vision layers.

Deliverable: Accurate target-language recognition.

Stage 2 — Document understanding
Multimodal continued pretraining
Train full-page transcription, layout, reading order, spotting, region reading, tables and grounding.

Data: Full document images with structural annotations.

Deliverable: Source-document reconstruction in Document IR.

Stage 3 — Translation adaptation
Text-only translation training
Train source-to-target translation using BPCC and high-quality bilingual data.

Data: Parallel text, back-translation and verified domain translations.

Deliverable: Reliable multilingual translation capability.

Stage 4 — Joint OCR + translation
Core research stage
Train the model to recognize, understand, extract and translate each document block jointly.

Data: Source document images with bilingual Document IR ground truth.

Deliverable: Image-to-translated-document generation.

Stage 5 — Application SFT

Train with actual product instructions, legal forms, financial forms, government documents and other target domains.

Add blank-field handling, uncertainty and schema validation.

Deliverable: Stable product-facing outputs.

Stage 6 — Preference optimization / RL

Optional. Improve text spotting, parsing, translation faithfulness, uncertainty and layout consistency with verifiable rewards.

Deliverable: Improved reliability on challenging samples.

Stage 7 — Long-document extension

Only when the previous stages establish that page-by-page processing is insufficient.

Evaluate R-SWA, SR-SWA or your document-memory approach.

Deliverable: Scalable multi-page inference.

Stage 8 — Edge optimization

Quantization, distillation, attention optimization, batch sizing and document-level memory management.

Deliverable: Deployable model within your edge resource budget.

## My request:
see i am using an strong sota model like geminito generate and also i have  datalike layout detected boboxes, src and tgt extracted. but its limited max to max 2-3k only.

but i need to train it hard on language lets start with marathi, hindi for now. so i way planning like take a sota model like deepseek 3b or unlimited ocr 3b and start with 1. visial adaption as u told but need a list of indic datasets for this. 2. doc understanding for my focus domain specific with this 2k-3lkk data and 4. translation i am planning to train the llm decoder seperate and attach. Need a revised plan for this and also what datasets and estimated gpu requirements.

## Response:
11/17/2026, 11:37:34 PM · gpt-6

> **Researching the OCR plan**
>
> I’m checking current Indic OCR and translation datasets, licensing, model compatibility, and realistic GPU needs. I’ll also separate evidence-based recommendations from assumptions about attaching an MT decoder.
>
> **Researched the OCR plan**
>
> 
>
> **Researching OCR datasets and GPU requirements**
>
>
> ****
>
> 
>
> **Researched OCR datasets and GPU requirements**
>
> 
>
> **Estimating training needs**
>
> I’m verifying model capabilities and dataset availability, then I’ll translate them into practical configurations: lightweight LoRA on 24–48 GB GPUs, versus full fine-tuning across larger multi-GPU systems.
>
> **Estimating training needs**
>
> I’m distinguishing a 2–3k-example pilot from a 2–3-lakh-page corpus. I’ll estimate GPU requirements separately, because LoRA, full fine-tuning, image resolution, and output length change the totals substantially.
>
> **Estimated training needs**
>
> 
>
> **Searched 18 websites**
>
> 
>
> Worked for 57s

# Revised plan: Hindi + Marathi OCR foundation model under 3B

Now I understand the actual constraint much better.

You already have\:

- A strong teacher model: Gemini.
- Document images.
- Layout-detected bounding boxes.
- Extracted source text (`src`\).
- Target-language translations (`tgt`\).
- A limited annotation budget of approximately 2K-3K Gemini-generated samples per generation batch, with a possible overall dataset of 2K-3 lakh examples.

You don't need to build the entire data engine from zero. You already have a valuable starting dataset.

I would change the strategy from the earlier roadmap in one major way\:

**Use a pretrained OCR VLM as the visual foundation, specialize its recognition for Hindi and Marathi, and separately develop translation capability before combining both into the document task.**

Also, I would not immediately train a separate 3B language decoder and attach it. There is a better way to experiment with that without committing to an incompatible architecture.

## 1\. First, choose the base model

I would shortlist three models, not just DeepSeek and Unlimited-OCR.

[![拆解DeepSeek-OCR：3个核心技术让OCR快20倍-夜雨聆风](https://images.openai.com/static-rsc-4/707A9aGJUoB67DPNdH171Emz4ldmFAmPR2I_LjVP_Y3BgK9HFFv9hLxPYCOBUJNznmw7AHtHxx7x19_x9gIp-gz0Ueaay7pLDn0U_EhDCieIlC5VuNhwtmPLHZZUJVmns9TWpvPoOzEeQTidDctwMRnuXSQcqTmDLHoEWtVDTbF_7di0SEcDiaYZMkd8_ktV?purpose=fullsize)](https://www.yeyulingfeng.com/268188.html)

**1\. DeepSeek-OCR**

(Research candidate)

- 3B MoE decoder, approximately 570M active parameters.
- SAM + CLIP visual encoder.
- 16× visual token compression.
- Strong document OCR architecture.

[Model weights](https://huggingface.co/deepseek-ai/DeepSeek-OCR?utm_source=chatgpt.com) · [Paper](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)

---

[![Unlimited OCR: How Baidu Solves Long Document OCR](https://images.openai.com/static-rsc-4/TrFdJw3JWN-YHkV18Io8AEDed3i4R2_la6CMEQuD2eYxfDLEtJ3MEIfBE6t_5iClEIFP__iDr-jf44KoDr8oX8yoaFF1uITWrqaWn5KMZYbazxaTUEUK4CIP6zmm9LiZPdo7sFriUHULvpFGmZxbohxVo8R8bcHfH1TqYuYXrYw141QATrb4woZb8rMN3U5a?purpose=fullsize)](https://www.labellerr.com/blog/baidu-unlimited-ocr/)

**2\. Unlimited-OCR 2**

(Long-document candidate)

- 3B parameters.
- Built on the DeepSeek-OCR lineage.
- Sparse Reference Sliding Window Attention.
- Designed for very long document parsing.

[Model weights](https://huggingface.co/baidu/Unlimited-OCR-2?utm_source=chatgpt.com) · [Paper](https://arxiv.org/abs/2610.39308?utm_source=chatgpt.com)

---

[![\[October 2025\] AI & Machine Learning Monthly Newsletter 💻🤖 | Zero To Mastery](https://images.openai.com/static-rsc-4/MdcLU9sd4OaXCHEo7WBJFuhEMQmSK69SpsOQYBro_5vMeNGtZ1YZ6kOV5wthvW4fg62MSGALS8cw6jn1lsYjs7q2BvdkJWQ9qyUojfh2RapaKTqN4owzfM0D5w0NAlxo7VRhDcNZ7Lq_Tcufpmp_JXyOX-nlKIxWVCm8ohT31HB8y2ytQDu9jfyZr-cI1CcG?purpose=fullsize)](https://zerotomastery.io/blog/ai-and-machine-learning-monthly-newsletter-october-2025/)

**3\. PaddleOCR-VL**

(Essential benchmark)

- Approximately 0.9B parameters.
- NaViT-style dynamic-resolution vision encoder.
- ERNIE-4.5-0.3B decoder.
- Explicitly supports Hindi and Marathi.

[Model weights](https://huggingface.co/PaddlePaddle/PaddleOCR-VL?utm_source=chatgpt.com) · [Paper](https://arxiv.org/abs/2510.14528?utm_source=chatgpt.com)

My preference is to start with DeepSeek-OCR as your research backbone and PaddleOCR-VL as the comparison baseline.

However, there is an important practical catch: Unlimited-OCR 2 has support for training through ms-swift, but its custom attention modifications and model implementation introduce additional complexity. DeepSeek-OCR's model repository is also not equivalent to a generic Transformers VLM checkpoint. Verify the exact training implementation, checkpoint compatibility, tokenizer and licence before committing. \[[UPSTREAM\_README.md](https://huggingface.co/cstr/Unlimited-OCR-2-GGUF/blob/main/UPSTREAM_README.md?utm_source=chatgpt.com); [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR?utm_source=chatgpt.com); [Unlimited OCR Works 2](https://arxiv.org/abs/2610.39308?utm_source=chatgpt.com); [PaddleOCR-VL Introduction](https://www.paddleocr.ai/v3.3.1/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html?utm_source=chatgpt.com)\]

I would first run a small fine-tuning experiment on both the DeepSeek and Unlimited checkpoints, then choose one. Don't select Unlimited merely because its inference architecture is more sophisticated.

---

# 2\. Your revised training architecture

I would split your model development into four major training programmes.

## Hindi + Marathi OCR model

**Pretrained OCR VLM**

DeepSeek-OCR / Unlimited-OCR

Visual encoder + multimodal decoder

(Training 1)

**Indic visual adaptation**

Printed words, lines, paragraphs, mixed scripts, degraded text.

(Training 2)

**Domain-specific document understanding**

Full-page OCR, layout, spotting, tables, region reading and extraction.

(Training 3)

**Hindi ↔ English and Marathi ↔ English translation**

Separately trained translation capability and later decoder integration.

(Training 4)

**Joint OCR + translation + Document IR**

Image → verified source text → translated text → grounded document representation.

**Edge model ≤3B**

Quantized inference and PDF reconstruction harness

The important thing is that these are training programmes, not necessarily four different models. You can progressively update the same multimodal model.

# 3\. Training 1 - Indic visual adaptation

(Your first actual training run)

This is where you teach the pretrained vision-language model to recognize Devanagari accurately.

Do not begin with your 2K-3K domain documents. Start with publicly available Hindi and Marathi word and line datasets.

The objective is to improve the model's visual recognition of\:

- Matras and vowel signs.
- Conjunct consonants.
- Nukta characters.
- Similar-looking glyphs.
- Hindi and Marathi vocabulary.
- Mixed Devanagari-English text.
- Low-resolution and degraded printed text.

## 3.1 Indic datasets to collect

These are the most relevant resources I would use.

[![Hindi OCR VLM](https://images.openai.com/static-rsc-4/QuxfuoJbOR10Ds3vIz0xMemKWf7yCeo2rv3fzFqiA91KtYzn4FSYNraIAwTofYHDKBNm8Oo6WOhzvygM30r3OK5_KATXWtXZRNWMBBdoxQrPBbIpofWBeOeTLtWc1X5Td0OGdu2yNa8Lpplq03qknLv4nhkfplt4FxfrjN0e4mtLcHVnyknvhxxJ2adJKvXb?purpose=fullsize)](https://souvikg544.github.io/HindiOCR-VLM.github.io/)

**1\. Mozhi**

(Highest priority)

- Hindi: approximately 80K training words.
- Marathi: approximately 80K training words.
- Both include line-level training data.
- Real scanned printed text.

[Dataset and paper](https://cvit.iiit.ac.in/usodi/tdocrmil.php?utm_source=chatgpt.com) · [Download portal](https://ilocr.iiit.ac.in/dataset/)

Best starting point for printed recognition. \[[Document Images](https://cvit.iiit.ac.in/usodi/tdocrmil.php?utm_source=chatgpt.com); [NLTM OCR](https://ilocr.iiit.ac.in/dataset/10/?utm_source=chatgpt.com)\]

---

[![Samples from the dataset of handwritten Devanagari images  | Download Scientific Diagram](https://images.openai.com/static-rsc-4/MMvRaq1eLG0Dn7ndAdCGpPPKnBLW9Byv4N8zyAxQxOK3qfnyktjiU8N6KI9qK-r0-l9ND_fx4moKGeoeLV8EHQrpg3BAhn27SahYC1BHmjsS1vfsrTmDZleJA1HR-mKpoVvym-p5nB8igjP05nkKsUU-gqRmA79GtCzcLc-oKAaAJS-eDzkhdQ2zMBbi9a1d?purpose=fullsize)](https://www.researchgate.net/figure/Samples-from-the-dataset-of-handwritten-Devanagari-images_fig3_288331541)

**2\. IIIT-INDIC-HW-WORDS**

- 872K handwritten instances across eight scripts.
- Includes Hindi/Devanagari and other Indic datasets.
- Individual word images and transcripts.

[Official dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words?utm_source=chatgpt.com)

Useful if handwriting is in your target domain. Don't mix it heavily into printed OCR training unless handwriting is a product requirement. \[[IIIT-INDIC-HW-WORDS: A Dataset for Indic Handwritten Text Recognition](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words?utm_source=chatgpt.com); [NLTM OCR](https://ilocr.iiit.ac.in/dataset/18/?utm_source=chatgpt.com)\]

---

[![IndicSTR12: A Dataset for Indic Scene Text Recognition](https://images.openai.com/static-rsc-4/hakwC6Li8dYIsq6N6eusDIQCX-ixdLHOl1jKhVcW6lw3HqKxDWX8KLWPwiEAruY7LtAS7ELpKWjIOGXEjtF1RfhVkfKZ1pb-NWbieb_z_x9JmDXrBrDt38dXQYPYdlbFM3wE9-644JmJ_iX5qWMNuZi4jMOB6SLKkNPFEIZzPJrRxMOm5bWyxmT3pdAibxZZ?purpose=fullsize)](https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr)

**3\. IndicSTR12**

- Hindi scene-text word images.
- Marathi scene-text word images.
- Perspective, blur, occlusion and real-world text.

[Dataset portal](https://ilocr.iiit.ac.in/dataset/)

Useful for robustness, but scene text should be a secondary component for your document-focused model. \[[NLTM OCR](https://ilocr.iiit.ac.in/dataset/?utm_source=chatgpt.com); [NLTM OCR](https://ilocr.iiit.ac.in/dataset/30/?utm_source=chatgpt.com)\]

---

[![ICDAR 2024 HWD](https://images.openai.com/static-rsc-4/l--Ga9bO6OEruIjSZLp2UinVKaAm6awJ8Tp2HXvNuFQ01OVpDc5McongB_Br6m3RSvjpMp1kPDXZLUiznxFRYTVuOP2xP9mc2b24F_jR8auyVd9rbHs1cCtuwZ9FFEDFZpZNvoku7KXgwUgfa1cCLrKD3nb7YGzFS8ocYkBOlQX0Ve_M81tZd6anPpUpHXDD?purpose=fullsize)](https://ilocr.iiit.ac.in/icdar_2024_hwd/dataset.html)

**4\. IndicPage-OCR**

- Full-page handwritten recognition.
- Hindi and other Indic scripts.
- Combines page, line and word evidence.

[Research page](https://indicpageocr.github.io/?utm_source=chatgpt.com)

Particularly interesting for page-level curriculum and handwritten adaptation. Check the released dataset and code availability; not everything is necessarily downloadable yet. \[[IndicPage-OCR](https://indicpageocr.github.io/?utm_source=chatgpt.com)\]

---

[![Hindi OCR VLM](https://images.openai.com/static-rsc-4/l737sUtk8onHYRcZ7huYsN4MQFChskNcBfHe4VaCZl-dkgmK0KtqS4oUJjmtV8pdzbmO5vBkdeA9sOrrzhb_3W6BWrImF199oZSQuKhwrMDfcIwe76c9b2rFn5Uyyab3GJIR00LnGfSqGZVamLO8L-tfiLHy_WOdxst2Gy1x92mQLg9R5qmcyq0AMt5ZLyfL?purpose=fullsize)](https://hindiocr-vlm.github.io/)

**5\. ICDAR 2025 Indic Handwritten Document Recognition**

- Hindi and Marathi handwritten pages.
- Word bounding boxes.
- Page transcription in reading order.

[Dataset portal](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/dataset.html?utm_source=chatgpt.com)

Useful for testing whether your model generalizes from word recognition to page recognition. The published validation sets are intended for fine-tuning; respect the academic-use conditions. \[[ICDAR 2025 IHDR](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/dataset.html?utm_source=chatgpt.com)\]

---

[![Document Images](https://images.openai.com/static-rsc-4/PZnPKW9e_WWmBsnEb7y988CBOKU1zihgkfRXqVCTrZ_FQ-kqjhs_ToJ8imtFY4E_5CjTU-L8RlagXxg5yiPqYRCOSDqZgGakCDGZt9tvq-DHYiEMNckKvXKANuZMePQsaNeLFNFi3Ivu0E2O0F-FqtTHvwU0gasZ4Do11euTEl-ybA0CxPM7cNfPtxvje74t?purpose=fullsize)](https://cvit.iiit.ac.in/usodi/tdocrmil.php)

**6\. Devanagari OCR ICL Benchmark**

- 10K Hindi evaluation pairs.
- 10K Marathi evaluation pairs.
- Approximately 6.6K additional retrieval examples.
- OCR-corrupted text paired with ground truth.

[Hugging Face dataset](https://huggingface.co/datasets/AbhishekBhandari/Devanagari-OCR-ICL-Benchmark?utm_source=chatgpt.com)

Use for post-OCR correction research and evaluation, not image recognition pretraining. Keep its test split untouched. \[[AbhishekBhandari/Devanagari-OCR-ICL-Benchmark](https://huggingface.co/datasets/AbhishekBhandari/Devanagari-OCR-ICL-Benchmark?utm_source=chatgpt.com)\]

## 3.2 What should your visual adaptation dataset look like?

I would build a combined dataset rather than maintain separate models for every crop size.

| Data | Hindi | Marathi | Purpose |
|--------|--------|--------|--------|
| Mozhi words | \~80K | \~80K | Recognition |
| Mozhi lines | \~6.5K | \~8.4K | Sequence recognition |
| Synthetic words | 500K-1M | 500K-1M | Glyph diversity |
| Synthetic lines | 200K-500K | 200K-500K | Line recognition |
| Real degraded samples | 20K-50K | 20K-50K | Robustness |
| Handwriting | Optional | Optional | Only if required |
| Scene text | Optional | Optional | Secondary robustness |

These are proposed synthetic-data production targets, not published dataset counts. Mozhi figures are approximate training-set sizes.

You can create synthetic examples from\:

- Hindi and Marathi text corpora.
- Government document templates.
- Newspaper text.
- Legal and financial vocabulary.
- Multiple Devanagari fonts.
- Random font sizes, weights and spacing.
- Blur, skew, compression, bleed-through and background noise.

One especially important detail: render complete Devanagari grapheme clusters correctly. Do not randomly split Unicode code points when generating corruptions. Matras and combining marks need to remain linguistically and visually valid.

## 3.3 How to train the base VLM

For a DeepSeek-OCR-style model\:

| Component | Initial strategy |
|--------|--------|
| SAM visual encoder | Freeze initially |
| CLIP visual encoder | Freeze initially |
| Visual connector | Train |
| Language decoder | LoRA initially |
| Embedding/tokenizer | Keep original |
| Later | Unfreeze selected visual layers |

I would run these experiments\:

**Experiment A\:** Connector only.

**Experiment B\:** Connector + decoder LoRA.

**Experiment C\:** Connector + decoder LoRA + final visual encoder layers.

Compare all three using the same test set.

Do not replace the tokenizer. Devanagari support is not simply a matter of inserting new Unicode tokens into a pretrained model; the decoder's existing token embeddings and learned language representations matter.

### Training objective

For this phase\:

$$
\mathcal{L}_{visual} =
-\sum_t \log P(y_t \mid y_{<t}, I)
$$

The target is exact text, not translation or JSON extraction.

Start with word images, progress to lines, then paragraphs.

**Exit gate\:** CER and WER must improve independently on Hindi and Marathi real printed text. If Marathi is weaker, increase its sampling weight rather than simply increasing total training epochs.

---

# 4\. Training 2 - Your 2K-3 lakh domain documents

This is where your existing Gemini-generated data becomes extremely valuable.

You said you have\:

- Layout boxes.
- Source text.
- Target text.
- Extracted document fields.

This is precisely the kind of supervision needed for domain-specific document understanding.

However, I would separate the data into two categories.

| Data | Training task |
|--------|--------|
| Image + boxes + block labels | Layout and spotting |
| Image + boxes + source text | OCR and page parsing |
| Image + source text + target text | Translation |
| Image + boxes + fields | Grounded extraction |
| Image + complete structured document | Document reconstruction |

## 4.1 Don't spend Gemini calls generating everything again

Use Gemini for the missing annotations, not for information you already have.

For example\:

**Your current document annotation**

[![Vikas Mitra Application Form | PDF](https://images.openai.com/static-rsc-4/ztkJfsd2vy_VNRMCsZOwHnATpzcw3-9EvPYpI75nq3te-F5zfzTnaNeImbxd2EhAfxEgvc4YiqmDlj9eAEpEfwP863zFGlBYtCjjCz2J_13Gtyz-KM2ZDLNQX6ZB97zC2xgy7jSOI391JaWOtQd3fX9lr1mSWC4NLJMLgn3gFK7IWfWZxT3FVEkOJvwPTH9H?purpose=fullsize)](https://www.scribd.com/document/909162750/Vikas-Mitra-Application-Form)

**Page 001**

Detected boxes: 24

Source text: available

Translation: available

---

**What you still need to validate**

- Does each text block correspond to the correct box?
- Is the source text exactly correct?
- Is the target translation faithful?
- Is the reading order correct?
- Are blank fields preserved?
- Are all page elements represented?
  
  (Existing data can supervise multiple tasks)

If Gemini only allows you to generate 2K-3K records in a batch, that is fine. Batch size is not the same as total dataset size.

Use a pipeline like this\:

1. Generate 2K-3K annotations.
2. Run schema validation.
3. Compare with existing OCR outputs.
4. Detect disagreement.
5. Human-review high-risk samples.
6. Store verified records.
7. Repeat.

You should also retain teacher identity and provenance for every annotation.

## 4.2 Dataset mixture for domain document training

I would start with this token mixture, not necessarily the same number of pages per task.

## Stage 2 proposed mixture

My proposed starting mixture. Adjust using per-task validation results and token counts.

Why keep 10% general Indic OCR replay? Because if you train only on your domain documents, the model can lose its broader script recognition ability.

### One crucial correction

If your 300K documents contain layout boxes but only 30K have verified source transcription, you cannot treat all 300K as page-text training samples.

You can still use all 300K for layout-only supervision, assuming their geometry and class labels are valid.

This is why separating annotations by task is so important.

# 5\. Training 3 - Train the translation decoder separately

This is the part where I would slightly change your proposed architecture.

You want to train a separate LLM decoder for Hindi/Marathi translation and attach it to the OCR model.

This is possible, but there are two ways to do it.

## 5.1 Option A - One shared decoder (recommended first)

**Vision encoder**

Reads the document image

**Shared language decoder**

OCR + translation + extraction + structured output

**OCR output**

Source text

**MT output**

Translated text

You separately train the same decoder on text-only translation samples.

For example\:

```text
Task: Translate Hindi to English.

Input:
भारत सरकार ने नई योजना शुरू की।

Output:
The Government of India has launched a new scheme.
```

Then you train the same decoder on\:

```text
Task: Read and translate this document image.

Input:
[Hindi document image]

Output:
{
  "source_text": "...",
  "translated_text": "..."
}
```

This gives you a shared decoder that has learned both OCR and translation.

Advantages\:

- One model.
- No separate translation service at inference.
- Shared language representations.
- Smaller deployment complexity.
- Easier quantization.

For your edge deployment objective, this is my preferred approach.

## 5.2 Option B - Separate translation decoder

If you specifically want an independent translation decoder, I would design it like this\:

**Visual encoder**

**OCR decoder**

Recognition and document structure

(Source tokens)

**Translation decoder**

Indic → English translation

(Target tokens)

**Grounded bilingual Document IR**

The translation decoder receives the recognized source tokens, along with language tags and optional document context.

For example\:

```text
OCR decoder:
भारत सरकार ने नई योजना शुरू की।

        ↓

Translation decoder:
The Government of India has launched a new scheme.
```

You can train the translation decoder independently using IndicTrans2-style data, then attach it through a projection or cross-attention interface.

But there are complications\:

- Tokenizer and embedding compatibility.
- Different hidden dimensions.
- Cross-attention integration.
- Source-to-target alignment.
- Additional inference memory.
- Joint training to ensure OCR errors don't destabilize translation.

If you use a separately pretrained LLM, you cannot simply plug its decoder weights into DeepSeek-OCR and expect them to work. You need an explicit interface and alignment training.

### My recommendation for translation

Use IndicTrans2 as your teacher and translation benchmark, not necessarily as the final separate decoder.

Useful resources\:

- [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2)
- [AI4Bharat model collection](https://huggingface.co/ai4bharat?utm_source=chatgpt.com)
- [BPCC parallel corpus](https://huggingface.co/datasets/ai4bharat/BPCC)

BPCC contains approximately 230M Indic-English bitext pairs, including 2.2M human gold-standard pairs. Samanantar is another useful resource with 49.7M English-Indic pairs across 11 Indic languages. \[[AI4Bharat](https://ai4bharat.iitm.ac.in/areas/nmt/?utm_source=chatgpt.com); [IndicTrans2/README.md at main · AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2/blob/main/README.md?plain=1&utm_source=chatgpt.com)\]

For your two target languages, I would prioritize\:

1. Hindi → English.
2. Marathi → English.
3. English → Hindi.
4. English → Marathi.

Use human-verified pairs for domain terminology and synthetic domain-document translations as additional supervision.

---

# 6\. Training 4 - Joint OCR + translation

Now combine the skills.

Your model should see a document image and generate both the source and target representations.

I would train with three types of examples\:

**Example A - Source transcription**

```json
{
  "task": "ocr",
  "source_text": "महाराष्ट्र शासन"
}
```

**Example B - Text translation**

```json
{
  "task": "translate",
  "source_text": "महाराष्ट्र शासन",
  "target_text": "Government of Maharashtra"
}
```

**Example C - Image to bilingual Document IR**

```json
{
  "task": "document_translate",
  "blocks": [
    {
      "bbox": [100, 50, 850, 120],
      "source": "महाराष्ट्र शासन",
      "target": "Government of Maharashtra",
      "type": "heading",
      "reading_order": 0
    }
  ]
}
```

Use your Gemini-generated `src`, `tgt` and bounding-box annotations here.

The important part is that the translation must correspond to the exact source block, not merely to the overall document.

### Suggested loss

$$
\mathcal{L} =
\lambda_1\mathcal{L}_{OCR}
+\lambda_2\mathcal{L}_{MT}
+\lambda_3\mathcal{L}_{layout}
+\lambda_4\mathcal{L}_{joint}
$$

I would begin with OCR and translation losses independently, then progressively increase the joint-document objective.

**Exit gate\:** The model must be at least as good as its separate OCR and translation baselines on held-out documents, with no meaningful degradation in source transcription.

---

# 7\. GPU requirements and training budget

Now the important part.

I am going to estimate this for a 3B DeepSeek-OCR-style model, using BF16 or QLoRA training, high-resolution document images, gradient checkpointing and FlashAttention where supported.

These are planning estimates, not measured throughput. Your actual requirements will depend heavily on image resolution, visual token count, output length, effective batch size and the implementation's custom attention kernels.

## 7.1 Hardware configurations

## Practical GPU choices

**1 × RTX A6000 48GB** (Good for LoRA)

- Connector and decoder LoRA.
- Small batch, gradient accumulation.
- Initial visual adaptation experiments.
- Limited full-resolution page training.

---

**1 × MI300X 192GB** (Excellent for experiments)

- BF16 LoRA and substantial model unfreezing.
- Full-page experiments.
- Potential single-GPU full fine-tuning for suitable model implementations.
- Requires a compatible ROCm training stack.

---

**1 × A100 80GB** (Good training baseline)

- Full-resolution LoRA.
- More comfortable batch sizes.
- Some full fine-tuning configurations with memory optimization.

---

**2-4 × A100 80GB / H100 80GB** (Research scale)

- Full fine-tuning.
- Larger batches and higher resolution.
- Faster domain training and ablations.

---

**8 × H100 80GB** (Large-scale training)

- Full model training at higher resolution.
- Extensive joint training.
- Large data throughput and multiple experiments.

Given your previously used A6000 and MI300X environments, I would use the MI300X for the first full-resolution experiments and the A6000 for debugging and smaller runs. This assumes your chosen model's kernels and training framework work correctly on ROCm.

## 7.2 Estimated GPU hours by stage

These estimates assume one 3B VLM, not a model trained entirely from random initialization.

| Stage | Data scale | Training strategy | Estimated GPU-hours |
|--------|--------|--------|--------|
| Stage 1: Visual adaptation | 1-3M word/line samples | LoRA + partial unfreezing | 80-250 |
| Stage 2: Document understanding | 30K-300K pages | LoRA / partial FT | 100-400 |
| Stage 3: Translation | 2-10M parallel pairs | Separate decoder training | 50-200 |
| Stage 4: Joint training | 30K-300K pages | LoRA / partial FT | 100-400 |
| Stage 5: Application SFT | 10K-50K verified examples | LoRA | 20-80 |
| Stage 6: RL | Selected difficult examples | Optional | 50-150 |
| Stage 7: Edge optimization | Benchmark runs | No major training | 10-40 |

These are rough aggregate GPU-hour planning ranges for the proposed experiments, not guarantees or published results. Translation hours assume a compact translation decoder or adapter, not full pretraining of a new multi-billion-parameter language model. Dataset sizes can overlap across stages.

(Recommended initial budget)

# 300-600

**GPU-hours**

First complete research cycle, without RL or long-document architecture.

# 1.5-2.5K

**GPU-hours**

Expanded experiments, additional fine-tuning and ablations.

These are project budgets across experiments, not minimum requirements for a successful OCR fine-tune.

### What if you only have 2K-3K documents?

This is an important distinction.

| Available data | What is realistic? |
|--------|--------|
| 2K-3K pages | Initial domain adaptation and proof of concept |
| 20K-50K pages | Useful domain-specific SFT |
| 100K-300K pages | Stronger document-domain adaptation |
| 1M+ word/line crops | Much better script recognition coverage |
| Millions of parallel text pairs | Stronger translation adaptation |

If you have 300K pages, you have a very useful domain dataset.

If you only have 3K pages, I would not expect a 3B VLM to become a strong Hindi/Marathi OCR model through that dataset alone. Use public recognition data and synthetic rendering to provide the missing visual coverage.

## 7.3 Full fine-tuning vs LoRA memory

For a 3B model, rough weight and training-state estimates\:

| Configuration | Approximate memory consideration |
|--------|--------|
| BF16 inference weights | 6GB |
| 4-bit inference weights | 1.5-2GB plus quantization metadata |
| BF16 full fine-tuning | Roughly 48GB for parameters, gradients and Adam states under a common 16-byte/parameter accounting |
| QLoRA | Much lower base-model memory; activations and vision features still matter |

These are simplified estimates and exclude some framework overhead, activations, KV cache and image feature buffers.

This is why an A6000 48GB is a sensible LoRA starting point but not a comfortable default for full training of a 3B VLM at high resolution.

# 8\. Translation and language datasets

You should build a separate language-data collection, not rely only on your domain documents.

| Dataset | Language coverage | Use |
|--------|--------|--------|
| [IndicLLMSuite / Sangraha](https://huggingface.co/datasets/ai4bharat/IndicLLMSuite?utm_source=chatgpt.com) | Indic languages | Monolingual language training |
| [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) | Hindi, Marathi and other Indic-English pairs | Translation |
| [Samanantar](https://ai4bharat.iitm.ac.in/areas/nmt/?utm_source=chatgpt.com) | English + 11 Indic languages | Translation |
| [FLORES-200](https://huggingface.co/datasets/ai4bharat/flores?utm_source=chatgpt.com) | Multilingual | Translation evaluation |
| [IndicTrans2 resources](https://github.com/AI4Bharat/IndicTrans2) | 22 Indic languages | Teacher and baseline |

For domain adaptation, also collect\:

- Marathi government terminology.
- Hindi government terminology.
- Legal and financial glossaries.
- Organization and person names.
- Common abbreviations.
- Dates, currency and measurement conventions.

Do not use all 230M BPCC pairs blindly. Filter for Hindi-English and Marathi-English, deduplicate, normalize Unicode and measure translation quality. Use the human-curated subset as a higher-quality component. \[[IndicTrans2/README.md at main · AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2/blob/main/README.md?plain=1&utm_source=chatgpt.com); [AI4Bharat](https://ai4bharat.iitm.ac.in/areas/nmt/?utm_source=chatgpt.com)\]

# 9\. What about the document layout datasets?

Your Gemini annotations are probably more valuable for domain-specific document layouts than generic public datasets.

But I would supplement them with\:

| Dataset | What to use it for |
|--------|--------|
| [IndicDLP](https://huggingface.co/datasets/IndicDLP/IndicDLP-dataset?utm_source=chatgpt.com) | Hindi/Marathi layout detection |
| [DocLayNet](https://github.com/DS4SD/DocLayNet?utm_source=chatgpt.com) | General page layout |
| [DocBank](https://github.com/IBM/DocBank?utm_source=chatgpt.com) | Fine-grained document layout |
| [PubTables-1M / Table Transformer](https://github.com/microsoft/table-transformer?utm_source=chatgpt.com) | Table structure |
| [FinTabNet](https://github.com/IBM/FinTabNet?utm_source=chatgpt.com) | Financial table structure |

IndicDLP contains 119,806 pages across 11 Indic languages and English, with 42 layout classes. It is especially relevant to your use case because it includes forms, acts, manuals, newspapers, brochures and other document types. \[[IndicDLP/IndicDLP-model](https://huggingface.co/IndicDLP/IndicDLP-model?utm_source=chatgpt.com); [IndicDLP: A Foundational Dataset for Multi-Lingual and Multi-Domain Document Layout Parsing](https://zenodo.org/records/15881917?utm_source=chatgpt.com)\]

One caveat: IndicDLP's layout labels do not automatically give you verified text transcription. Keep those tasks separate.

---

# 10\. Actual implementation order: first 90 days

I would not start all four training programmes simultaneously.

Here's the execution sequence I would follow.

# 90-day research execution plan

(Days 1-15)

**Baseline and data preparation**

- Download Mozhi Hindi and Marathi.
- Download IndicDLP.
- Prepare synthetic Devanagari word/line rendering.
- Establish 2K-5K verified document samples if available.
- Benchmark DeepSeek-OCR, Unlimited-OCR and PaddleOCR-VL.
- Build a golden test set.

Deliverable: Baseline report and data loader.

---

(Days 16-35)

**Visual adaptation**

- Train connector and decoder LoRA.
- Run word and line curriculum.
- Introduce synthetic degraded samples.
- Unfreeze selected visual encoder layers.
- Compare Hindi and Marathi CER.

Deliverable: Indic OCR checkpoint v1.

---

(Days 36-55)

**Domain document understanding**

- Add your Gemini-generated annotations.
- Train page parsing and layout.
- Train spotting and region reading.
- Add tables and structured extraction.
- Replay general OCR examples.

Deliverable: Domain OCR checkpoint v2.

---

(Days 56-70)

**Translation adaptation**

- Train a compact Hindi/Marathi-English translation decoder or adapter.
- Use BPCC and IndicTrans2 teacher outputs.
- Evaluate terminology, names and numbers.
- Compare shared versus separate decoder designs.

Deliverable: Translation checkpoint and integration decision.

---

(Days 71-90)

**Joint training and PDF reconstruction**

- Train image-to-bilingual-Document-IR.
- Validate source and target grounding.
- Integrate the PDF reconstruction harness.
- Test quantized inference.
- Benchmark on unseen document templates.

Deliverable: End-to-end document translation prototype.

These are engineering milestones, not promises that every stage will converge within its allotted days. The data volume and GPU throughput will determine the actual schedule.

---

# 11\. The experiments I want you to run before committing

There are three decisions that will substantially affect your architecture.

| Experiment | Comparison | What it tells you |
|--------|--------|--------|
| E1 | DeepSeek-OCR vs Unlimited-OCR 2 | Which backbone is easier to adapt and better for Indic OCR? |
| E2 | Frozen encoder vs partial unfreezing vs full visual adaptation | Does your target script need visual representation adaptation? |
| E3 | Shared decoder vs separate translation decoder | Is the additional decoder actually worth its memory and latency? |

For E1, use exactly the same images, prompts, output schema and evaluation metrics.

For E2, use the same training budget and compare per-language CER.

For E3, compare end-to-end translation quality, OCR preservation, total model memory and latency.

Do not let the model's general document-parsing benchmark score decide these experiments. Your own Hindi and Marathi evaluation set is the deciding factor.

# 12\. Final architecture decision

This is what I would commit to for V1.

(My recommended V1)

# IndicDoc-VLM

# 1.5B-2B dense VLM

Target size, subject to encoder and decoder ablations.

**Vision**

- Pretrained high-resolution visual encoder.
- OCR-specific continued adaptation.
- Partial unfreezing.

**Language**

- Existing multimodal decoder.
- Hindi and Marathi OCR specialization.
- Text-only translation adaptation.
- Shared OCR/translation decoder initially.

**Training**

- Mozhi + synthetic recognition data.
- IndicDLP + your Gemini-annotated pages.
- BPCC + verified domain translations.
- Joint image-to-bilingual-Document-IR SFT.

**Inference**

- Single multimodal model.
- External deterministic PDF reconstruction harness.
- Optional second-pass low-confidence correction.

**Deployment**

- BF16 development checkpoint.
- INT8/4-bit deployment experiments.
- Edge benchmarking.

## 13\. Three things I need to finalize the exact training configuration

There are three details that change the training architecture and GPU budget considerably.

**1\. Which base model do you want to start with?**

DeepSeek-OCR 3B

Unlimited-OCR 3B / 2

Compare both before selecting

---

**2\. What is your actual verified domain dataset size?**

2K-3K pages

20K-50K pages

1-3 lakh pages

Different counts for layout, OCR and translation

---

**3\. Which GPU resources can you allocate?**

1 × MI300X 192GB

1 × RTX A6000 48GB

2-8 × H100 80GB

Multiple GPU types / cloud rental

---

**One additional detail: what document domain comes first?**

Government forms, FIRs and legal documents

Financial documents, invoices and statements

Mixed documents from multiple domains

[Finalize my training configuration]

**My strongest advice\:** Don't spend your initial GPU budget training a new vision encoder or a new 3B language decoder. First prove that an existing OCR VLM can become substantially better at Hindi and Marathi with your recognition data and domain annotations. Then invest in custom architecture only where the experiments show a genuine weakness.

## Prompt:
11/17/2026, 11:42:58 PM

Finalize my Hindi and Marathi OCR training specification with these choices: Base model: deepseek. Verified dataset size: 3k. GPU resources: a6000. Initial document domain: mixed. Give me the exact model architecture, dataset mixture, LoRA/full fine-tuning configuration, GPU memory and hours, training commands and evaluation gates.

## Prompt:
11/17/2026, 11:43:22 PM

Finalize my Hindi and Marathi OCR training specification with these choices: Base model: deepseek. Verified dataset size: 3k. GPU resources: mixed. Initial document domain: government. Give me the exact model architecture, dataset mixture, LoRA/full fine-tuning configuration, GPU memory and hours, training commands and evaluation gates.

## Response:
11/17/2026, 11:43:30 PM · gpt-6

> **Evaluated GPU architectures, government use, and DeepSeek OCR support возможности**
>
> 
>
> **Planning dataset adaptation**
>
> I’m interpreting 3,000 as verified pages and proposing a 70/15/15 split: 2,100 training, 450 development, and 450 test pages. I’m balancing Hindi and Marathi, preserving page-level annotations, and combining public, synthetic, and government-document data.
>
> **Planned dataset adaptation**
>
> 
>
> **Estimating compute needs**
>
> I’m planning around a CUDA-based A6000 run, keeping MI300X for separate testing unless its software stack is validated. LoRA, gradient checkpointing, and small batches should make 48GB workable. I’m estimating roughly 60–150 GPU-hours overall, excluding rendering.
>
> **Estimated compute needs**
>
> 
>
> **Searched 18 websites**
>
> 
>
> Worked for 1m 8s

# IndicDoc-OCR: Final Hindi + Marathi training specification

(Recommended research configuration) (Government documents)

Your updated choices are\:

| Parameter | Decision |
|--------|--------|
| Base model | DeepSeek-OCR |
| Target languages | Hindi + Marathi |
| Verified domain dataset | 3,000 pages |
| Initial domain | Government documents |
| Hardware | Mixed: RTX A6000 + MI300X |
| Target architecture | Existing 3B DeepSeek-OCR VLM |
| Initial training | LoRA, then selective full fine-tuning |
| Translation | Joint OCR + translation experiments |
| Final PDF generation | External deterministic reconstruction harness |
| RL | Not in V1 |

I am interpreting your mixed GPU resources as the RTX A6000 48GB and AMD MI300X 192GB environments you've used previously.

There is one important adjustment to the earlier plan\:

**Do not train a new OCR foundation model from scratch, and do not train a new language decoder yet. Adapt DeepSeek-OCR's existing visual-language architecture.**

Your 3,000 verified pages are enough to start domain adaptation, but not enough to independently establish broad Hindi and Marathi visual recognition. We will supplement them with public recognition datasets and synthetic document rendering.

I would call the first checkpoint `IndicDoc-OCR-v0.1`\.

---

# 1\. Exact model architecture

## 1.1 Base: DeepSeek-OCR

The published architecture is\:

## DeepSeek-OCR architecture

**Document image**

1024-1280 px initial training resolution

**DeepEncoder - approximately 380M**

**SAM Base**

## 80M

Visual perception

**CLIP Large**

## 300M

Visual knowledge

**2-layer convolutional compressor**

16× visual token downsampling

**DeepSeek 3B MoE decoder**

# 3B total

Approximately 570M activated parameters per token, as reported in the paper

**OCR / Document IR / Extraction / Translation**

This architecture and parameter breakdown come from the original DeepSeek-OCR paper. \[[DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com); [\<visual\_element id="e1"\>](https://www.deepseek-ocr.ai/DeepSeek_OCR_paper.pdf?utm_source=chatgpt.com)\]

Official resources\:

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR?utm_source=chatgpt.com)
- [DeepSeek-OCR checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-OCR?utm_source=chatgpt.com)
- [Original research paper](https://arxiv.org/abs/2510.18234?utm_source=chatgpt.com)

One implementation warning: the official repository primarily provides inference code, not a turnkey supervised fine-tuning pipeline. You will need to implement and validate a custom training loop or a compatible framework integration. Do not assume that an ordinary Qwen-VL LoRA script will work unchanged. \[[GitHub - deepseek-ai/DeepSeek-OCR: Contexts Optical Compression](https://github.com/deepseek-ai/DeepSeek-OCR?utm_source=chatgpt.com); [DeepSeek-OCR/README.md at main · deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/README.md?utm_source=chatgpt.com)\]

## 1.2 What we will freeze and train

This is the exact V1 training policy.

| Component | Phase 1 | Phase 2 | Phase 3 |
|--------|--------|--------|--------|
| SAM Base | Freeze | Freeze | Partial unfreeze |
| CLIP Large | Freeze | Freeze | Last layers |
| Conv compressor | Freeze | Train | Train |
| MoE attention | LoRA | LoRA | LoRA |
| MoE experts | Freeze | Selected LoRA | Selected unfreeze if needed |
| Router | Freeze | Freeze | Freeze initially |
| Embeddings | Freeze | Freeze | Freeze |
| LM head | Freeze | Freeze | Freeze |
| Document output schema | Existing tokens | Existing tokens | Existing tokens |

Why?

Your initial dataset is too small to justify updating the entire visual encoder and language decoder immediately.

We want the pretrained model to retain its existing capabilities while learning Devanagari recognition and government-document conventions.

**Do not replace the tokenizer.** The existing decoder already has Devanagari token support. First measure its tokenization and recognition quality before considering any vocabulary changes.

---

# 2\. Dataset specification

I am treating your 3,000 verified records as 3,000 complete government-document pages, not 3,000 individual word crops.

We will construct three data layers.

## Data architecture

**Layer A - Public recognition data**

Mozhi, synthetic word/line images, optional handwriting.

(Visual adaptation)

**Layer B - Public layout data**

IndicDLP, DocLayNet, government-form synthetic pages.

(Document understanding)

**Layer C - Your verified government documents**

3,000 pages with bounding boxes, source text, target translation and extraction annotations.

(Domain supervision)

## 2.1 Public datasets to download

These are the priority order.

| Dataset | Hindi | Marathi | Usage |
|--------|--------|--------|--------|
| [Mozhi](https://cvit.iiit.ac.in/usodi/tdocrmil.php?utm_source=chatgpt.com) | \~80K words | \~80K words | Printed recognition |
| [IIIT-INDIC-HW-WORDS](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words?utm_source=chatgpt.com) | Yes | Check availability of separate Devanagari release | Optional handwriting |
| [ICDAR Indic HDR](https://ilocr.iiit.ac.in/icdar_2025_Indic_HDR/dataset.html?utm_source=chatgpt.com) | Yes | Yes | Optional handwriting pages |
| [IndicDLP](https://indicdlp.github.io/?utm_source=chatgpt.com) | Yes | Yes | Layout |
| [Sarvam Indic OCR Bench](https://huggingface.co/datasets/sarvamai/indic-ocr-bench?utm_source=chatgpt.com) | Yes | Yes | Evaluation only |

Mozhi has approximately 79.8K Hindi training words and 80.2K Marathi training words, plus line-level data. It is particularly suitable for your first stage. \[[Towards Deployable OCR Models for Indic Languages](https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in/dataset/static/assets/img/publication/printed/deployable_ocr.pdf?utm_source=chatgpt.com)\]

IndicDLP provides 119,806 pages, 42 layout classes, 12 languages and 12 document categories. It includes government-relevant categories such as Acts &amp; Rules, Forms and Notices. \[[IndicDLP : A Foundational Dataset for Multi-Lingual and Multi-Domain Document Layout Parsing](https://indicdlp.github.io/?utm_source=chatgpt.com); [IndicDLP/IndicDLP-model](https://huggingface.co/IndicDLP/IndicDLP-model?utm_source=chatgpt.com)\]

For the government domain, I would initially prioritize\:

- Acts and Rules.
- Government forms.
- Notices and circulars.
- Official letters.
- Certificates.
- Government reports.
- Bilingual documents.

Do not mix handwriting heavily into the first checkpoint. Make printed government documents your initial target.

## 2.2 Your 3,000-page dataset

First split by original document identity, not by page.

## Verified dataset split

# 2,100

Train

70%

# 450

Validation

15%

# 450

Test

15%

---

Illustrative split assuming 3,000 distinct pages. If pages belong to the same multi-page document, all pages must stay in one split.

Within the train split, aim for approximately equal Hindi and Marathi representation. Stratify by document type and language.

If the actual language distribution is unequal, use balanced sampling rather than duplicating the entire minority-language dataset.

### Annotation requirements

For each page, you should have\:

```json
{
  "id": "gov_00001",
  "image": "images/gov_00001.jpg",
  "language": "mar",
  "document_type": "government_notice",
  "page_size": [2480, 3508],
  "blocks": [
    {
      "id": "b001",
      "bbox": [120, 80, 2200, 280],
      "type": "heading",
      "reading_order": 0,
      "src": "महाराष्ट्र शासन",
      "tgt": "Government of Maharashtra",
      "verified_src": true,
      "verified_tgt": true
    }
  ]
}
```

The bounding boxes should use a consistent coordinate convention, preferably normalized to 0-1000 for model output, while retaining original pixel coordinates in your annotations.

You must validate\:

- Source text against the actual pixels.
- Translation against the source.
- Box alignment.
- Reading order.
- Blank fields.
- Language identification.

Gemini outputs that have not been verified should remain teacher predictions, not ground truth.

---

# 3\. Exact training stages and dataset mixtures

We will use four stages, with a fifth optional refinement stage.

## Stage 0 - Baseline and data preparation

(No GPU training)

Before training anything, run the original DeepSeek-OCR checkpoint on\:

1. 200 Hindi government pages.
2. 200 Marathi government pages.
3. 100 mixed Hindi-English pages.
4. 100 mixed Marathi-English pages.

Record\:

- CER and WER.
- Exact field extraction accuracy.
- Reading-order accuracy.
- Bounding-box IoU.
- Translation quality.
- Inference latency.

This becomes your baseline.

Also create a separate 100-page difficult-case set containing small fonts, stamps, skew, damaged scans, tables and low-contrast text.

**Deliverable\:** `baseline_deepseek_ocr.json`\.

## Stage 1 - Hindi and Marathi visual adaptation

(First training run)

The goal is to improve character recognition, not document reasoning.

### Dataset mixture

I would start with\:

## Stage 1 mixture

For every sample, use image plus exact transcription.

Use this curriculum\:

| Step | Image input | Output |
|--------|--------|--------|
| 1 | Word crops | Exact word |
| 2 | Line crops | Exact line |
| 3 | Paragraph crops | Exact paragraph |
| 4 | Small document regions | Transcription + optional box |

### Model configuration

| Parameter | Starting value |
|--------|--------|
| Vision encoder | Frozen |
| Compressor | Trainable |
| Decoder | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate, LoRA | 1e-4 |
| Learning rate, compressor | 5e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup | 5% |
| Scheduler | Cosine |
| Effective batch | 32 examples |
| Epochs | 2-3 |
| Precision | BF16 on supported hardware |

Start with the attention projections in the decoder: `q_proj`, `k_proj`, `v_proj`, and `o_proj`\.

Only add LoRA to expert projections if the attention-only experiment underfits. Inspect the actual module names in the loaded checkpoint; these names are targets to verify, not a guaranteed module map.

**Expected result\:** `indicdoc_stage1_lora`\.

Do not use your 3,000 pages as the only visual training data. Mozhi and synthetic rendering provide the character-level coverage that the government pages cannot.

## Stage 2 - Government document understanding

(Main domain adaptation)

This is where your 3,000 verified pages become central.

Train one multimodal model on the following tasks\:

1. Full-page transcription.
2. Full-page parsing.
3. Layout detection.
4. Text spotting.
5. Region reading.
6. Phrase grounding.
7. Table extraction.
8. Form extraction.

### Dataset mixture

## Stage 2 task mixture

Proposed sampling by training-example probability. This is not a fixed count of unique documents.

### How to populate these tasks

| Task | Data source |
|--------|--------|
| Full-page parsing | Your 3K pages + synthetic government pages |
| Page transcription | Your verified `src` \+ Mozhi rendered pages |
| Layout only | IndicDLP + your bounding boxes |
| Spotting | Synthetic word boxes + verified word boxes |
| Region reading | Crops generated from your bounding boxes |
| Grounding | Phrase-to-box annotations |
| Tables/forms | Your structured government documents |
| General OCR replay | Mozhi + synthetic OCR |

One important point: IndicDLP should primarily supervise layout classes and geometry. Its layout annotations do not automatically supply verified page transcription.

### Model configuration

| Parameter | Value |
|--------|--------|
| Encoder | Frozen initially |
| Compressor | Trainable |
| Decoder | LoRA |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Learning rate, LoRA | 5e-5 |
| Learning rate, compressor | 2e-5 |
| Effective batch | 4-8 pages |
| Resolution | 1024 initially |
| Later resolution | 1280-1536 |
| Max output | 4096 initially |
| Later output | 8192 |
| Epochs | 3-5 |
| Precision | BF16 |
| Gradient checkpointing | Enabled |

For pages with many small characters, don't force 16× compression if it destroys glyph details. Compare the original supported image-resolution modes and retain the high-detail crops where necessary.

The 1024-to-1536 progression is deliberate: establish stable document parsing before paying the memory cost of higher-resolution pages.

### Important: layout hints

You already have detected bounding boxes.

Use them in two ways\:

- 50% of page-parsing samples: supply detected boxes as input hints.
- 50%: provide only the page image.

Of the hinted samples, deliberately corrupt approximately 20% of the hints.

This teaches the model to work with your external layout detector without becoming dependent on its predictions.

**Deliverable\:** `indicdoc_stage2_domain`\.

## Stage 3 - Translation adaptation

This should be a separate experimental branch.

I would not immediately add another 3B decoder.

Use IndicTrans2 as a translation teacher and baseline, and adapt the existing DeepSeek decoder with text-only translation supervision.

### Dataset mixture

| Data | Share |
|--------|--------|
| BPCC Hindi-English | 40% |
| BPCC Marathi-English | 40% |
| Verified government-domain parallel text | 15% |
| Government terminology and glossary examples | 5% |

This is a proposed starting mixture. Filter for actual language-pair availability, deduplicate, and exclude low-quality synthetic translations.

Train with\:

```text
Task: translate
Source language: Marathi
Target language: English

Input:
महाराष्ट्र शासनाने नवीन परिपत्रक जारी केले.

Output:
The Government of Maharashtra has issued a new circular.
```

And the corresponding Hindi examples.

### Configuration

| Parameter | Value |
|--------|--------|
| Base decoder | DeepSeek 3B |
| Vision encoder | Not used |
| Training method | LoRA |
| LoRA rank | 32 |
| Learning rate | 5e-5 |
| Effective batch | 64 text sequences |
| Sequence length | 2048 |
| Epochs | 1-2 |
| Precision | BF16 |

This is text-only decoder adaptation. It is not visual OCR training.

Keep this as a separate checkpoint until you have compared it with the shared OCR/translation decoder.

## Stage 4 - Joint OCR + translation

(Main product checkpoint)

Now combine Stage 2 and Stage 3.

The model receives a document image and outputs a bilingual Document IR.

For example\:

```json
{
  "document_type": "government_notice",
  "language": "mar",
  "blocks": [
    {
      "bbox": [50, 20, 950, 100],
      "type": "heading",
      "source": "महाराष्ट्र शासन",
      "translation": "Government of Maharashtra",
      "reading_order": 0
    }
  ]
}
```

Train on\:

- Your verified bilingual blocks.
- Rendered BPCC translation documents.
- Synthetic government forms.
- Existing OCR replay.
- Blank and uncertain fields.

### Loss function

$$
\mathcal L =
0.40\mathcal L_{\text{OCR}}+
0.25\mathcal L_{\text{translation}}+
0.20\mathcal L_{\text{layout}}+
0.15\mathcal L_{\text{extraction}}
$$

These are initial experimental weights, not a proven optimum. Normalize task losses and inspect gradient magnitudes before treating the weights as comparable.

Use\:

- Stage 2 checkpoint as initialization.
- Decoder LoRA rank 32.
- Compressor trainable.
- Encoder frozen initially.
- Learning rate around $1\text{e-}5$\.
- One to two epochs initially.
- 10-20% general OCR replay.

Do not unfreeze the whole encoder simply because the joint task underperforms. First establish whether the failure comes from OCR, translation, layout or serialization.

## Stage 5 - Optional preference optimization

Do not allocate GPU time to RL in the first release.

Once the model produces valid Document IR, use verifiable rewards\:

| Objective | Reward |
|--------|--------|
| OCR | Character-level edit similarity |
| Layout | Box IoU |
| Reading order | Pairwise ordering accuracy |
| Extraction | Field-level F1 |
| Translation | Human/teacher quality evaluation and terminology checks |
| Grounding | Correct source block correspondence |

A wrong translation of a government order is not something you should reward merely because it sounds fluent.

---

# 4\. Exact LoRA and full fine-tuning strategy

This is the configuration I would actually use for your hardware.

## 4.1 Experiment matrix

| Experiment | Trainable parameters | Purpose |
|--------|--------|--------|
| E0 | None | Baseline |
| E1 | Compressor only | Visual alignment |
| E2 | Compressor + attention LoRA | Main initial experiment |
| E3 | Compressor + attention/expert LoRA | Stronger adaptation |
| E4 | E3 + final CLIP layers | Visual specialization |
| E5 | Selected full fine-tuning | Only if E4 demonstrates a need |

Do not start with E5.

## 4.2 LoRA settings

## V1 recommended configuration

# 32

LoRA rank

# 64

LoRA alpha

# 0.05

Dropout

# 1e-4

Initial adapter LR

---

**Attention target modules**

`q_proj`, `k_proj`, `v_proj`, `o_proj`

**Second experiment**

Add expert `gate_proj`, `up_proj`, `down_proj` modules after verifying the MoE module structure.

**Keep frozen**

SAM, CLIP, tokenizer, embeddings and LM head initially.

One nuance: DeepSeek-OCR uses an MoE decoder, so a generic LoRA target list may not reach all expert weights. Inspect the actual module names and verify trainable parameter counts before launching a long job.

## 4.3 Full fine-tuning

Only consider selective full fine-tuning after LoRA has converged.

My recommendation\:

1. Unfreeze the compressor.
2. Unfreeze the last 2-4 CLIP transformer blocks.
3. Keep SAM frozen.
4. Keep the majority of the MoE decoder frozen.
5. Keep the embeddings and LM head frozen.

Suggested learning rates\:

| Component | Learning rate |
|--------|--------|
| Compressor | 1e-5 |
| Last CLIP blocks | 1e-6 |
| Decoder | 2e-6 |
| Existing LoRA adapters | 5e-6 |

Use a short one-epoch comparison first.

Do not full-fine-tune all 3B parameters on 3,000 government pages. That is an unnecessarily expensive and high-overfitting-risk experiment.

---

# 5\. GPU architecture and memory budget

This is where your mixed hardware needs a little care.

You have two useful systems, but they should not automatically be treated as one distributed cluster.

[![GPU Options for ThinkSystem Servers > Lenovo Press](https://images.openai.com/static-rsc-4/chqEw0nD0tR4kH-t2DFfXHv7Xa8cCTGJBGpwYrAd_njksiXqp5QwUGDMzfuByhEx7WvubQuXK0vEZSMi4xh3tevNe995BZ64lx72_M0LIxflsBzk-fyEKkfWaHexbJFnPOPMpGtu_CdDZ2vRqdevpSEQtC75Ras4_DsO2s5y3FDz5pJ-UxQN9o2PEf6moDa-?purpose=fullsize)](https://lenovopress.lenovo.com/lp0767-gpu-options-for-thinksystem-servers)

**RTX A6000**

# 48GB

(Primary CUDA training machine)

- DeepSeek-OCR training experiments.
- BF16 LoRA.
- Gradient checkpointing.
- 1024px document training.

---

[![Die MI300 ist AMDs Hoffnungsträger für das Jahr 2024 | heise online](https://images.openai.com/static-rsc-4/ZLbGIQwHQrVRf1n7ELn2SEy4mhaOb8tQ_x2CC7Ri4Kl_CTuiEjmv1HquhpUgRvAfS4DeRmeY3Zu0BrzMMG8vQaOYZBhtOQm7lkM1fp3QGpguuwepxcXtjX7Q9TrEhxxcxAglcnkhewxGxNBABFTIoBEyTkNWE6IWOki4nUIlUT4_6xid_j9A09k90grLnn_-?purpose=fullsize)](https://www.heise.de/news/Konkurrenz-fuer-Nvidia-AMD-springt-auf-den-KI-Zug-auf-9614679.html)

**AMD MI300X**

# 192GB

(High-memory experiment machine)

- High-resolution experiments.
- BF16 experiments.
- Larger effective batches.
- Separate ROCm training compatibility tests.

## 5.1 Can we combine A6000 + MI300X?

Not as a normal single PyTorch DDP job.

They use different GPU execution stacks\:

- NVIDIA: CUDA.
- AMD: ROCm.

Use separate training runs, or port and validate a framework capable of heterogeneous distributed execution. Do not assume NCCL-based DeepSeek training will work across both.

My recommendation\:

| Workload | Hardware |
|--------|--------|
| Data preprocessing | CPU |
| Dataset rendering | CPU |
| Model baseline | A6000 |
| Stage 1 LoRA | A6000 |
| Stage 2 LoRA | A6000 |
| High-resolution ablation | MI300X, after compatibility validation |
| Full fine-tuning experiments | MI300X, if model and kernels work |
| Evaluation | Either machine, separately |

The DeepSeek-OCR repository documents a CUDA 11.8 / PyTorch 2.6 environment. The vLLM project also lists MI300X as a verified inference platform, but that does not establish that the original model's training kernels work on ROCm. \[[GitHub - deepseek-ai/DeepSeek-OCR: Contexts Optical Compression](https://github.com/deepseek-ai/DeepSeek-OCR?utm_source=chatgpt.com); [recipes/models/deepseek-ai/DeepSeek-OCR.yaml at main · vllm-project/recipes](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-OCR.yaml?utm_source=chatgpt.com)\]

## 5.2 Memory estimates

The following are planning estimates for training with the vision encoder frozen, LoRA adapters and checkpointing.

| Component | A6000 48GB | MI300X 192GB |
|--------|--------|--------|
| BF16 base model weights | \~6GB | \~6GB |
| Vision encoder | Included in base weights | Included in base weights |
| LoRA and optimizer states | 1-4GB | 1-4GB |
| Activations and image features | 12-22GB | 12-35GB |
| Gradients / temporary buffers | 2-5GB | 2-8GB |
| CUDA/ROCm and framework overhead | 3-6GB | 5-12GB |
| Expected working range | \~24-43GB | \~26-65GB |

These are approximate planning ranges, not measured DeepSeek-OCR profiling results. Actual memory depends on attention implementation, crop mode, output length, batch size and whether expert weights are materialized or quantized.

**Important\:** The A6000 configuration should begin with micro-batch 1. Do not start with batch 4 simply because the model has 570M active parameters. The entire MoE weight set still needs to be resident or otherwise available.

If the A6000 OOMs\:

1. Reduce image resolution.
2. Enable gradient checkpointing.
3. Reduce output length.
4. Use gradient accumulation.
5. Consider 4-bit frozen decoder weights only after verifying quantization compatibility with the custom MoE and multimodal implementation.

## 5.3 Exact initial training settings

**A6000 starting profile**

| Setting | Value |
|--------|--------|
| Micro-batch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 8 |
| Image long side | 1024 |
| Precision | BF16 |
| Gradient checkpointing | Yes |
| Optimizer | AdamW |
| Optimizer state | FP32 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| LoRA rank | 16 initially |
| Save interval | 250 steps |
| Evaluation interval | 250 steps |
| Max output tokens | 4096 |

For Stage 1 word and line training, you can increase the effective batch to 32 using gradient accumulation. For full-page document training, retain micro-batch 1.

---

# 6\. GPU-hour budget and training timeline

Let's estimate this specifically for your 3,000-page situation rather than a foundation model trained on tens of millions of pages.

## 6.1 Expected compute

| Experiment | A6000 GPU-hours |
|--------|--------|
| Baseline inference and profiling | 3-6 |
| Stage 1 visual LoRA | 15-35 |
| Stage 2 document LoRA | 20-50 |
| Stage 3 text-only translation | 5-15 |
| Stage 4 joint OCR + translation | 10-25 |
| Ablations and final evaluation | 5-15 |
| **Total** | **58-146** |

(Initial allocation)

**Reserve approximately 100-150 A6000 GPU-hours for V1.**

This is a planning allowance for a modest LoRA project, not a measured runtime guarantee.

The MI300X can be used for separate experiments, but I would not subtract hypothetical MI300X hours from the budget until you have measured throughput using the exact model and sequence configuration.

### What if we use full fine-tuning?

I would reserve another 30-100 GPU-hours for selective-unfreezing experiments, depending on how much of the visual encoder you unfreeze and the resulting throughput.

I would not budget for full 3B parameter fine-tuning in V1.

### Storage

Plan for\:

| Item | Suggested storage |
|--------|--------|
| Original document images | 20-50GB |
| Public OCR crops | 10-30GB |
| Synthetic images | 30-100GB |
| Cached processed datasets | 20-50GB |
| Checkpoints and adapters | 15-30GB |
| Evaluation outputs and logs | 5-10GB |

I would provision around 250GB for the initial research workspace.

---

# 7\. Actual implementation and commands

Now let's turn this into an executable project.

There is a distinction I want to be transparent about: DeepSeek publishes its inference implementation, but not a complete official multimodal LoRA training recipe. Therefore, the commands below give you the exact environment and data workflow, followed by a project-specific training entry point that you will need to implement and validate.

I would not give you a fabricated `swift sft` command and claim that it will train DeepSeek-OCR successfully.

## 7.1 Project directory

```text
IndicDoc-OCR/
│
├── configs/
│   ├── stage1_visual.yaml
│   ├── stage2_document.yaml
│   ├── stage3_translation.yaml
│   └── stage4_joint.yaml
│
├── data/
│   ├── mozhi/
│   ├── indicdlp/
│   ├── government/
│   │   ├── images/
│   │   ├── annotations/
│   │   └── splits/
│   ├── synthetic/
│   └── processed/
│
├── scripts/
│   ├── prepare_mozhi.py
│   ├── prepare_government.py
│   ├── generate_synthetic.py
│   ├── validate_annotations.py
│   └── build_conversations.py
│
├── training/
│   ├── model_loader.py
│   ├── multimodal_collator.py
│   ├── train_deepseek_ocr.py
│   └── evaluation.py
│
├── checkpoints/
│
└── experiments/
```

## 7.2 Environment setup: A6000

Start with the original repository's documented CUDA environment.

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git

conda create -n indicdoc python=3.12.9 -y
conda activate indicdoc

cd DeepSeek-OCR
```

Install the documented baseline dependencies\:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation

pip install \
  peft \
  datasets \
  accelerate \
  tensorboard \
  pydantic \
  opencv-python \
  pillow \
  jiwer \
  albumentations
```

The exact combination of PEFT, Transformers and the custom model code must be smoke-tested before training. Preserve the model's original custom implementation and avoid blindly upgrading Transformers.

Verify\:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())
print("Memory:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
```

## 7.3 Download the public datasets

```bash
mkdir -p data/{mozhi,indicdlp,synthetic,government}
```

For IndicDLP\:

```bash
pip install -U huggingface_hub

hf download IndicDLP/IndicDLP-dataset \
  --repo-type dataset \
  --local-dir data/indicdlp
```

The dataset is gated, so accept its access conditions first.

For Sarvam evaluation\:

```bash
hf download sarvamai/indic-ocr-bench \
  --repo-type dataset \
  --local-dir data/indic_ocr_bench
```

For Mozhi, use its official dataset portal and download the Hindi and Marathi printed recognition releases. The exact archive layout depends on the selected release.

## 7.4 Prepare the verified data

Your annotation validator should enforce\:

```python
required_fields = ["id", "image", "language", "document_type", "blocks"]

required_block_fields = ["bbox", "type", "reading_order", "src"]
```

For joint translation, require `tgt` and its verification status as well.

Run\:

```bash
python scripts/validate_annotations.py \
  --input data/government/annotations \
  --check-images \
  --check-bboxes \
  --check-reading-order \
  --check-source-text \
  --check-translation \
  --split-by-document \
  --output data/processed/validated.jsonl
```

This is the proposed CLI for your own validation script, not a built-in DeepSeek command.

## 7.5 Stage 1 training command

Your custom training implementation should accept\:

```bash
torchrun --standalone --nproc_per_node=1 \
  training/train_deepseek_ocr.py \
  --config configs/stage1_visual.yaml \
  --model deepseek-ai/DeepSeek-OCR \
  --dataset data/processed/stage1_visual.jsonl \
  --output checkpoints/indicdoc-stage1 \
  --bf16 \
  --gradient-checkpointing \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --epochs 3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --learning-rate 1e-4
```

The script must implement\:

- Loading the DeepSeek-OCR custom model.
- Freezing SAM and CLIP.
- Enabling gradients on the compressor.
- Injecting LoRA into verified decoder modules.
- Loading the model-specific image inputs.
- Calculating loss only on assistant target tokens.
- Saving adapter weights and optimizer state.

**Do not run this command until the model loader and collator have passed the smoke tests described below.**

## 7.6 Stage 2 training command

```bash
torchrun --standalone --nproc_per_node=1 \
  training/train_deepseek_ocr.py \
  --config configs/stage2_document.yaml \
  --model checkpoints/indicdoc-stage1 \
  --dataset data/processed/stage2_document.jsonl \
  --output checkpoints/indicdoc-stage2 \
  --bf16 \
  --gradient-checkpointing \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 5e-5
```

Again, `--model` must load the base checkpoint together with the Stage 1 adapter, not treat the adapter directory as a standalone model.

## 7.7 Stage 3 and Stage 4

Keep translation training separate\:

```bash
python training/train_translation_adapter.py \
  --base-model checkpoints/indicdoc-stage2 \
  --dataset data/processed/translation.jsonl \
  --languages hin,mar \
  --target-language eng \
  --lora-r 32 \
  --learning-rate 5e-5 \
  --epochs 2 \
  --output checkpoints/indicdoc-translation
```

Then joint training\:

```bash
torchrun --standalone --nproc_per_node=1 \
  training/train_deepseek_ocr.py \
  --config configs/stage4_joint.yaml \
  --model checkpoints/indicdoc-stage2 \
  --dataset data/processed/stage4_joint.jsonl \
  --output checkpoints/indicdoc-v0.1 \
  --bf16 \
  --gradient-checkpointing \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 2 \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 1e-5
```

I would initially use the shared decoder path for joint training, while keeping the separate translation adapter as a comparison.

## 7.8 Mandatory smoke tests before training

Before launching the first epoch, run these four tests\:

| Test | Expected result |
|--------|--------|
| Load model | DeepSeek-OCR loads without custom-code errors |
| Freeze check | Only intended compressor and adapter parameters require gradients |
| Forward pass | One image produces finite loss |
| Backward pass | Gradients reach LoRA and compressor parameters |

Also inspect\:

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
```

Verify that\:

- SAM is frozen.
- CLIP is frozen.
- Compressor is trainable.
- LoRA parameters are trainable.
- No unintended decoder or embedding weights are trainable.

Save a one-batch checkpoint and verify that it can be reloaded before starting a long run.

---

# 8\. Evaluation gates

This is the most important part of the entire project.

You should not move to the next stage simply because the training loss decreases.

## 8.1 Evaluation datasets

Use three separate evaluation sources\:

| Dataset | Purpose |
|--------|--------|
| Your 450-page government test set | Main product evaluation |
| [Sarvam Indic OCR Bench](https://huggingface.co/datasets/sarvamai/indic-ocr-bench?utm_source=chatgpt.com) | Independent Hindi/Marathi character recognition |
| Mozhi official test splits | Printed word and line recognition |

Sarvam's benchmark contains 6,909 manually reviewed text-block samples across 23 languages, including 228 Hindi and 300 Marathi samples. It is an excellent independent language benchmark, but it does not replace your government-page test set. \[[README.md](https://huggingface.co/datasets/sarvamai/indic-ocr-bench/blob/main/README.md?code=true&utm_source=chatgpt.com); [Sarvam Vision 2.1 document intelligence](https://www.sarvam.ai/blogs/sarvam-vision-2-1?utm_source=chatgpt.com)\]

## 8.2 Stage-wise exit gates

These are my proposed initial acceptance thresholds, not reported DeepSeek performance numbers.

| Metric | Stage 1 | Stage 2 | Final V1 |
|--------|--------|--------|--------|
| Printed line CER | ≤3% target | ≤3% | ≤3% |
| Government page CER | Measure baseline | ≤5% target | ≤5% |
| Layout mAP\@50 | - | ≥0.85 | ≥0.85 |
| Bounding-box IoU | - | ≥0.80 | ≥0.80 |
| Extraction field F1 | - | ≥0.85 | ≥0.90 |
| Reading-order accuracy | - | ≥0.90 | ≥0.95 |
| Translation faithfulness | - | Human-reviewed | Human-reviewed |
| Valid structured JSON | - | ≥98% | ≥99% |

A few qualifications\:

- CER must be reported separately for Hindi and Marathi.
- Report both normalized and raw transcription scores.
- Do not normalize away meaningful matras, nukta distinctions or character substitutions.
- Translation must preserve names, dates, amounts and legal meaning.
- Confidence intervals are important because 450 test pages are not necessarily 450 independent documents.

**My most important gate\:** Your adapted model must outperform the original DeepSeek-OCR on the same government-document test set, without damaging its general OCR capability.

For release, I would also require\:

- No silent deletion of document blocks.
- No invented extraction fields.
- All uncertain text flagged.
- No broken Devanagari shaping in the reconstructed PDF.
- No text clipping or accidental overlaps.

## 8.3 Automated evaluation

Your evaluator should produce something like\:

```json
{
  "model": "indicdoc-v0.1",
  "language": "mar",
  "ocr": {
    "cer": 0.042,
    "wer": 0.091
  },
  "layout": {
    "map50": 0.87,
    "mean_iou": 0.82
  },
  "extraction": {
    "field_f1": 0.91
  },
  "translation": {
    "human_review_required": true
  }
}
```

These numbers are illustrative output values, not predictions of your model's performance.

---

# 9\. What I would deliberately NOT build in V1

Your Architecture v2 has several interesting ideas, but they should remain experiments.

| Feature | V1 decision | Reason |
|--------|--------|--------|
| MoE redesign | Don't modify | Existing DeepSeek MoE is sufficient |
| New tokenizer | Don't modify | Avoid disturbing pretrained language capabilities |
| Separate 3B translation decoder | Defer | Adds memory and integration complexity |
| Latent document memory | Defer | Not needed to validate single-page OCR |
| Adaptive decoding loop | Defer | Additional complexity before baseline |
| RL/GRPO | Defer | First establish supervised quality |
| Custom vision encoder | Defer | Prove the existing encoder is insufficient first |

Your original Indic OCR Foundation Model plan already proposes separating visual alignment, document understanding, instruction tuning and optional reinforcement. I would retain that ordering but remove the from-scratch text-pretraining requirement because we are reusing DeepSeek's pretrained decoder. fileciteturn4file0

I would also retain the original plan's rule of splitting evaluation by source document and keeping the test set untouched. fileciteturn4file4

---

# 10\. Your first 30 days

I would use this as the immediate execution plan rather than trying to complete all five stages at once.

# First 30-day execution checklist

1. **Days 1-5: Baseline**
   
   - [ ] Set up DeepSeek-OCR on A6000.
   - [ ] Test 200 Hindi and 200 Marathi pages.
   - [ ] Calculate CER/WER.
   - [ ] Validate existing Gemini annotations.
   - [ ] Freeze the 450-page test split.
   
   (Deliverable: Baseline report)
2. **Days 6-10: Data preparation**
   
   - [ ] Download Mozhi.
   - [ ] Download IndicDLP.
   - [ ] Generate synthetic Hindi/Marathi words and lines.
   - [ ] Create Stage 1 JSONL.
   - [ ] Validate image/annotation alignment.
   
   (Deliverable: Visual training dataset)
3. **Days 11-18: Visual adaptation**
   
   - [ ] Train compressor-only experiment.
   - [ ] Train attention LoRA experiment.
   - [ ] Compare both checkpoints.
   - [ ] Evaluate on Mozhi and Sarvam.
   - [ ] Check general OCR regression.
   
   (Deliverable: Indic OCR v0.1)
4. **Days 19-25: Domain adaptation**
   
   - [ ] Prepare full-page parsing samples.
   - [ ] Add layout-only IndicDLP samples.
   - [ ] Add spotting and region-reading tasks.
   - [ ] Train document LoRA.
   - [ ] Evaluate layout and page CER.
   
   (Deliverable: Government OCR v0.2)
5. **Days 26-30: Translation experiment**
   
   - [ ] Prepare BPCC Hindi-English and Marathi-English pairs.
   - [ ] Train text-only translation adapter.
   - [ ] Run image-to-bilingual-IR inference.
   - [ ] Compare shared and separate decoder outputs.
   - [ ] Generate translated PDFs using the reconstruction harness.
   
   (Deliverable: End-to-end prototype)

The 30-day schedule is an engineering target. The 100-150 GPU-hour allocation covers the larger initial training/evaluation cycle, not necessarily all of it within those 30 calendar days.

---

# 11\. My final recommendation

I would lock the following decisions\:

# IndicDoc-OCR v0.1

(Approved starting specification)

**Foundation\:** DeepSeek-OCR 3B MoE

**Languages\:** Hindi + Marathi

**Domain\:** Government documents

**Vision\:** Existing DeepEncoder, frozen initially

**Adaptation\:** Compressor + decoder LoRA

**Recognition data\:** Mozhi + synthetic Devanagari

**Layout data\:** IndicDLP + your annotations

**Domain data\:** 3,000 verified pages

**Translation\:** Shared decoder first; separate adapter experiment

**Training\:** Four supervised stages

**Hardware\:** A6000 primary; MI300X separate compatibility experiments

**Initial compute allocation\:** 100-150 A6000 GPU-hours

**Final output\:** Grounded bilingual Document IR, followed by PDF reconstruction.

## 12\. Three research decisions worth making before your first run

There are three things I would investigate alongside the training.

1\. **DeepSeek's visual token compression\:** Test whether the 16× compression loses Devanagari matras and conjunct details. Compare native image modes and high-resolution crops before modifying the encoder.

2\. **Teacher agreement\:** Measure Gemini's actual Hindi and Marathi transcription and translation accuracy on your verified set. Don't assume its annotations are equally reliable across both languages.

3\. **Translation fidelity\:** Test whether a shared decoder preserves source text accurately when it translates. A fluent translation is not evidence of correct OCR.

The first experiment I would personally prioritize is the compressor-only versus attention-LoRA comparison. It is inexpensive relative to full fine-tuning and tells you whether the existing visual representations are sufficient for Devanagari.

**One last thing\:** Your 3,000 government pages are your domain-specialization asset. Mozhi teaches the model how to read Hindi and Marathi; IndicDLP teaches it where document elements are; your verified pages teach it what government documents actually contain. Those three sources should work together rather than compete with each other.

## Prompt:
11/17/2026, 11:49:22 PM

also the last idea part i mean training the same ocrs llm decoder seperately with psame tokenizer on extensively on text only translation task and connected it back to the ocr model for joint training

## Response:
11/17/2026, 11:49:30 PM · gpt-6

> ****
>
> 
>
> Worked for 5s

Ahh, now I understand exactly what you're proposing. And yes - this is a much more interesting training strategy than simply adding a translation adapter to the OCR model.

You want to **take the language decoder from DeepSeek-OCR, create a separate copy, extensively train that copy on Hindi/Marathi text and translation, and then bring its learned language capabilities back into the original OCR model.**

Essentially, you want two parallel training branches that eventually converge into one multimodal OCR-translation model.

I would incorporate this into the architecture.

# 1\. Your proposed architecture

## Dual-branch decoder adaptation

**DeepSeek-OCR pretrained checkpoint**

Same decoder initialization and tokenizer

(Branch A)

**Visual OCR branch**

SAM + CLIP

Visual compressor

**OCR decoder**

Visual alignment + OCR

(Branch B)

**Language branch**

Text-only input

Same tokenizer

**Decoder copy**

Hindi/Marathi continued training + translation

**Weight/adapter integration**

(Final model)

**IndicDoc-VLM**

# OCR + Understanding + Translation

One multimodal inference model, one shared tokenizer

The important part is that Branch B is not a completely new language model. It starts from the exact same pretrained decoder weights as Branch A.

# 2\. How I would train Branch B

I would call this stage\:

**Indic Language Decoder Continued Pretraining (IL-DCP).**

You will train the decoder separately, without images.

### Dataset mixture

I would use four categories of data.

| Dataset | Purpose | Starting share |
|--------|--------|--------|
| IndicLLMSuite / Sangraha | Hindi and Marathi language modelling | 35% |
| BPCC | Hindi-English and Marathi-English translation | 40% |
| Verified government bilingual corpus | Domain translation | 15% |
| Structured government text | Forms, tables, legal terminology, document schemas | 10% |

These are starting sampling proportions, not claims about the optimal mixture.

Resources\:

- [IndicLLMSuite](https://huggingface.co/datasets/ai4bharat/IndicLLMSuite?utm_source=chatgpt.com)
- [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC?utm_source=chatgpt.com)
- [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2?utm_source=chatgpt.com)
- [Sangraha research paper](https://arxiv.org/abs/2403.06350?utm_source=chatgpt.com)

### Training objectives

Do not train only translation.

Use three complementary objectives\:

| Objective | Example |
|--------|--------|
| Causal language modelling | Continue a Marathi government circular |
| Translation | Hindi → English, Marathi → English |
| Structured generation | Generate valid government-document JSON from text |

Example\:

```text
<task:translate>
<source_lang:mar>
<target_lang:eng>

महाराष्ट्र शासनाने नवीन परिपत्रक जारी केले.
```

Target\:

```text
The Government of Maharashtra has issued a new circular.
```

For the language-modelling objective\:

```text
<task:language_modeling>
<lang:mar>

महाराष्ट्र शासनाच्या सामान्य प्रशासन विभागाने...
```

This teaches both language generation and translation, rather than only memorizing parallel sentence pairs.

# 3\. Exact decoder training configuration

For the first experiment, I would use a separate LoRA branch.

| Parameter | Configuration |
|--------|--------|
| Initialization | Exact DeepSeek-OCR decoder checkpoint |
| Tokenizer | Original DeepSeek tokenizer |
| Embeddings | Frozen initially |
| LM head | Frozen initially |
| Vision encoder | Not loaded |
| Training | LoRA + continued training |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Dropout | 0.05 |
| Learning rate | 2e-5 |
| Sequence length | 2048 initially |
| Effective batch | 64 sequences |
| Precision | BF16 |
| Epochs | 1-3, based on token budget |
| Optimizer | AdamW |

One important distinction: use the exact decoder implementation and preserve its MoE routing and expert weights. The tokenizer alone is not enough to make two independently trained decoders compatible.

You should also retain 5-10% general-language replay from the initial checkpoint's training distribution, where legally available, to reduce language-capability regression.

# 4\. How do we connect the two branches?

This is the most important architectural decision.

There are three ways.

## Option A - Adapter integration

(My recommendation for V1)

Keep the original OCR decoder unchanged and attach the language-trained LoRA adapter.

```text
Original OCR decoder
        |
        +---- OCR adapter
        |
        +---- Indic translation adapter
        |
        +---- Shared embeddings
```

Use task-conditioned adapter routing\:

- OCR tasks activate the OCR adapter.
- Translation tasks activate the translation adapter.
- Joint OCR-translation tasks activate both through a controlled composition.

You don't necessarily need two independent adapters in every layer. You can share the attention adaptation and keep task-specific expert adapters where experiments show they help.

Advantages\:

- No destructive overwrite of OCR weights.
- Easy to compare the original and adapted decoder.
- Easy to disable translation adaptation.
- Low additional parameter cost.

## Option B - Merge the language branch into the OCR decoder

After training, merge the learned LoRA update\:

$$
W_{\text{merged}} =
W_{\text{OCR}} + \lambda \Delta W_{\text{translation}}
$$

Start with $\lambda=0.25$, then test 0.5 and 1.0.

This creates a single modified decoder without requiring a separate adapter at inference.

However, merging does not guarantee that OCR capabilities survive. Test OCR regression before accepting the merged weights.

## Option C - Full decoder weight transfer

Copy the full language-trained decoder weights into the OCR model.

This is possible if both branches started from identical decoder weights and retained exactly compatible architecture and parameter shapes.

But it is the riskiest option because extensive translation training can change the decoder's visual grounding and OCR behaviour.

I would not use it as your first experiment.

---

# 5\. The final joint training stage

Once Branch B has converged, connect it to the visual branch and perform multimodal joint training.

This is where the model learns to do both operations in one forward pipeline.

**Input**

Government document image

**Shared visual encoder**

**Adapted decoder**

OCR and language adaptations available

**Source extraction**

Hindi / Marathi + boxes

**Translation**

English + aligned blocks

**Bilingual Document IR**

Source text + translated text + coordinates + reading order

### Joint training mixture

| Task | Share |
|--------|--------|
| Visual OCR and transcription | 30% |
| Full-page document parsing | 20% |
| Text-only translation replay | 20% |
| Image-to-translation | 15% |
| Structured extraction | 10% |
| General language replay | 5% |

The translation replay is important. Otherwise, the model can lose some of the translation capability it learned in Branch B while adapting to image-conditioned outputs.

### Joint loss

$$
\mathcal L =
0.35L_{\text{OCR}}
+0.25L_{\text{translation}}
+0.20L_{\text{layout}}
+0.15L_{\text{extraction}}
+0.05L_{\text{language}}
$$

Start with these weights, then tune based on per-task validation performance and gradient magnitudes.

# 6\. One important experiment: Does decoder transfer actually help?

I want you to run three experiments rather than assume the separate training branch is beneficial.

| Experiment | Architecture |
|--------|--------|
| E1 | Original DeepSeek-OCR + domain LoRA |
| E2 | OCR + separately trained translation adapter |
| E3 | OCR + merged translation decoder + joint multimodal tuning |

Evaluate all three on\:

- Hindi CER.
- Marathi CER.
- Government document extraction F1.
- Translation adequacy.
- Translation terminology consistency.
- OCR-to-translation faithfulness.
- Peak VRAM and latency.

The key measurement is whether E2 or E3 improves translation without increasing OCR errors.

---

# 7\. Revised training roadmap

# IndicDoc dual-branch training

1. **Branch A - Visual adaptation**
   
   DeepSeek visual encoder + compressor + OCR decoder LoRA.
   
   Output: OCR-specialized checkpoint.
2. **Branch B - Language adaptation**
   
   Decoder copy + original tokenizer + Hindi/Marathi continued training and translation.
   
   Output: Translation-specialized decoder adapter.
3. **Integration**
   
   Load both branches and activate their learned adaptations in the same decoder.
   
   Output: Combined OCR-language checkpoint.
4. **Joint multimodal SFT**
   
   Image-to-source text, image-to-translation and image-to-Document-IR tasks.
   
   Output: IndicDoc-VLM v0.1.
5. **PDF reconstruction**
   
   External deterministic rendering, font management, coordinate mapping and validation.
   
   Output: Grounded translated PDF.

**My revised compute recommendation\:** Reserve approximately 100-150 A6000 GPU-hours for the initial OCR adaptation, language-adapter training and joint experiments, then revise the budget using measured tokens/second. The language branch should be considerably cheaper than training a complete multimodal model because it does not load or backpropagate through the visual encoder.

## My one architectural recommendation

I would keep one physical decoder at inference, with two independently learned adaptation paths during training.

That gives you the benefit of extensive text-only Hindi/Marathi training without having to maintain two separate 3B decoders in your edge deployment.

And importantly, it preserves your original goal: **a single model that sees a government document, accurately reads it, understands it, translates it and returns grounded structured output for PDF reconstruction.**

## Prompt:
11/18/2026, 7:27:01 PM

create a list of of sequencial training methods and datasets for each in a quick table for easier locate

## Response:
11/18/2026, 7:27:06 PM · gpt-6

# IndicDoc-OCR: Sequential training roadmap

**Base model\:** DeepSeek-OCR 3B &#124; **Languages\:** Hindi + Marathi &#124; **Domain\:** Government documents &#124; **Target\:** Single model, OCR + translation + document reconstruction.

The two decoder branches (visual and language) can be trained independently after the initial baseline, then combined.

| Order | Training stage | Training method | Datasets / data sources | Output |
|--------|--------|--------|--------|--------|
| 0 | Baseline evaluation | Zero-shot inference | 3K verified government pages; Mozhi test sets | Baseline metrics |
| 1A | Indic visual adaptation | Multimodal SFT + LoRA | [cvit.iiit.ac.in](https://cvit.iiit.ac.in/usodi/tdocrmil.php?utm_source=chatgpt.com), synthetic Hindi/Marathi words and lines, verified text crops | OCR decoder v1 |
| 1B | Indic language adaptation | Text-only continued pretraining (CPT) + LoRA | [huggingface.co](https://huggingface.co/datasets/ai4bharat/IndicLLMSuite?utm_source=chatgpt.com), Hindi/Marathi monolingual corpora | Language decoder v1 |
| 2 | Translation specialization | Supervised fine-tuning (SFT) | [huggingface.co](https://huggingface.co/datasets/ai4bharat/BPCC?utm_source=chatgpt.com), [ai4bharat.iitm.ac.in](https://ai4bharat.iitm.ac.in/areas/nmt/?utm_source=chatgpt.com), verified government translations | Translation decoder v1 |
| 3 | Domain document understanding | Multimodal SFT + LoRA | [indicdlp.github.io](https://indicdlp.github.io/?utm_source=chatgpt.com), [github.com](https://github.com/DS4SD/DocLayNet?utm_source=chatgpt.com), your 3K verified pages | Document OCR v2 |
| 4 | Decoder integration | LoRA composition / weight merging + joint tuning | OCR branch + language branch + BPCC + government bilingual annotations | Combined decoder |
| 5 | Joint OCR + translation | Multitask multimodal SFT | Your 3K pages: image + bbox + source + target + extraction JSON | IndicDoc-VLM v1 |
| 6 | Application specialization | Instruction SFT | Government forms, notices, circulars, FIRs, certificates, legal documents | Product model |
| 7 | Preference optimization | DPO / optional GRPO with verifiable rewards | Human-reviewed outputs, OCR references, bbox references, translation evaluations | Reliability improvement |
| 8 | Edge optimization | Quantization + distillation + inference optimization | Training and evaluation corpus; teacher/student outputs | Edge deployment |

## Dataset locator by purpose

| Dataset | Primary purpose | Priority |
|--------|--------|--------|
| [cvit.iiit.ac.in](https://cvit.iiit.ac.in/usodi/tdocrmil.php?utm_source=chatgpt.com) | Hindi/Marathi printed word and line recognition | (Essential) |
| [huggingface.co](https://huggingface.co/datasets/ai4bharat/IndicLLMSuite?utm_source=chatgpt.com) | Indic language continued pretraining | (Essential) |
| [huggingface.co](https://huggingface.co/datasets/ai4bharat/BPCC?utm_source=chatgpt.com) | Hindi-English / Marathi-English translation | (Essential) |
| [ai4bharat.iitm.ac.in](https://ai4bharat.iitm.ac.in/areas/nmt/?utm_source=chatgpt.com) | Additional parallel translation data | Recommended |
| [indicdlp.github.io](https://indicdlp.github.io/?utm_source=chatgpt.com) | Indic document layout | (Essential) |
| [github.com](https://github.com/DS4SD/DocLayNet?utm_source=chatgpt.com) | General document layout | Recommended |
| [github.com](https://github.com/microsoft/table-transformer?utm_source=chatgpt.com) | Table structure recognition | Recommended |
| Your Gemini annotations | Domain OCR, translation, grounding and extraction | (Essential) |
| [huggingface.co](https://huggingface.co/datasets/facebook/flores?utm_source=chatgpt.com) | Translation evaluation | Recommended |
| [huggingface.co](https://huggingface.co/datasets/sarvamai/indic-ocr-bench?utm_source=chatgpt.com) | Indic OCR evaluation | Recommended |

## Decoder training: keep this distinction clear

**Branch A - Visual decoder**

- Initialize from DeepSeek-OCR.
- Train with image + transcription.
- Learn visual recognition and OCR.

---

**Branch B - Language decoder**

- Clone the same pretrained decoder.
- Preserve the exact tokenizer.
- Train on monolingual Hindi/Marathi CPT.
- Continue with bilingual translation SFT.

---

**Integration**

- Start with LoRA adapter composition.
- Compare against adapter merging.
- Jointly fine-tune using image + source + translation.
- Evaluate OCR regression before accepting the merged model.

**My recommended execution order\:** Train 1A and 1B as parallel branches after Stage 0. Continue 1B into Stage 2, and 1A into Stage 3. Then integrate at Stage 4.

One important distinction: Stage 1B is language-model continued pretraining, while Stage 2 is supervised translation training. Don't treat them as the same objective.

This gives you a clean research progression without training two separate 3B models from scratch.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)