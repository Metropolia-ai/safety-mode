# Meta Model API: notes for Safety Mode (taken from dev.meta.ai 18.10.2026)

Base URL `https://api.meta.ai/v1`, header `Authorization: Bearer $MODEL_API_KEY`. Compatible with the OpenAI SDK (base_url + key).

## Models
`muse-spark-1.3` (also 1.2, 1.1; there is a `-contributor` tier), `muse-wonder-1.0` (preview), `muse-image-1.0`, `muse-voice-transcribe-1.0`, `sam-3.1`. Spark: 1M token context, parallel tool calls, structured output.

## Muse Spark, structured output
Chat Completions: `response_format={"type":"json_schema","json_schema":{"name":..., "strict":true, "schema":{...}}}`. Strict mode: the root is an object, `additionalProperties:false` everywhere, `required` lists all properties, no `allOf/oneOf`, `anyOf` is allowed below the root. The answer is in `choices[0].message.content`, parse it with `json.loads`. For the Responses API the parameter is `text.format` (the docs give no example).

## SAM 3.1 (checked against dev.meta.ai/docs/sam/segmenting, reading-segmentation, client-libraries, pricing-rate-limits and live calls on 27.10.2026)
- Endpoint: `POST /v1/responses`, model `sam-3.1`. Both ways work. `client.responses.create(...)` without streaming returns all text in `output_text` (tested, `safety_mode/detect/` works this way). `stream=True` returns one `response.output_text.delta` per frame. `usage`: 0 tokens, `images_processed=1`.
- Input: `input_text` with one noun + `input_image` (`image_url`, URL or data-URL) or `input_video` (`video_url` or `file_id` from `POST /v1/files`, purpose `user_data`, the upload is synchronous). `input_video` with `file_id` tested on 27.10.
- **One object per request.** Lists, "and", commands, questions, negations and spatial phrases are unreliable. N objects = N requests on the same frame/clip.
- ~~Text prompts only~~ **Corrected 27.10:** the API also has box/point prompts (up to 16 objects, `<Nf>id<|box;...|>` / `<|point;...|>` in `input_text`, `-` before a point = negative point, every token carries `w`/`h` of the media size). The answer to them is always `one_bit`, and frames may come out of order. We do not need this yet.
- `metadata` (strings): `mask_encoding` = `lossless` (default, marker `~`) or `one_bit` (`!`, 20–25 times shorter on our masks); `include_confidence` = `"true"` adds `;c=<0..1>` to box and mask; `propagation_direction` (video with box/point only).
- Video: MP4/MOV, H.264/H.265/MPEG-4 (not VP9/WebM), up to 15 000 frames (~10 min) per request, up to 16 objects per frame, `-movflags +faststart` (otherwise the server remuxes the file itself).
- Output: one line per frame `<Nf>` (N from zero), then comma-separated records `id<|box;x1;y1;x2;y2;w;h[;c]|><|mask;x=0;y=0[;c];data=H,W,<enc>payload|>`. For a concept prompt on video the frames come in ascending order. In the live run all 254 frames came as consecutive lines. Ids are stable across frames, but they are not dense and do not start at zero, so key by id. An image = one line `<0f>`. No matches = empty `output_text`, which is not an error.
- Box: pixels of the sent image/clip, **x2/y2 inclusive** (the official parser `meta-sam-parser` does `right = x2 + 1`), `w`/`h` = media size, repeated in every record. If the frame was downscaled before sending, scale the coordinates back.
- Mask: **the raster is local to the box**: `H = y2-y1+1`, `W = x2-x1+1` (checked on all 65 records from 62 image responses and 267 video records), `x=0;y=0` is always literal. It is not a frame mask at reduced resolution. The payload is a custom base85 (alphabet `!`..`~` without `" \ , ; < > |`, a 5-character length prefix) on top of a range coder with a context model. `one_bit` = one bit per pixel; `lossless` = 8-bit coverage, threshold ≥ 129. We do not need to write a decoder: the official `pip install meta-sam-parser` (0.1.0, Meta), `decode_mask_to_raster(SegmentationMask(encoding, payload, width, height))` → bytes H×W 0/1. Pure Python: ~9 µs per pixel for `one_bit` (a 337×276 mask = 0.86 s). Boxes without masks need no decoding.
- Confidence: with `include_confidence` all received values are ≥ 0.507 (60 images, 57 records), on video 0.79–0.96. It looks like the API cuts at 0.5 itself, like the local bench. The docs do not say this; the conclusion comes from the data.
- Errors: 400 on a wrong prompt/metadata (do not retry unchanged), 404 `model_not_found`.
- Price (official pricing-rate-limits page, 27.10): images $2.50 per 1000, video $0.20 per 1000 frames, **minimum 50 frames per video request**. Limit for `sam-3.1` is 3000 RPM per team. Spark 1.3: $1.25 / $4.25 per 1M in/out, cache $0.15.
- Live measurements on 27.10: an image up to 1280 px → 0.5–4.2 s per request, 6 parallel requests without failures. Video 10 s, 1280×720, 25 fps, 254 frames, 2.5 MB: upload 5.4 s, segmentation 12.4 s, response 100 KB (`one_bit`), one person with id 0 in all 254 frames, including shot changes.

## Consequences for the design
1. Video detection is cheaper and gives track ids. For the demo clips, run `input_video` on the whole clip for each object (3–4 requests per clip) and cache the answer by file hash and prompt.
2. For the webcam: one frame per second, `input_image`, the same prompts, cache by frame hash.
3. Compute `wearing` from mask overlap (hard hat ∩ upper third of the person). There is a score (`include_confidence`), but the API already returns only ≥ 0.5. Suppress false positives with `min_duration_s`.
4. The format parser and mask decoder were built on 27.10: `safety_mode/detect/` (decoder from `meta-sam-parser`), tests `tests/test_sam_*.py` on saved responses.

Sources: dev.meta.ai/docs, docs/structured-output, docs/sam/segmenting, docs/sam/reading-segmentation, docs/media-segmentation.
