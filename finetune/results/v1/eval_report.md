# ListSmart writer: base vs fine-tuned

- Base model: `Qwen/Qwen3-8B`
- Fine-tuned: `tinker://a44e4907-15df-5225-9bbc-42a02b6f7c03:train:0/sampler_weights/listsmart-writer`
- Test examples: 120 (held out, never seen in training)
- Generated: 2026-10-18T08:06:05+00:00

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| format_valid | 0.992 | 0.842 | -0.150 |
| constraint_compliance | 0.858 | 0.742 | -0.117 |
| title_length_ok | 0.992 | 0.825 | -0.167 |
| bullets_exactly_5 | 0.992 | 0.842 | -0.150 |
| tags_ok | 0.883 | 0.775 | -0.108 |
| description_paragraphs_ok | 0.967 | 0.833 | -0.133 |
| search_query_ok | 0.992 | 0.833 | -0.158 |
| attribute_coverage | 0.897 | 0.595 | -0.301 |
| rouge_l | 0.208 | 0.268 | +0.061 |

**Gemma judge (gemma-4-26b-a4b-it, blind pairwise, random A/B order, n=120):** fine-tuned wins 1%, base wins 78%, ties 22%.
(20 pairs were decided automatically because one or both outputs were not valid JSON.)

**Speed** (Tinker (hosted GPUs; latency includes network + queue)): base median 3.579 s per listing, 75.14 tokens/s; fine-tuned median 7.38 s, 45.56 tokens/s.
**Cost per 1000 listings:** local ₹0, Tinker hosted ₹34.95, closed API n/a.

Metric notes: format_valid = parses into a Listing with all fields; constraint_compliance = passes every
platform rule (title length, exactly 5 bullets, tag count/length, 2-4 paragraphs, 3-6 word query);
attribute_coverage = share of attribute values (type, colours, material, pattern, style) mentioned in
title+description; rouge_l = ROUGE-L F1 vs the real reference listing.
