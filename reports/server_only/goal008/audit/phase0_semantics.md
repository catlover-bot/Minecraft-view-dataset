# Goal008 Phase 0: representations and generation provenance

No proxy scores, correlations, provider requests or key lookups were performed by this audit.

**O.** The complete saved v4 JSON visual-description record itself, including human-readable fields and original metadata; not free-form prose.

**J.** Exact typed-JSON-tree-preserving reserialization of O, with sorted keys and compact formatting; not historical Structured IR.

**P.** Partial ground-truth oracle: occupied XYZ dimensions, world bbox, exact material set and voxel count; per-voxel geometry/material assignment omitted.

**E.** Complete discrete ground-truth cuboid specification with palette and coordinate semantics; exact decode verified for60 scenes. Valid-only outcomes condition on successful model output and are not a whole-population upper bound.

P/E also change available GT information and Builder decoding/execution demands. Their differences do not identify pure language-information loss.

The three description providers produced saved image-to-text records. The two Builders independently generated action programs from each record. These are separate model roles.

## Human-readable content inside JSON

The record contains prose in summary, shape descriptions, material roles, reconstruction hints and uncertainties, alongside typed dimensions and lists of elements/materials. Original provider/model/building/time metadata is also retained in O and J. Full records remain at their canonical source paths.

```json
{
  "openai": {
    "summary": 60,
    "building_type": 60,
    "shape": 60,
    "dimensions_estimate": 60,
    "materials": 60,
    "elements": 60,
    "rebuild_hints": 60,
    "uncertainties": 60
  },
  "anthropic": {
    "summary": 60,
    "building_type": 60,
    "shape": 60,
    "dimensions_estimate": 60,
    "materials": 60,
    "elements": 60,
    "rebuild_hints": 60,
    "uncertainties": 60
  },
  "gemini": {
    "summary": 60,
    "building_type": 60,
    "shape": 60,
    "dimensions_estimate": 60,
    "materials": 60,
    "elements": 60,
    "rebuild_hints": 60,
    "uncertainties": 60
  }
}
```

## Source-bound examples

openai / buildings_100_v4_building_087: `/root/workspace/Minecraft-view-dataset/outputs/i2t2b/buildings_100_v4/building_087/description_openai_gpt_5_mini/description.json`; SHA256 `42549dd95ce4e4f5fb4a058c5353ae8d88272cf793dd543c37c2cc30b0898550`. Partial excerpt only:

```json
{
  "summary": "A single-story, hollow rectangular stone structure with an interior courtyard/open center (U-shaped roofline visible). Thick stone walls are patterned with alternating bands of smooth stonebrick and cobblestone; the roof is stepped with light-colored slabs and a dark stripe near the top. A small cen",
  "dimensions_estimate": {
    "width": 18,
    "depth": 14,
    "height": 8
  },
  "materials_first_two": [
    {
      "name": "stonebrick",
      "role": "primary wall blocks and roof slab field",
      "confidence": 0.95
    },
    {
      "name": "cobblestone",
      "role": "decorative bands, buttresses and perimeter low blocks",
      "confidence": 0.95
    }
  ],
  "elements_first_two": [
    "windows",
    "entrance"
  ]
}
```

anthropic / buildings_100_v4_building_087: `/root/workspace/Minecraft-view-dataset/outputs/i2t2b/buildings_100_v4/building_087/description_anthropic_claude_haiku_4_5_20251001/description.json`; SHA256 `208d9c0cd78ddd873b7c915cbadf28b3ee251fa18fede86ec24aef26918d6173`. Partial excerpt only:

```json
{
  "summary": "A rectangular fortification structure with thick stonebrick walls, a flat roof topped with dark wood/nether brick layers, and a central courtyard. The building features buttress-like corner pillars and a single entrance. The structure appears to be a defensive bunker or small fort with an open inter",
  "dimensions_estimate": {
    "width": 16,
    "depth": 20,
    "height": 6
  },
  "materials_first_two": [
    {
      "name": "stonebrick",
      "role": "primary wall construction",
      "confidence": 0.95
    },
    {
      "name": "nether_brick",
      "role": "roof capping layers",
      "confidence": 0.85
    }
  ],
  "elements_first_two": [
    "thick perimeter walls",
    "corner buttress pillars"
  ]
}
```

gemini / buildings_100_v4_building_087: `/root/workspace/Minecraft-view-dataset/datasets/buildings_100_v4/building_087/description_gemini_gemini_3_1_pro_preview/description.json`; SHA256 `647615dcf49b44c3606bec57fdb70827aacdc3c5065e7b0ec8cfa732b9933f24`. Partial excerpt only:

```json
{
  "summary": "A massive, U-shaped stone structure resembling a fortress or temple. It features thick walls with horizontal banding of different stone types, vertical cobblestone buttresses, and a tiered roof with dark accents. An inner courtyard is enclosed by the U-shape, and a low, broken cobblestone wall surro",
  "dimensions_estimate": {
    "width": 25,
    "depth": 21,
    "height": 13
  },
  "materials_first_two": [
    {
      "name": "stone",
      "role": "primary wall material",
      "confidence": 0.9
    },
    {
      "name": "stonebrick",
      "role": "wall banding",
      "confidence": 0.9
    }
  ],
  "elements_first_two": [
    "courtyard",
    "tiered roof"
  ]
}
```

## Actual O/J call and sampling audit

O and J are separate single-output generation calls for the same typed JSON information; one additional O attempt is an approved authentication retry. No payload seed, temperature, top-p or penalties are explicitly set. Provider-reported values describe returned metadata, not an experimental guarantee of deterministic decoding. Anthropic unreported defaults remain unknown.

O has 361 attempts across360 slots, including the original401 and approved successful retry at global940; J has 360 attempts across360 slots. Global942 remains the separate unknown-usage HTTP520 observation. There is no controlled repeated-sampling design.

The seed20261013 used for cohort selection and statistical resampling is not a decoding seed. O/J contrasts cannot by themselves isolate a formatting-only causal effect from provider decoding non-determinism and single-output variability.

Request fields and safe response-metadata distributions are recorded in `sampling_audit.json`. Unreported provider parameters remain unknown; omission never means temperature0 or a fixed seed.

## Frozen code evidence

- `offline_evaluation.server_v2.cohort_v4.canonicalize_saved_text`: [cohort_v4.py](/root/workspace/Minecraft-view-dataset/offline_evaluation/server_v2/cohort_v4.py:127), SHA256 `45f9ed256938f0e9df8ffe0cb88ae81fdcfedf80354da91c6cd4eb0a1d930e88`.
- `offline_evaluation.server_v2.cohort_v4.typed_tree`: [cohort_v4.py](/root/workspace/Minecraft-view-dataset/offline_evaluation/server_v2/cohort_v4.py:83), SHA256 `45f9ed256938f0e9df8ffe0cb88ae81fdcfedf80354da91c6cd4eb0a1d930e88`.
- `offline_evaluation.server_v2.oracle.partial_oracle`: [oracle.py](/root/workspace/Minecraft-view-dataset/offline_evaluation/server_v2/oracle.py:110), SHA256 `459b7448b38a990bd23595d2b372e2c29593328aa82ced960de439ab47fd690c`.
- `offline_evaluation.resmoke_v2.cuboids.exact_builder_text`: [cuboids.py](/root/workspace/Minecraft-view-dataset/offline_evaluation/resmoke_v2/cuboids.py:138), SHA256 `ded178ab7c4410b0e0a795963f97ef034973c5e2460f69fccebdc8867bba389c`.
- `offline_evaluation.resmoke_v2.cuboids.decode_exact`: [cuboids.py](/root/workspace/Minecraft-view-dataset/offline_evaluation/resmoke_v2/cuboids.py:81), SHA256 `ded178ab7c4410b0e0a795963f97ef034973c5e2460f69fccebdc8867bba389c`.
- `offline_evaluation.resmoke_v2.adapter.build_payload`: [adapter.py](/root/workspace/Minecraft-view-dataset/offline_evaluation/resmoke_v2/adapter.py:27), SHA256 `b0845f0e66a297779fa8456ea71e0780a6760a78be9d0587afec53bdcd3ce646`.

The source gate preserves Goal007’s20 confirmatory comparisons and its RQ2 status NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES. A later CPU attribute proxy is new exploratory evidence, not a recovered historical quality score.
