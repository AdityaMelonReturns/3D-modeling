"""Qwen3-VL wrapper - the one VLM used throughout this pipeline for three
jobs: quality-gate + auto-box (stage [V1]), mask verification (stage [V2]),
and final severity description (stage [J]). Loaded once, reused for all
three - see DESIGN.md for why one model does all three instead of separate
tools.

Runs locally on the GPU pod (not a paid API) - Apache 2.0 licensed, no key
needed. This is the actual fix for what killed the equivalent Claude-vision
stage in both earlier folders: no ANTHROPIC_API_KEY was ever set there, so
that stage silently never ran, the entire project. A local model load can't
have that problem.

Checkpoint: Qwen/Qwen3-VL-4B-Instruct, NOT the 30B-A3B MoE variant originally
planned in DESIGN.md. Found live on the pod (2026-10-04): this pod's
/workspace has a real, hard storage quota around ~40GB total (separate from
the "195TB free" df shows, which is the whole shared cluster, not this pod's
allowance) - confirmed by two 30B-A3B download attempts both failing with
the same `Disk quota exceeded (os error 122)` partway through sixty
gigabytes, even after freeing ~28GB of old conda envs/leftover run folders
first. 4B-Instruct (~8GB on disk) downloaded and loaded in under 10 seconds
on the same pod with the same baseline usage otherwise untouched - see
DESIGN.md's updated risk list.

Uses AutoModelForImageTextToText (not a hardcoded Qwen3VLMoeForConditionalGeneration)
so this keeps working if the checkpoint is changed later to another
Qwen3-VL size - dense and MoE variants need different model classes, the
Auto class dispatches correctly either way.

UNVERIFIED UNTIL THE FIRST REAL RUN: passing a PIL Image object directly in
the chat-template "image" field (rather than a URL/file path) is standard
across the Qwen-VL family's processors, but hasn't been tested against this
exact model/transformers-from-source combo yet. If it errors, the fallback
is saving the array to a temp file and passing that path instead (see
Geometry_Second's depth_local.py for the same class of "save a crop to a
temp file first" pattern already used elsewhere in this project).
"""
import json
import re

import numpy as np
import torch
from PIL import Image


class QwenVLM:
    def __init__(self, checkpoint="Qwen/Qwen3-VL-4B-Instruct", device=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.checkpoint = checkpoint

    def _ask(self, image, prompt, max_new_tokens=300):
        """image: PIL Image, numpy RGB array, or file path/URL string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text

    @staticmethod
    def _extract_json(text):
        """Qwen sometimes wraps JSON in ```json fences or adds a sentence of
        chatter around it - pull out the first {...} block rather than
        assuming the whole response is clean JSON."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def quality_gate_and_box(self, image_rgb, defect_type_hint=None):
        """Stage [V1]: replaces the hand-drawn box prompts from earlier
        folders. Returns dict(usable, box, defect_type_guess, reason).
        Fails CLOSED (usable=False) if Qwen's output can't be parsed as
        JSON - never silently proceeds with a made-up box."""
        # Qwen3-VL's grounding is trained on relative coordinates in 0-1000,
        # not raw pixels - ask in its native format and convert ourselves,
        # rather than hope it does pixel math for an image it saw resized.
        hint = f' The defect is expected to look like: "{defect_type_hint}".' if defect_type_hint else ""
        prompt = (
            "You are inspecting a house-defect photo (a hole, crack, peeling paint, "
            "rot, or similar damage)." + hint + " Respond with JSON only, no other "
            'text: {"usable": true/false, "box_2d": [x0,y0,x1,y1] tightly around '
            "the visible defect, in relative coordinates from 0 to 1000, "
            '"defect_type_guess": "hole|crack|peeling_paint|rot|inclination|other", '
            '"reason": "one short sentence"}. Set usable=false if the photo is '
            "too blurry, too dark, or the defect isn't actually visible in it."
        )
        text = self._ask(image_rgb, prompt)
        parsed = self._extract_json(text)
        rel = (parsed or {}).get("box_2d") or (parsed or {}).get("box")
        if parsed is None or not isinstance(rel, (list, tuple)) or len(rel) != 4:
            return {"usable": False, "box": None, "defect_type_guess": None,
                    "reason": f"could not parse Qwen output: {text[:200]!r}",
                    "raw_response": text}

        h, w = image_rgb.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in rel)
        box = [int(round(x0 / 1000 * w)), int(round(y0 / 1000 * h)),
               int(round(x1 / 1000 * w)), int(round(y1 / 1000 * h))]
        box = [max(0, min(box[0], w - 1)), max(0, min(box[1], h - 1)),
               max(0, min(box[2], w - 1)), max(0, min(box[3], h - 1))]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return {"usable": False, "box": None, "box_rel_1000": list(rel),
                    "defect_type_guess": parsed.get("defect_type_guess"),
                    "reason": f"degenerate box from Qwen: {rel}", "raw_response": text}

        parsed["box_rel_1000"] = list(rel)
        parsed["box"] = box
        parsed.setdefault("usable", False)
        parsed["raw_response"] = text
        return parsed

    def verify_mask(self, overlay_rgb, defect_type_hint=None):
        """Stage [V2]: overlay_rgb is the photo with SAM2's mask painted on
        top in red (same image render_masks.py produced by hand in earlier
        folders). Returns dict(matches, confidence, reason). This does NOT
        block the pipeline on a "no" - it's logged as a confidence flag on
        that shot, same philosophy as the DA3-vs-Depth-Pro disagreement flag
        in Geometry_third (surface the disagreement, don't hide it, don't
        halt over it either)."""
        hint = f' The defect being measured is: "{defect_type_hint}".' if defect_type_hint else ""
        prompt = (
            "This photo has a red overlay marking a region a segmentation model "
            "selected as a defect." + hint + " Does the red region tightly and "
            "correctly match the actual visible defect (not the background, not a "
            'different object)? Respond with JSON only: {"matches": true/false, '
            '"confidence": 0-1, "reason": "one short sentence"}.'
        )
        text = self._ask(overlay_rgb, prompt)
        parsed = self._extract_json(text)
        if parsed is None:
            return {"matches": False, "confidence": 0.0,
                    "reason": f"could not parse Qwen output: {text[:200]!r}",
                    "raw_response": text}
        parsed["raw_response"] = text
        return parsed

    def describe_severity(self, image_rgb, measurements, defect_id):
        """Stage [J]: same job Claude's claude_vision_interpret did in
        earlier folders' interpret.py, which never actually ran there (no
        API key was ever set the whole project). This one runs, since it's
        the same local model already loaded for stages V1/V2."""
        prompt = (
            f'This is a photo of a house defect ("{defect_id}"). Measured geometry '
            f"from a 3D reconstruction:\n{json.dumps(measurements, indent=2, default=str)}\n\n"
            "Given the photo and these measurements, respond with JSON only: "
            '{"defect_type": "...", "severity": "low|moderate|severe", '
            '"description": "one plain-English sentence", '
            '"measurement_plausible": true/false, '
            '"notes": "anything the numbers get wrong vs what you see, if anything"}'
        )
        text = self._ask(image_rgb, prompt, max_new_tokens=400)
        parsed = self._extract_json(text)
        if parsed is None:
            return {"raw_response": text, "parse_error": True}
        return parsed
