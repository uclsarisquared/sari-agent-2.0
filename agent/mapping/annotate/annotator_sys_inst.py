"""Compose annotator prompts and JSON schemas; text lives in agent/prompts/annotator.

Stage 1 classifies a perpendicular image as shelf/non-shelf. Stage 2 combines shared
rules with an overlay selected by effective_kind(), rather than raw topology kind.
Callers label images STANDING/CROUCHED; shelf items are deduplicated across views.

Product labels supply identity: fold brand into name and omit unreadable labels.
Generic descriptions are not searchable products. Category signs may belong to
distant aisles, so sign_text is observational only. The graph supplies spatial
relationships; neighboring shelf context is not injected into annotation prompts.

Closer views and focused label questions improve recall without inventing names.
Qwen thinking must be disabled to avoid spending the output budget on reasoning;
dense repeated facings can still cause decoding loops (see annotate_qwen.py).
Schema enforcement uses response_format/json_schema; the tested vLLM backend
silently ignored guided_json. On-arrival verification checks index claims.
"""

from agent_core.prompt_loader import load_prompt


# Category enum

SHELF_CATEGORIES = [
    "Water", "Soda", "Juice", "Dairies", "Liquor",
    "Biscuit", "Can", "Chips", "Nuts", "Soup", "Noodles",
]
"""The store's own product taxonomy, BAKED from
SariSandboxV2/Assets/Resources/Data/Categories.json as of 2026-07.

Baked on purpose: reading that file at design time to fix a static list is NOT a runtime Unity
dependency - the live pipeline never queries Unity for categories, it just hands the VLM this
frozen list to choose from. That keeps the runtime observation-only while still using the real
taxonomy. Regenerate this list if the store's catalog changes."""

CATEGORY_OTHER = "other"
CATEGORY_ENUM = SHELF_CATEGORIES + [CATEGORY_OTHER]


def _category_lines():
    """The enum exactly as the prompt shows it: the store's categories plus the `other` escape,
    glossed inline.

    `other` is rendered as a member of the list rather than explained in prose underneath it so
    that the printed list IS the schema's enum. Otherwise the prompt tells the model to pick one
    value "from this list" and then hands it a twelfth value the list never mentioned."""
    lines = [f"    - {c}" for c in SHELF_CATEGORIES]
    lines.append(
        f"    - {CATEGORY_OTHER}  (the products fit none of the above, or you cannot tell what they are)"
    )
    return "\n".join(lines)


# Stage 1 - classify

# Count nearby shelves anywhere in frame, including aisle ends and gaps.
# Center-only classification rejected readable off-center shelves; ignore only
# small, distant edge shelves.
SYS_INST_CLASSIFY = load_prompt("annotator/classify")


# Stage 2 - shared base

SYS_INST_ANNOTATE_BASE = load_prompt("annotator/base")


# Stage 2 - shelf overlay

SYS_INST_ANNOTATE_SHELF = load_prompt("annotator/shelf")


# The non-shelf prompt is used verbatim. The shelf prompt formats categories,
# so its literal JSON braces must be doubled.

SYS_INST_ANNOTATE_NON_SHELF = load_prompt("annotator/non_shelf")


# JSON schemas - pass to the server if it supports guided/structured output

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["shelf", "non_shelf"]}},
    "required": ["label"],
    "additionalProperties": False,
}

SHELF_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_summary": {"type": "string"},
        # 1-2 values, dominant first. Real shelves are mixed more often than not (a measured
        # example: one shelf interleaving Tostitos/Pringles/Clover with Pancit Canton/Jin Ramen
        # row by row), so forcing a single label was lossy on a coin-flip basis.
        "shelf_type": {
            "type": "array",
            "items": {"type": "string", "enum": CATEGORY_ENUM},
            "minItems": 1,
            "maxItems": 2,
        },
        "sign_text": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Brand is folded INTO name, deliberately - see the measured note above.
                    "name": {"type": "string"},
                    "variant": {"type": ["string", "null"]},
                    # String, not number: the tag may read "29.95", "29", or partially. Kept
                    # nullable and NOT required - a mis-paired price is worse than a missing one,
                    # and the catalog's pricePHP gives a ground truth to score these against.
                    "price": {"type": ["string", "null"]},
                    "appearance": {"type": ["string", "null"]},
                    # Required and non-nullable: it is the flat index's query key, and with a
                    # 1-2 value shelf_type there is no single value left to default it to.
                    # Asking per item is also what makes a mixed shelf's rows correct.
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                },
                "required": ["name", "category"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["semantic_summary", "shelf_type", "items"],
    "additionalProperties": False,
}

NON_SHELF_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_summary": {"type": "string"},
        "sign_text": {"type": ["string", "null"]},
    },
    "required": ["semantic_summary"],
    "additionalProperties": False,
}


# Composition

SHELF_KIND = "shelf"
NON_SHELF_KIND = "non_shelf"


def effective_kind(topology_kind, classifier_label=None):
    """The kind that decides which Stage-2 overlay a checkpoint gets. Start here.

    Two signals disagree by design, so neither one alone is the answer:
      * `topology_kind` is GEOMETRIC. shelf_coverage.py calls a checkpoint "shelf" because it
        placed it perpendicular to a rectangular bulge in the occupancy grid - and a bulge can
        turn out to be a bare wall.
      * `classifier_label` is VISUAL. Stage 1 looked at the primary image and reported what is
        actually there ("shelf" / "non_shelf").

    A checkpoint is annotated AS a shelf only where the two agree. A "shelf" the classifier
    rejected keeps its place and its connectivity in the graph; it just gets annotated for what it
    really is. Resolving that disagreement is the entire reason Stage 1 exists, which is why this
    is a function and not a note asking the caller to remember.

    Pass classifier_label=None to skip Stage 1 and trust the geometry. That is the right call for
    structural checkpoints (junction/end/doorway): they have no perpendicular surface, so there is
    nothing for a classifier to rule on.
    """
    if topology_kind != SHELF_KIND:
        return topology_kind
    if classifier_label is None or classifier_label == SHELF_KIND:
        return SHELF_KIND
    return NON_SHELF_KIND


def build_annotation_instructions(kind):
    """Compose the Stage-2 system instruction for a checkpoint of this `kind`.

    `kind` must be a value from effective_kind(), NOT a raw topology kind. Only "shelf" gets the
    shelf overlay; everything else ("junction", "end", "doorway", "non_shelf") gets the non-shelf
    one. Pass a raw topology kind and a bare wall Stage 1 already rejected receives a prompt that
    says "list the products on this shelf" - the exact hallucination the classifier is there to
    prevent.
    """
    if kind == SHELF_KIND:
        overlay = SYS_INST_ANNOTATE_SHELF.format(categories=_category_lines())
    else:
        overlay = SYS_INST_ANNOTATE_NON_SHELF
    return f"{SYS_INST_ANNOTATE_BASE}\n{overlay}"


def schema_for(kind):
    """The JSON schema matching build_annotation_instructions(kind)'s contract. `kind` comes from
    effective_kind(), same as there."""
    return SHELF_ANNOTATION_SCHEMA if kind == SHELF_KIND else NON_SHELF_ANNOTATION_SCHEMA


if __name__ == "__main__":
    # Eyeball the composed prompts: python mapping/annotate/annotator_sys_inst.py
    print("effective_kind(topology_kind, classifier_label):")
    for _topo, _label in (("shelf", "shelf"), ("shelf", "non_shelf"), ("shelf", None), ("junction", None)):
        _lbl = repr(_label)
        print(f"    topology={_topo!r:10s} classifier={_lbl:12s} -> {effective_kind(_topo, _label)!r}")
    print()

    for title, text in (
        ("STAGE 1 - CLASSIFY", SYS_INST_CLASSIFY),
        ("STAGE 2 - SHELF          (topology=shelf, classifier=shelf)",
         build_annotation_instructions(effective_kind("shelf", "shelf"))),
        ("STAGE 2 - NON-SHELF      (topology=shelf, classifier=non_shelf -> a bare wall)",
         build_annotation_instructions(effective_kind("shelf", "non_shelf"))),
    ):
        print("=" * 90)
        print(title)
        print("=" * 90)
        print(text)
