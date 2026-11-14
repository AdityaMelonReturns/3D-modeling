"""Standard sizes of building parts used as a stand-in for the ruler
(Task A of the Reference-Object Scale brief).

Every size records where it came from. "kind" decides how it is measured:
  repeating  spacing between many repeats (shingle courses, board gaps, brick
             courses) - averaged over all repeats, so one bad edge matters less
  single     the width of one object between its two edges (a 2x6 rail cap,
             an outlet plate) - scale along that one direction only

tolerance_pct is how much the real object can differ from the standard size;
it is added into each number's uncertainty_pct.
"""

IN = 2.54

REFERENCE_SIZES = {
    "shingle_3tab": dict(
        kind="repeating", size_cm=5 * IN, what="course spacing (exposure) of 3-tab asphalt shingles, 12 x 36 in",
        tolerance_pct=3.0,
        source="GAF Royal Sovereign specifications: 12 x 36 in, 5 in (127 mm) exposure - "
               "https://www.gaf.com/en-us/products/royal-sovereign/specifications",
        note="West-coast metric 3-tab is 13-1/4 x 39-3/8 in with 5-5/8 in exposure (Atlas) - ask which is on the roof"),
    "shingle_architectural": dict(
        kind="repeating", size_cm=5.625 * IN, what="course spacing of metric-size / architectural shingles",
        tolerance_pct=3.0,
        source="Atlas Roofing (via Roofing Contractor): metric 13-1/4 x 39-3/8 in, 5-5/8 in exposure - "
               "https://roofingcontractor.com/articles/89369-value-shingles"),
    "brick_modular_3course": dict(
        kind="repeating", size_cm=8 * IN / 3, what="one course of US modular brick incl. 3/8 in joint (3 courses = 8 in)",
        tolerance_pct=2.0,
        source="Brick Industry Association Technical Note 10, Dimensioning and Estimating Brick Masonry - "
               "https://gobrick.com/media/file/10-dimensioning-and-estimating-brick-masonry.pdf",
        note="assumes 3/8 in mortar joints; 8 in per 3 courses is stated in secondary sources, check TN 10 Table 1"),
    "cmu_course": dict(
        kind="repeating", size_cm=8 * IN, what="one course of 8x8x16 concrete block (7-5/8 in face + 3/8 in joint)",
        tolerance_pct=2.0,
        source="NCMA TEK 2-1, Typical Sizes and Shapes of Concrete Masonry Units: actual 7-5/8 x 7-5/8 x 15-5/8 in, "
               "+-1/8 in (ASTM C90); summarised in Angelus CMU Basics - "
               "https://www.angelusblock.com/assets/docs/Angelus_CMU_Basics_&_Glossary.pdf"),
    "deck_board": dict(
        kind="repeating", size_cm=(5.5 + 0.1875) * IN, what="5/4x6 or 2x6 deck board: 5-1/2 in face + ~3/16 in gap",
        tolerance_pct=3.0,
        source="American Softwood Lumber Standard PS 20 via SFPA Standard Sizing (nominal 6 in -> 5-1/2 in dressed; "
               "5/4x6 decking -> 1 x 5-1/2 in) - "
               "https://www.southernpine.com/resources/specifying-southern-pine-lumber/standard-sizing/",
        note="the 3/16 in gap is a typical install gap, NOT from a spec (1/8-1/4 in seen); it is part of the tolerance"),
    "lumber_2x4_face": dict(
        kind="single", size_cm=3.5 * IN, what="2x4 face width", tolerance_pct=2.0,
        source="PS 20 via SFPA Standard Sizing: nominal 4 in -> 3-1/2 in dressed dry - "
               "https://www.southernpine.com/resources/specifying-southern-pine-lumber/standard-sizing/",
        note="green / pressure-treated stock can be slightly wider"),
    "lumber_2x6_face": dict(
        kind="single", size_cm=5.5 * IN, what="2x6 face width (e.g. a deck rail cap)", tolerance_pct=2.0,
        source="PS 20 via SFPA Standard Sizing: nominal 6 in -> 5-1/2 in dressed dry - "
               "https://www.southernpine.com/resources/specifying-southern-pine-lumber/standard-sizing/",
        note="green / pressure-treated stock can be slightly wider"),
    "outlet_plate_standard": dict(
        kind="single", size_cm=2.75 * IN, what="standard 1-gang wall plate width (2.75 x 4.5 in)", tolerance_pct=1.0,
        source="Leviton Wallplate Size Guide Q-1289 - https://frpublish.leviton.com/content/dam/leviton/"
               "commercial-industrial/product_documents/solution_sheets/Wallplate%20Size%20Guide%20Q-1289.pdf",
        note="midway plates are 3.13 x 4.88 in - check the shape before using"),
    "ceiling_tile_2x2": dict(
        kind="repeating", size_cm=24 * IN, what="2x2 ft suspended ceiling grid module (tile ~23.7 in + grid)",
        tolerance_pct=1.0,
        source="Armstrong Brighton 2x2 listing: actual 23.719 x 23.719 in (retailer listing, not a spec sheet) - "
               "https://superarbor.io/collections/tile/products/3338832-armstrong-ceilings-24-in-x-24-in-brighton-"
               "16-pack-white-textured-15-16-in-drop-acoustic-panel-ceiling-tiles",
        note="repeat spacing on a grid is the full 24 in module; tile alone is 23.7 in"),
    "lap_siding": dict(
        kind="repeating", size_cm=None, what="lap siding exposure", tolerance_pct=3.0,
        source="varies 4-8 in by product - only usable when the job records the siding exposure "
               "(pass it as size_cm in the prompt)"),
}
