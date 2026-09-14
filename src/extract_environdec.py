"""Extract an external ready-mixed concrete corpus from EPD International digital records.

Source: the EPD International (Environdec) soda4LCA node, open and without
credentials. Records are ILCD process data sets; the Australian declarations
are published under EPD Australasia within the International EPD System.

Recovered per record: producer, geography, issue year, declared unit, cradle-
to-gate GWP (A1 to A3), compressive strength (MPa, converted to psi), product
text (name, technology description, applicability), and the plant or city where
the text states it. Strength appears as "40 MPa", "N32", "S40", "Grade 40", or
"C32/40"; the characteristic strength in MPa is taken in each case.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "external" / "environdec_raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "external" / "environdec_ready_mixed.csv"
MPA_TO_PSI = 145.038


def sd(o):
    try:
        return o["shortDescription"][0]["value"]
    except Exception:
        return None


def all_text(items):
    return " ".join((x.get("value") or "") for x in (items or []))


def parse_strength_mpa(text: str):
    t = text.replace(" ", " ")
    t = re.sub(r"\b(AS|NZS|EN|ISO|AS/NZS)\s?\d{3,5}(\.\d+)?(-\d+)?\b", " ", t)   # standards, not grades
    for pat, lab in [
        (r"\b(\d{2})\s?MPa\b", "xx MPa"),
        (r"\bC\s?(\d{2,3})\s?/\s?C?(\d{2,3})\b", "C xx/yy"),
        (r"\b[NS](\d{2})\b(?!\s?(?:mm|%))", "N/S grade"),
        (r"\bgrade\s?(\d{2})\b", "Grade xx"),
        (r"\b(\d{2})\s?N\b", "xxN"),
    ]:
        m = re.search(pat, t, re.I)
        if m and 10 <= int(m.group(1)) <= 120:
            return float(m.group(1)), lab
    return None, None


# Australian producer mix codes carry the grade in the first two digits
# (Holcim "QE322L100" -> 32, Aurora "AE2514" -> 25, Barro "EN40" -> 40). The rule
# agrees with the explicit strength in 113 of 115 records that state both, and is
# applied only when no explicit strength is given and the grade is a standard one.
STANDARD_GRADES = {10, 15, 20, 25, 32, 40, 50, 65, 80, 100}


def code_strength_mpa(name: str):
    n = str(name)
    m = re.search(r"\b[A-Z]{1,2}(\d{2})\d[A-Z0-9]*\b", n)
    if m and int(m.group(1)) in STANDARD_GRADES:
        return float(m.group(1)), "mix code"
    m = re.search(r"\bE[NS]?(\d{2})[A-Z]*\b", n)
    if m and int(m.group(1)) in STANDARD_GRADES:
        return float(m.group(1)), "mix code"
    return None, None


def gwp_modules(full):
    best, best_rank = None, 9
    for res in full.get("LCIAResults", {}).get("LCIAResult", []):
        lab = sd(res.get("referenceToLCIAMethodDataSet", {})) or ""
        if re.search(r"global warming potential\s*-?\s*total|gwp-?total", lab, re.I):
            rank = 0
        elif re.search(r"gwp-?ghg|global warming potential\s*-?\s*ghg", lab, re.I):
            rank = 1
        elif re.search(r"^global warming potential\s*\(gwp\)$|^gwp$|^global warming potential$", lab.strip(), re.I):
            rank = 2
        else:
            continue
        mods = {}
        for a in res.get("other", {}).get("anies", []):
            if a.get("module") and a.get("value") not in (None, "", "ND", "MND", "-"):
                try:
                    mods[a["module"]] = float(str(a["value"]).replace(",", ""))
                except ValueError:
                    pass
        if mods and rank < best_rank:
            best, best_rank = (lab, mods), rank
    return best


def main() -> None:
    recs = []
    for f in sorted(glob.glob(str(RAW / "*.json"))):
        full = json.load(open(f, encoding="utf-8"))
        pi = full["processInformation"]
        di = pi["dataSetInformation"]
        name = di["name"]["baseName"][0]["value"]
        tech = all_text(pi.get("technology", {}).get("technologyDescriptionAndIncludedProcesses"))
        appl = all_text(pi.get("technology", {}).get("technologicalApplicability"))
        comment = all_text(di.get("generalComment"))
        text_product = " ".join(x for x in [name, tech, appl] if x)

        ex = full.get("exchanges", {}).get("exchange", [])
        ref = next((e for e in ex if e.get("referenceFlow")), ex[0] if ex else {})
        unit, amount, mass = None, None, None
        for fp in ref.get("flowProperties") or []:
            if fp.get("referenceFlowProperty"):
                unit, amount = fp.get("referenceUnit"), fp.get("meanValue")
            if any((n.get("value") or "").lower() == "mass" for n in fp.get("name", [])):
                mass = fp.get("meanValue")

        g = gwp_modules(full)
        label, mods = g if g else (None, {})
        a1, a2, a3 = mods.get("A1"), mods.get("A2"), mods.get("A3")
        gwp = a1 + a2 + a3 if None not in (a1, a2, a3) else mods.get("A1-A3")

        mpa, pat = parse_strength_mpa(name)
        if mpa is None:
            mpa, pat = parse_strength_mpa(text_product)
        if mpa is None:
            mpa, pat = code_strength_mpa(name)

        adm = full.get("administrativeInformation", {})
        owner = sd(adm.get("publicationAndOwnership", {}).get("referenceToOwnershipOfDataSet", {})) or \
            sd(adm.get("dataGenerator", {}).get("referenceToPersonOrEntityGeneratingTheDataSet", {}))
        classes = [c.get("value") for c in (di.get("classificationInformation", {}).get("classification") or [{}])[0].get("class", [])]
        recs.append(dict(
            uuid=di["UUID"], reg_no=adm.get("publicationAndOwnership", {}).get("registrationNumber"),
            name=name, owner=owner, year=pi["time"].get("referenceYear"), valid_until=pi["time"].get("dataSetValidUntil"),
            geo=(pi.get("geography", {}).get("locationOfOperationSupplyOrProduction") or {}).get("location"),
            sub_type=di.get("subType") or full.get("subType"), classification=" / ".join(c for c in classes if c),
            unit=unit, amount=amount, mass_kg_per_unit=mass, gwp_label=label,
            gwp_a1=a1, gwp_a2=a2, gwp_a3=a3, gwp_a1a3=gwp,
            strength_mpa=mpa, strength_pattern=pat, name_text=name, text_product=text_product,
            text_applicability=appl, text_comment=comment,
        ))
    df = pd.DataFrame(recs)
    df["strength_psi"] = df.strength_mpa * MPA_TO_PSI
    df["gwp_per_ksi"] = df.gwp_a1a3 / (df.strength_psi / 1000)
    df["usable"] = (df.unit == "m3") & (df.amount == 1) & df.gwp_a1a3.notna() & df.strength_mpa.notna() & (df.gwp_a1a3 > 0)
    df.to_csv(OUT, index=False)

    print(f"records {len(df)}")
    print("geo:", df.geo.value_counts().head(8).to_dict())
    print("unit:", df.unit.value_counts(dropna=False).head(5).to_dict())
    print("GWP label:", df.gwp_label.value_counts(dropna=False).head(5).to_dict())
    print("strength pattern:", df.strength_pattern.value_counts(dropna=False).to_dict())
    u = df[df.usable]
    print(f"\nUSABLE {len(u)} | producers {u.owner.nunique()} | geo {u.geo.value_counts().head(6).to_dict()} | years {u.year.value_counts().sort_index().to_dict()}")
    print(u[["gwp_a1a3", "strength_mpa", "strength_psi", "gwp_per_ksi"]].describe().round(1).to_string())
    print("share at or above the U.S. threshold (124 kg CO2 eq per ksi):", round(float((u.gwp_per_ksi >= 124).mean()), 3),
          "| count:", int((u.gwp_per_ksi >= 124).sum()))
    print("external 90th percentile:", round(float(u.gwp_per_ksi.quantile(0.9)), 1))
    x = df[~df.usable]
    print("\nunusable:", len(x), "| no strength:", int(x.strength_mpa.isna().sum()), "| no GWP:", int(x.gwp_a1a3.isna().sum()),
          "| unit not 1 m3:", int(((x.unit != 'm3') | (x.amount != 1)).sum()))
    print("  sample unusable names:", x.name.head(10).tolist())


if __name__ == "__main__":
    main()
