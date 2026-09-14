"""Extract an external ready-mixed concrete corpus from EPD-Norge digital records.

Source: EPD-Norge soda4LCA node (open, no credentials), class "Bygg / Ferdig
betong" (ready-mixed concrete). Each record is an ILCD process data set. The
fields recovered are the ones the screening model consumes: producer, issue
year, declared unit, cradle-to-gate GWP (A1 to A3), compressive strength class,
and free text.

Strength conversion: EN 206 class C xx/yy gives the characteristic cylinder
strength xx in MPa; the Norwegian class B xx is the cylinder strength in MPa
(B35 corresponds to C35/45). Cylinder MPa x 145.038 = psi, which is the basis of
the U.S. f'c values in the development data.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "external" / "epdnorge_raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "external" / "epdnorge_ready_mixed.csv"
MPA_TO_PSI = 145.038


def sd(o):
    try:
        return o["shortDescription"][0]["value"]
    except Exception:
        return None


def all_text(items):
    return " ".join((x.get("value") or "") for x in (items or []))


def parse_strength_mpa(text: str):
    """Return (cylinder MPa, pattern) or (None, None)."""
    t = text.replace(" ", " ")
    # standard references such as "NBN B15-001" or "NS-EN 206" must not be read as classes
    t = re.sub(r"\b(NBN|NS|DS|SS|EN|ISO)[\s-]*[A-Z]?\s?\d{2,5}(-\d+)?\b", " ", t)
    m = re.search(r"\bC\s?(\d{2,3})\s?/\s?C?(\d{2,3})\b", t)          # C25/30, C30/C37
    if m:
        return float(m.group(1)), "C xx/yy"
    m = re.search(r"\bB\s?(\d{2})(?![\d/-])", t)                        # B35, B35MF45
    if m and 15 <= int(m.group(1)) <= 100:
        return float(m.group(1)), "B xx"
    m = re.search(r"\b(\d{2})\s?MPa\b", t)                              # 16 MPa
    if m:
        return float(m.group(1)), "xx MPa"
    m = re.search(r"\bC(\d{2})(?=[\s\-]|$)", t)                         # C45 (NBN), C25-ECO
    if m and 12 <= int(m.group(1)) <= 100:
        return float(m.group(1)), "C xx"
    return None, None


def gwp_modules(full):
    """GWP by module, preferring GWP-total (EN 15804+A2) then GWP (EN 15804+A1)."""
    best, best_rank = None, 9
    for res in full.get("LCIAResults", {}).get("LCIAResult", []):
        lab = sd(res.get("referenceToLCIAMethodDataSet", {})) or ""
        if re.search(r"global warming potential\s*-\s*total|gwp-?total", lab, re.I):
            rank = 0
        elif re.search(r"^global warming potential\s*\(gwp\)$|^gwp$", lab.strip(), re.I):
            rank = 1
        elif re.search(r"gwp-?ghg", lab, re.I):
            rank = 2
        else:
            continue
        mods = {}
        for a in res.get("other", {}).get("anies", []):
            if a.get("module") and a.get("value") not in (None, ""):
                try:
                    mods[a["module"]] = float(a["value"])
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
        comment = all_text(di.get("generalComment"))
        tech = all_text(pi.get("technology", {}).get("technologyDescriptionAndIncludedProcesses"))
        tech_app = all_text(pi.get("technology", {}).get("technologicalApplicability"))
        text_product = " ".join(x for x in [name, tech, tech_app] if x)
        text = " ".join([name, tech, tech_app, comment])

        ex = full.get("exchanges", {}).get("exchange", [])
        ref = next((e for e in ex if e.get("referenceFlow")), ex[0] if ex else {})
        unit, amount, mass_kg = None, None, None
        for fp in ref.get("flowProperties") or []:
            if fp.get("referenceFlowProperty"):
                unit, amount = fp.get("referenceUnit"), fp.get("meanValue")
            if any((n.get("value") or "").lower() == "mass" for n in fp.get("name", [])):
                mass_kg = fp.get("meanValue")

        g = gwp_modules(full)
        label, mods = g if g else (None, {})
        a1, a2, a3 = mods.get("A1"), mods.get("A2"), mods.get("A3")
        if a1 is not None and a2 is not None and a3 is not None:
            gwp = a1 + a2 + a3
        else:
            gwp = mods.get("A1-A3")

        mpa, pat = parse_strength_mpa(name)
        if mpa is None:
            mpa, pat = parse_strength_mpa(text)

        adm = full.get("administrativeInformation", {})
        owner = sd(adm.get("publicationAndOwnership", {}).get("referenceToOwnershipOfDataSet", {})) or \
            sd(adm.get("dataGenerator", {}).get("referenceToPersonOrEntityGeneratingTheDataSet", {}))
        recs.append(dict(
            uuid=di["UUID"], reg_no=adm.get("publicationAndOwnership", {}).get("registrationNumber"),
            name=name, owner=owner, year=pi["time"].get("referenceYear"),
            valid_until=pi["time"].get("dataSetValidUntil"),
            geo=(pi.get("geography", {}).get("locationOfOperationSupplyOrProduction") or {}).get("location"),
            unit=unit, amount=amount, mass_kg_per_unit=mass_kg,
            gwp_label=label, gwp_a1=a1, gwp_a2=a2, gwp_a3=a3, gwp_a1a3=gwp,
            strength_mpa=mpa, strength_pattern=pat, text=text, text_product=text_product,
            text_comment=comment,
        ))
    df = pd.DataFrame(recs)
    df["strength_psi"] = df.strength_mpa * MPA_TO_PSI
    df["gwp_per_ksi"] = df.gwp_a1a3 / (df.strength_psi / 1000)
    df["usable"] = (df.unit == "m3") & (df.amount == 1) & df.gwp_a1a3.notna() & df.strength_mpa.notna()
    df.to_csv(OUT, index=False)

    print(f"records {len(df)}")
    print("unit:", df.unit.value_counts(dropna=False).to_dict())
    print("GWP label:", df.gwp_label.value_counts(dropna=False).to_dict())
    print("strength pattern:", df.strength_pattern.value_counts(dropna=False).to_dict())
    u = df[df.usable]
    print(f"USABLE {len(u)} records | producers {u.owner.nunique()} | years {u.year.min():.0f}-{u.year.max():.0f} | geo {u.geo.value_counts().to_dict()}")
    print(u[["gwp_a1a3", "strength_mpa", "strength_psi", "gwp_per_ksi"]].describe().round(1).to_string())
    print("share at or above the U.S. threshold (124 kg CO2 eq per ksi):", round(float((u.gwp_per_ksi >= 124).mean()), 3))
    print("external 90th percentile of carbon intensity:", round(float(u.gwp_per_ksi.quantile(0.9)), 1))
    print("\nunusable reasons:")
    x = df[~df.usable]
    print("  no strength:", int(x.strength_mpa.isna().sum()), "| no GWP:", int(x.gwp_a1a3.isna().sum()), "| unit not 1 m3:", int(((x.unit != 'm3') | (x.amount != 1)).sum()))
    print("  sample unusable names:", x.name.head(8).tolist())


if __name__ == "__main__":
    main()
