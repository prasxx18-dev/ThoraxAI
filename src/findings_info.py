"""Short plain-language explanations for the 14 ChestX-ray14 findings.

Single source of truth for finding text shown in the app. Each explanation is one
or two sentences and is limited to what the cited sources say (read 2026-10-17;
listed in docs/medical_references.md). General educational content - it does not
describe any particular image and contains no advice.

Keep this module free of Streamlit imports so it can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import config

CHEST_XRAY_SOURCE = ("MedlinePlus: Chest x-ray", "https://medlineplus.gov/ency/article/003804.htm")

RESULTS_NOTICE = ("AI-generated results are intended to support understanding of chest X-ray findings. "
                  "Please consult a qualified healthcare professional for medical interpretation.")


@dataclass(frozen=True)
class FindingInfo:
    label: str  # model label (config.LABELS)
    name: str  # display name
    meaning: str  # one or two sentences, plain English
    sources: tuple[tuple[str, str], ...]


_R = "https://radiopaedia.org/articles/"

FINDINGS: dict[str, FindingInfo] = {
    "Atelectasis": FindingInfo(
        "Atelectasis", "Atelectasis",
        "Part of a lung has collapsed or is not filling fully with air. It is common after surgery and can be "
        "caused by a blocked airway.",
        (("MedlinePlus: Atelectasis", "https://medlineplus.gov/ency/article/000065.htm"),
         ("Radiopaedia: Atelectasis", _R + "atelectasis"))),
    "Cardiomegaly": FindingInfo(
        "Cardiomegaly", "Cardiomegaly",
        "The heart appears larger than usual. This is usually a sign of another condition that strains the heart, "
        "such as long-term high blood pressure or coronary artery disease.",
        (("Cleveland Clinic: Enlarged heart (cardiomegaly)",
          "https://my.clevelandclinic.org/health/diseases/21490-enlarged-heart-cardiomegaly"),
         ("Radiopaedia: Cardiomegaly", _R + "cardiomegaly"))),
    "Effusion": FindingInfo(
        "Effusion", "Effusion",
        "Fluid has collected in the space between the lungs and the chest wall (pleural effusion). Heart failure "
        "is a common cause.",
        (("MedlinePlus: Pleural effusion", "https://medlineplus.gov/ency/article/000086.htm"),
         ("Radiopaedia: Pleural effusion", _R + "pleural-effusion"))),
    "Infiltration": FindingInfo(
        "Infiltration", "Infiltration",
        "Material such as fluid or inflammatory cells may be present in lung tissue. This can be associated with "
        "infection or inflammation.",
        (("NCBI MedGen (NCI definition): Pulmonary infiltrates", "https://www.ncbi.nlm.nih.gov/medgen/116009"),)),
    "Mass": FindingInfo(
        "Mass", "Mass",
        "A rounded area of abnormal tissue in the lung larger than about 3 cm. Masses are usually examined further "
        "to find their cause.",
        (("Annals of Thoracic Medicine (2019): Lung nodules — a comprehensive review",
          "https://pmc.ncbi.nlm.nih.gov/articles/PMC6784443/"),)),
    "Nodule": FindingInfo(
        "Nodule", "Nodule",
        "A small, round spot in the lung, up to about 3 cm across. Most nodules are not cancer and are often "
        "checked again with follow-up imaging.",
        (("Cleveland Clinic: Pulmonary (lung) nodules",
          "https://my.clevelandclinic.org/health/diseases/14799-pulmonary-nodules"),
         ("Annals of Thoracic Medicine (2019): Lung nodules — a comprehensive review",
          "https://pmc.ncbi.nlm.nih.gov/articles/PMC6784443/"))),
    "Pneumonia": FindingInfo(
        "Pneumonia", "Pneumonia",
        "An infection that fills the lung's air sacs with fluid or pus. It is most often caused by bacteria or "
        "viruses.",
        (("MedlinePlus: Pneumonia", "https://medlineplus.gov/pneumonia.html"),)),
    "Pneumothorax": FindingInfo(
        "Pneumothorax", "Pneumothorax",
        "Air has leaked into the space around the lung, which can stop the lung from expanding fully. It can "
        "follow a chest injury or occur without a clear cause.",
        (("MedlinePlus: Collapsed lung (pneumothorax)", "https://medlineplus.gov/ency/article/000087.htm"),
         ("Radiopaedia: Pneumothorax", _R + "pneumothorax"))),
    "Consolidation": FindingInfo(
        "Consolidation", "Consolidation",
        "Air in part of the lung is replaced by fluid or other material. This can occur with pneumonia.",
        (("NCBI MedGen (NCI definition): Lung consolidation", "https://www.ncbi.nlm.nih.gov/medgen/636648"),)),
    "Edema": FindingInfo(
        "Edema", "Edema",
        "Fluid has built up inside the lungs (pulmonary edema), which can make breathing harder. Heart problems are "
        "a common cause.",
        (("MedlinePlus: Pulmonary edema", "https://medlineplus.gov/ency/article/000140.htm"),
         ("Radiopaedia: Pulmonary oedema", _R + "pulmonary-oedema"))),
    "Emphysema": FindingInfo(
        "Emphysema", "Emphysema",
        "The walls of the lung's tiny air sacs are damaged, so the lungs move air less efficiently. It is most "
        "often linked to long-term smoking.",
        (("MedlinePlus: Emphysema", "https://medlineplus.gov/emphysema.html"),
         ("Radiopaedia: Emphysema", _R + "emphysema"))),
    "Fibrosis": FindingInfo(
        "Fibrosis", "Fibrosis",
        "Scar tissue has built up in the lungs, making them thicker and stiffer. In many cases no specific cause "
        "is found.",
        (("MedlinePlus: Pulmonary fibrosis", "https://medlineplus.gov/pulmonaryfibrosis.html"),)),
    "Pleural_Thickening": FindingInfo(
        "Pleural_Thickening", "Pleural Thickening",
        "The thin lining around the lungs has become thicker or scarred. It can follow earlier inflammation or "
        "asbestos exposure.",
        (("Journal of Occupational Medicine and Toxicology (2008): Asbestos-related diffuse pleural thickening — "
          "a review", "https://pmc.ncbi.nlm.nih.gov/articles/PMC2553409/"),)),
    "Hernia": FindingInfo(
        "Hernia", "Hernia",
        "Part of an abdominal organ, often the stomach, appears to push up through the diaphragm into the chest. "
        "It often causes no symptoms.",
        (("MedlinePlus: Hiatal hernia", "https://medlineplus.gov/hiatalhernia.html"),
         ("Radiopaedia: Hiatus hernia", _R + "hiatus-hernia"))),
}

assert list(FINDINGS) == list(config.LABELS), "FINDINGS must cover every label in config order"


def get_info(label: str) -> FindingInfo:
    return FINDINGS[label]


def all_references() -> list[tuple[str, str]]:
    """Unique (title, url) references, in label order, plus the general chest X-ray source."""
    seen, refs = set(), []
    for info in FINDINGS.values():
        for title, url in info.sources:
            if url not in seen:
                seen.add(url)
                refs.append((title, url))
    refs.append(CHEST_XRAY_SOURCE)
    return refs
