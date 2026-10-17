# ThoraxAI - Medical reference sources

Each finding card shows a one- or two-sentence explanation from `src/findings_info.py`, the single source of
truth for finding text. The explanations are general educational content, limited to statements supported by
the sources below (read on **2026-10-17**). Earlier, longer drafts were checked against the same sources;
statements the sources did not support were removed.

The text is not medical advice, contains no treatment recommendations, and does not describe any particular
image. Re-check the links periodically, because external pages can change.

| Finding | Explanation shown in the app | Sources |
|---|---|---|
| Atelectasis | Part of a lung has collapsed or is not filling fully with air. It is common after surgery and can be caused by a blocked airway. | [MedlinePlus: Atelectasis](https://medlineplus.gov/ency/article/000065.htm)<br>[Radiopaedia: Atelectasis](https://radiopaedia.org/articles/atelectasis) |
| Cardiomegaly | The heart appears larger than usual. This is usually a sign of another condition that strains the heart, such as long-term high blood pressure or coronary artery disease. | [Cleveland Clinic: Enlarged heart (cardiomegaly)](https://my.clevelandclinic.org/health/diseases/21490-enlarged-heart-cardiomegaly)<br>[Radiopaedia: Cardiomegaly](https://radiopaedia.org/articles/cardiomegaly) |
| Effusion | Fluid has collected in the space between the lungs and the chest wall (pleural effusion). Heart failure is a common cause. | [MedlinePlus: Pleural effusion](https://medlineplus.gov/ency/article/000086.htm)<br>[Radiopaedia: Pleural effusion](https://radiopaedia.org/articles/pleural-effusion) |
| Infiltration | Material such as fluid or inflammatory cells may be present in lung tissue. This can be associated with infection or inflammation. | [NCBI MedGen (NCI definition): Pulmonary infiltrates](https://www.ncbi.nlm.nih.gov/medgen/116009) |
| Mass | A rounded area of abnormal tissue in the lung larger than about 3 cm. Masses are usually examined further to find their cause. | [Annals of Thoracic Medicine (2019): Lung nodules — a comprehensive review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6784443/) |
| Nodule | A small, round spot in the lung, up to about 3 cm across. Most nodules are not cancer and are often checked again with follow-up imaging. | [Cleveland Clinic: Pulmonary (lung) nodules](https://my.clevelandclinic.org/health/diseases/14799-pulmonary-nodules)<br>[Annals of Thoracic Medicine (2019): Lung nodules — a comprehensive review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6784443/) |
| Pneumonia | An infection that fills the lung's air sacs with fluid or pus. It is most often caused by bacteria or viruses. | [MedlinePlus: Pneumonia](https://medlineplus.gov/pneumonia.html) |
| Pneumothorax | Air has leaked into the space around the lung, which can stop the lung from expanding fully. It can follow a chest injury or occur without a clear cause. | [MedlinePlus: Collapsed lung (pneumothorax)](https://medlineplus.gov/ency/article/000087.htm)<br>[Radiopaedia: Pneumothorax](https://radiopaedia.org/articles/pneumothorax) |
| Consolidation | Air in part of the lung is replaced by fluid or other material. This can occur with pneumonia. | [NCBI MedGen (NCI definition): Lung consolidation](https://www.ncbi.nlm.nih.gov/medgen/636648) |
| Edema | Fluid has built up inside the lungs (pulmonary edema), which can make breathing harder. Heart problems are a common cause. | [MedlinePlus: Pulmonary edema](https://medlineplus.gov/ency/article/000140.htm)<br>[Radiopaedia: Pulmonary oedema](https://radiopaedia.org/articles/pulmonary-oedema) |
| Emphysema | The walls of the lung's tiny air sacs are damaged, so the lungs move air less efficiently. It is most often linked to long-term smoking. | [MedlinePlus: Emphysema](https://medlineplus.gov/emphysema.html)<br>[Radiopaedia: Emphysema](https://radiopaedia.org/articles/emphysema) |
| Fibrosis | Scar tissue has built up in the lungs, making them thicker and stiffer. In many cases no specific cause is found. | [MedlinePlus: Pulmonary fibrosis](https://medlineplus.gov/pulmonaryfibrosis.html) |
| Pleural Thickening | The thin lining around the lungs has become thicker or scarred. It can follow earlier inflammation or asbestos exposure. | [Journal of Occupational Medicine and Toxicology (2008): Asbestos-related diffuse pleural thickening — a review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2553409/) |
| Hernia | Part of an abdominal organ, often the stomach, appears to push up through the diaphragm into the chest. It often causes no symptoms. | [MedlinePlus: Hiatal hernia](https://medlineplus.gov/hiatalhernia.html)<br>[Radiopaedia: Hiatus hernia](https://radiopaedia.org/articles/hiatus-hernia) |
| General: what a chest X-ray shows | — | [MedlinePlus: Chest x-ray](https://medlineplus.gov/ency/article/003804.htm) |

## Source types

- **MedlinePlus** (U.S. National Library of Medicine, NIH): consumer health encyclopedia and topic pages.
- **NCBI MedGen**: NIH concept definitions (from the National Cancer Institute thesaurus).
- **PubMed Central (PMC)**: peer-reviewed review articles.
- **Cleveland Clinic**: patient health library.
- **Radiopaedia**: peer-reviewed radiology reference.

## Dataset label note

ChestX-ray14 labels were text-mined from radiology reports. The label *Hernia* does not specify the hernia
type; its explanation describes a hiatal hernia (part of the stomach above the diaphragm) as the common case.
