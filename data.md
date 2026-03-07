The **YOTO (You Only Think Once) dataset** on OpenNeuro (ds005815) is the most versatile find for Task 1. It contains EEG from **26 participants** recorded at **1000 Hz** during unimodal visual, unimodal auditory, and combined audiovisual perception plus mental imagery. Visual stimuli include a gray square and two facial images; auditory stimuli include three piano tones at distinct pitches — **C (261.63 Hz), D (293.66 Hz), E (329.63 Hz)** — plus three vowels (/a/, /i/, /o/). The factorial design yields **27 unique stimulus conditions**, making it directly suitable for modality classification (visual vs. auditory) and within-auditory pitch classification (three discrete frequencies). The dataset is BIDS-formatted and freely downloadable.

- **Link:** [https://openneuro.org/datasets/ds005815](https://openneuro.org/datasets/ds005815)
- **Format:** BIDS (.vhdr/.vmrk/.eeg)
- **Relevance:** Task 1 — modality classification ★★★★★; within-auditory pitch classification ★★★★☆; Task 2 — not an explicit congruency paradigm, but AV combinations could be reframed for crossmodal analysis ★★★☆☆

The YOTO dataset does not use simple colored patches for its visual stimuli (it uses grayscale squares and faces), so it cannot serve Task 1’s color component. Combining YOTO with a dedicated color EEG dataset is the strongest strategy.

---

## Top datasets for visual color classification from EEG

Color decoding from EEG is a niche but growing area. The datasets below are ordered by relevance and accessibility.

**Hajonides et al. (2021) — “Decoding visual colour from scalp EEG”** is the benchmark. Thirty subjects viewed bilateral Gabor gratings in **48 distinct hues sampled from CIELAB color space** (fixed lightness L=54), presented for 300 ms. EEG was recorded with a **61-channel BioSemi** system at 1000 Hz. LDA-based decoding of 12 color bins from posterior electrodes (100–350 ms window) was demonstrated in the paper, confirming the data supports color classification. Preprocessed epoched data and analysis scripts are hosted on OSF. Access requires a view-only token published in the paper, but the data is functionally open.

- **Link:** [https://osf.io/j289e/?view_only=b13407009b4245f7950960c34a5474a6](https://osf.io/j289e/?view_only=b13407009b4245f7950960c34a5474a6)
- **Format:** MATLAB (.mat), preprocessed epochs
- **Relevance:** Task 1 — within-visual color/hue classification ★★★★★

**Wu et al. (2023) — “Decoding the neural representation of visual RGB color”** offers a simpler paradigm: **red, green, and blue** color patches presented as fast random visual events. The authors achieved successful SVM-based 3-class RGB classification. Data was recorded with a Neuracle EEG system at **1000 Hz** and stored in BDF format with open access (CC BY 4.0). Subject count is approximately 10–15. This is the most straightforward color classification dataset if a simple 3-class problem is desired.

- **Link:** Data referenced in PeerJ Computer Science (doi: 10.7717/peerj-cs.1376); check paper’s data availability statement
- **Format:** BDF (readable via MNE-Python or EEGLAB)
- **Relevance:** Task 1 — 3-class RGB classification ★★★★☆

**Bae & Luck — Color working memory decoding** provides another strong option. Subjects viewed colored stimuli drawn from a **16-color wheel** in a delayed-estimation paradigm (view, maintain, reproduce color). Data from **16–30 subjects** using BioSemi EEG (20+ channels) is hosted on OSF with MATLAB analysis scripts. The paradigm captures both perception and maintenance periods, enabling classification at multiple cognitive stages.

- **Link:** [https://osf.io/jnwut/](https://osf.io/jnwut/)
- **Format:** MATLAB (.mat)
- **Relevance:** Task 1 — within-visual color classification ★★★★☆

Chauhan et al.- Decoding of EEG signals reveals non-uniformities in the neural geometry of colour

paper : [https://www.sciencedirect.com/science/article/pii/S1053811923000332](https://www.sciencedirect.com/science/article/pii/S1053811923000332)

data : [https://osf.io/v9ewj/overview](https://osf.io/v9ewj/overview)

Torres garcia et al.

paper : [https://www.researchgate.net/publication/335589358_Analyzing_the_Recognition_of_Color_Exposure_and_Imagined_Color_from_EEG_Signals](https://www.researchgate.net/publication/335589358_Analyzing_the_Recognition_of_Color_Exposure_and_Imagined_Color_from_EEG_Signals)

data :  contact Saim Rasheed, original paper with original data ([https://www.researchgate.net/publication/283766150_Classification_of_EEG_Signals_Produced_by_RGB_Colour_Stimuli](https://www.researchgate.net/publication/283766150_Classification_of_EEG_Signals_Produced_by_RGB_Colour_Stimuli))

Two additional datasets merit mention for color. **[Chauhan et al. (2023)](https://www.sciencedirect.com/science/article/pii/S1053811923000332)** used isoluminant unique hues (red, green, blue, yellow, and intermediates) with ~20 subjects on 64-channel EEG, achieving hue decoding with RSA-based representational geometry analysis. Data may be available by contacting the authors (University of Edinburgh). **Torres-García et al. (2019)** recorded RGB color exposure and color imagination with 7–18 subjects using consumer-grade Emotiv EPOC (4–8 channels, 256 Hz), achieving ~45% three-class accuracy with EEGNet. Both require author contact for data access.

---

## Top datasets for auditory tone and pitch classification

Beyond YOTO (described above), several auditory oddball datasets provide tone stimuli useful for within-auditory classification.

**Nencki-Symfonia (OpenNeuro ds004621)** includes a **3-stimulus auditory oddball** (standard, target, distractor) alongside other cognitive tasks, recorded from **42 healthy young adults** using high-density EEG in BIDS format. The three distinct auditory stimuli enable multi-class tone classification. It also includes resting-state and interference tasks for broader analysis.

- **Link:** [https://openneuro.org/datasets/ds004621](https://openneuro.org/datasets/ds004621)
- **Format:** BIDS
- **Relevance:** Task 1 — 3-class auditory classification ★★★★☆
- [https://openneuro.org/datasets/ds004621/versions/1.0.4](https://openneuro.org/datasets/ds004621/versions/1.0.4)

**OpenNeuro ds003490 — 3-stimulus auditory oddball (Parkinson’s study)** provides the same 3-stimulus oddball paradigm from **50 subjects** (25 Parkinson’s patients, 25 controls), with patients tested both on and off medication. The clinical population limits generalizability, but the healthy control subset alone is substantial.

- **Link:** [https://openneuro.org/datasets/ds003490](https://openneuro.org/datasets/ds003490)
- **Format:** BIDS
- **Relevance:** Task 1 — auditory classification ★★★☆☆ (clinical caveat)
- [https://openneuro.org/datasets/ds003490/versions/1.1.0](https://openneuro.org/datasets/ds003490/versions/1.1.0)

**ERP CORE** deserves special attention for modality classification. The same **40 subjects** completed six paradigms including an **auditory MMN passive oddball** (standard vs. deviant tones differing in intensity) and several visual tasks (N170 face perception, N2pc visual search, P3b active visual oddball, N400 word pairs, flankers). While the MMN uses a single frequency with intensity variation (not pitch), the fact that visual and auditory paradigms are recorded from the **same 40 subjects** makes this dataset uniquely valuable for cross-paradigm modality classification.

- **Link:** [https://osf.io/thsqg/](https://osf.io/thsqg/) and [https://openneuro.org/datasets/ds004660](https://openneuro.org/datasets/ds004660)
- **Format:** BIDS (BrainVision files) with EEGLAB/ERPLAB scripts
- **Relevance:** Task 1 — modality classification (visual vs. auditory) ★★★★★; within-modality pitch classification ★★☆☆☆
- [https://www.sciencedirect.com/science/article/pii/S1053811920309502?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S1053811920309502?via%3Dihub)

**OpenNeuro ds000116 — Auditory and visual oddball EEG-fMRI** provides both modalities from **17 subjects** with matched paradigm structure: auditory oddball (390 Hz tone vs. broadband sound) and visual oddball (small green circle vs. large red circle). Data is BIDS-formatted with simultaneous fMRI. The concurrent MRI recording introduces gradient artifacts, but corrected EEG files are included.

- **Link:** [https://openneuro.org/datasets/ds000116](https://openneuro.org/datasets/ds000116)
- **Format:** BIDS (EEG + fMRI)
- **Relevance:** Task 1 — modality classification ★★★★☆; color classification ★★☆☆☆ (only 2 colors, confounded with target status)
- [https://openneuro.org/datasets/ds000116/versions/00003](https://openneuro.org/datasets/ds000116/versions/00003)


| Dataset                        | Subjects | Channels / Rate          | Paradigm                                              | Task Relevance                  | Access                     |
| ------------------------------ | -------- | ------------------------ | ----------------------------------------------------- | ------------------------------- | -------------------------- |
| **YOTO** (ds005815)            | 26       | ~32+ ch / 1000 Hz        | 3 tones + 3 vowels + 3 visual, 27 AV combos + imagery | T1: modality ★★★★★, pitch ★★★★☆ | Open (OpenNeuro)           |
| **Hajonides 2021** (OSF j289e) | 30       | 61 ch BioSemi / 1000 Hz  | 48 CIELAB colors, Gabor gratings, working memory      | T1: color ★★★★★                 | Open (OSF, token in paper) |
| **Wu 2023** (PeerJ CS)         | ~10–15   | Neuracle / 1000 Hz       | Red, green, blue color patches, fast VEP              | T1: color ★★★★☆                 | Open (CC BY 4.0)           |
| **Bae & Luck** (OSF jnwut)     | 16–30    | 20+ ch BioSemi           | 16-color wheel, delayed estimation                    | T1: color ★★★★☆                 | Open (OSF)                 |
| **ERP CORE** (ds004660)        | 40       | ~30 ch BioSemi           | 6 paradigms: MMN, N170, N2pc, P3b, N400, flankers     | T1: modality ★★★★★              | Open (CC BY-SA 4.0)        |
| **Nencki-Symfonia** (ds004621) | 42       | High-density             | 3-stimulus oddball + MSIT + resting                   | T1: auditory ★★★★☆              | Open (OpenNeuro)           |
| **ds003490** (OpenNeuro)       | 50       | ~64 ch                   | 3-stimulus oddball (PD + controls)                    | T1: auditory ★★★☆☆              | Open (OpenNeuro)           |
| **ds000116** (OpenNeuro)       | 17       | 34 ch (EEG-fMRI)         | Auditory + visual oddball, matched paradigm           | T1: modality ★★★★☆              | Open (CC0)                 |
| **Wilson 2023** (ds004306)     | 12       | 124 ch ANT Neuro         | 3 concepts × 3 modalities (audio/text/image)          | T1: modality ★★★★☆; T2: ★★★☆☆   | Open (OpenNeuro)           |
| **Sciortino & Kayser 2023**    | ~20–30   | 128 ch BioSemi / 1028 Hz | SSVEP: pitch-size, pitch-hue, pitch-saturation        | T2: ★★★★★                       | On request (Bielefeld)     |
| **Brožová et al. 2025**        | 30       | ~64+ ch                  | IAT: pitch-size + pitch-elevation congruency          | T2: ★★★★★                       | On request (Bielefeld)     |
| **CNSP/Lalor Lab**             | 16–31    | 128 ch BioSemi / 512 Hz  | AV speech, congruent vs. incongruent attention        | T2: ★★★★☆                       | Partially open (CNSP)      |
| **THINGS-EEG** (OSF 3jk45)     | 10       | 64 ch                    | 16,740 natural images, 82K trials                     | T1: color (indirect) ★★★☆☆      | Open (OSF)                 |
| **EAV** (Zenodo)               | 42       | 30 ch / 500 Hz           | EEG + audio + video, emotion conversations            | T2: ★★☆☆☆                       | Open (Zenodo)              |


