# DPC + SVM referentni model — Dokumentacija

## Izvorni rad

**Gupta S, Kapoor P, Chaudhary K, Gautam A, Kumar R, et al. (2013)**
*In Silico Approach for Predicting Toxicity of Peptides and Proteins.*
PLoS ONE 8(9): e73957.
[doi:10.1371/journal.pone.0073957](https://doi.org/10.1371/journal.pone.0073957)

Web server: [ToxinPred](http://crdd.osdd.net/raghava/toxinpred/)

---

## 1. Opis metode iz rada

### 1.1 Dataset

- **Toksični peptidi (pozitivni primjeri):** 1805 jedinstvenih toksičnih peptida/proteina s ≤35 aminokiselinskih ostataka, prikupljenih iz baza: ATDB, Arachno-Server, ConoServer, DBETH, BTXpred, NTXpred i SwissProt (KW-0800).
- **Netoksični peptidi (negativni primjeri):** Peptidi iz UniProt-a s ključnim riječima `NOT KW-0800 NOT KW-0020` (isključeni toksini i alergeni), duljine ≤35 aminokiselina.
- **Omjer klasa:** Nebalansiran — otprilike 1:2 (toksični : netoksični) u glavnom datasetu, 1:7 u alternativnom datasetu.
- **Balansirani dataset:** Za kontrolu su nasumično odabrana jednaka količina (1805) netoksičnih peptida.

### 1.2 Feature engineering — Dipeptide Composition (DPC)

Rad koristi **dipeptidnu kompoziciju (DPC)** kao primarnu značajku za SVM model:

- **Dimenzija:** 400 (svih 20 × 20 kombinacija standardnih aminokiselina).
- **Postupak:**
  1. Kliznim prozorom veličine 2 prolazi se kroz peptidnu sekvencu.
  2. Broje se sve susjedne dipeptidne kombinacije (AA, AC, AD, ... YW, YY).
  3. Ukupan broj dipeptida u sekvenci duljine \( L \) iznosi \( L - 1 \).
  4. Svaka komponenta vektora normalizira se kao:

     \[
     \text{DPC}(i) = \frac{\text{count}(i)}{L - 1}
     \]

- **Redoslijed dipeptida:** Fiksan, leksikografski nad 20 standardnih aminokiselina (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).

### 1.3 Model — SVM

- **Alat:** SVMlight (Joachims, 1999)
- **Kernel:** RBF (radial basis function)
- **Hiperparametri:** Rad ne navodi eksplicitne vrijednosti C i gamma — SVMlight koristi vlastitu optimizaciju parametara.
- **Cost factor:** 1 (jednaki troškovi pogrešne klasifikacije za obje klase).

### 1.4 Evaluacija

- **Metoda:** 5-fold unakrsna validacija (cross-validation).
- **Metrike:** Accuracy, MCC (Matthews Correlation Coefficient), Sensitivity, Specificity.
- **Rezultati dipeptidnog modela:**
  - Accuracy: **94.50%**
  - MCC: **0.88**
- **Na balansiranom datasetu:**
  - Accuracy: **93.88%**
  - MCC: **0.88**
- **Na nezavisnom testnom skupu:** Accuracy oko **90%**.

### 1.5 Ostali modeli u radu (nisu implementirani)

Rad također opisuje:
- **Amino acid composition (AAC)** model — 20-dimenzionalni vektor.
- **Binary profile** model — SVM na binarnim profilima sekvenci.
- **Motif-based** model — koristi MEME/MAST za pronalaženje motiva u toksičnim peptidima.
- **Hybrid** model — kombinacija DPC + motif informacija (najbolji model u radu).

---

## 2. Naša implementacija

### 2.1 Što se poklapa s radom

| Aspekt | Rad (ToxinPred) | Naša implementacija |
|---|---|---|
| Feature vektor | DPC, 400 dimenzija | DPC, 400 dimenzija ✓ |
| Klizni prozor | veličine 2 | veličine 2 ✓ |
| Normalizacija | count / (L−1) | count / (L−1) ✓ |
| Redoslijed dipeptida | fiksni, leksikografski | fiksni, leksikografski ✓ |
| 20 std. aminokiselina | ACDEFGHIKLMNPQRSTVWY | ACDEFGHIKLMNPQRSTVWY ✓ |
| SVM kernel | RBF | RBF ✓ |
| Klasifikacija | binarna (toksičan/netoksičan) | binarna (1/0) ✓ |
| Dataset | 1805 toksičnih + netoksični iz UniProt | isti peptidi s ToxinPred servera ✓ |

### 2.2 Razlike od rada

| Aspekt | Rad (ToxinPred) | Naša implementacija | Razlog |
|---|---|---|---|
| Cross-validacija | 5-fold | 10-fold | Usklađeno s GNN modelima za fer usporedbu |
| SVM implementacija | SVMlight | scikit-learn SVC | Modernija biblioteka, isti algoritam |
| Class weight | cost factor = 1 | `class_weight="balanced"` | Bez ovoga SVM predviđa samo majority klasu na ovom datasetu |
| Feature scaling | nije navedeno | StandardScaler (per fold) | Potrebno za scikit-learn SVC s RBF kernelom |
| Hiperparametri | SVMlight default | C=5, gamma=0.001 | Ručno postavljeni referentni parametri |
| CV splitovi | interni (SVMlight) | isti kao GNN modeli | Omogućuje fer usporedbu s GNN i TL modelima |

### 2.3 Opis implementacijskih prilagodbi

#### Class weight: `balanced` umjesto cost factor = 1

Rad koristi cost factor = 1 (jednake kazne za obje klase), ali na balansiranom datasetu (1805 + 1805). Naš dataset ima omjer ~1:2 (1805 toksičnih + 3593 netoksičnih). Bez `class_weight="balanced"`, SVM na nebalansiranom datasetu predviđa sve uzorke kao majority klasu (netoksično), što rezultira nultim MCC, Precision, Recall i F1 metrikom.

Opcija `class_weight="balanced"` automatski skalira kazne obrnuto proporcionalno frekvenciji klasa:
\[
w_j = \frac{n}{k \cdot n_j}
\]
gdje je \( n \) ukupan broj uzoraka, \( k \) broj klasa, a \( n_j \) broj uzoraka klase \( j \).

#### Feature scaling: StandardScaler

SVMlight može interno drugačije rukovati skaliranjem značajki. Scikit-learn SVC s RBF kernelom zahtijeva standardizirane značajke jer RBF kernel računa euklidsku udaljenost u prostoru značajki. Bez skaliranja, značajke s većom varijancom dominiraju kernelom.

StandardScaler se primjenjuje per fold:
- **Fit** na training podatcima
- **Transform** na training i test podatcima

---

## 3. Struktura skripti

### 3.1 `prepare_dataset.py`

Priprema `DPC_SVM_dataset.xlsx`:
1. Preuzima FASTA sekvence s ToxinPred servera (pozitivni i negativni skup).
2. Filtrira nestandardne aminokiseline.
3. Konvertira sekvence u kanonski SMILES (RDKit `MolFromSequence`).
4. Mapira i filtrira prema postojećem `ToxinSequenceSMILES.xlsx`.
5. Uklanja duplikate.
6. Sprema XLSX sa stupcima: `FASTA | SMILES | ACTIVITY`.

### 3.2 `create_aligned_splits.py`

Kreira `aligned_peptide_data.pkl`:
1. Učitava `ToxinSequenceSMILES.xlsx` identično GNN pipelineu (isto filtriranje nevalidnih SMILES-a).
2. Učitava GNN CV splitove (`large_layers_cv_splits_peptide.pkl`).
3. Verificira podudaranje s GNN pickle-om.
4. Sprema usklađene FASTA sekvence, labele i CV splitove.

### 3.3 `dpc_svm_model.py`

Trenira i evaluira DPC + SVM model:
1. Učitava `aligned_peptide_data.pkl`.
2. Računa DPC vektore za sve sekvence.
3. Izvodi 10-fold CV s istim splitovima kao GNN modeli.
4. Za svaki fold: StandardScaler → SVM trening → evaluacija.
5. Sprema 10 modela u `models/dpc_svm_fold_{1..10}.pkl`.
6. Sprema rezultate u `DPC_SVM_results.xlsx` i `dpc_svm_fold_results.pkl`.

---

## 4. Datoteke i ovisnosti

### Ulazne datoteke
| Datoteka | Opis |
|---|---|
| `datasets/ToxinSequenceSMILES.xlsx` | Izvorni peptide dataset (SEQUENCE, SMILES, TOXICITY) |
| `large_layers_cv_splits/large_layers_cv_splits_peptide.pkl` | GNN CV splitovi (10 foldova) |
| `inflated_models/large_layers_overtrained_peptide.pkl` | GNN pickle (za verifikaciju labela) |

### Generirane datoteke
| Datoteka | Opis |
|---|---|
| `svm/aligned_peptide_data.pkl` | Usklađeni FASTA + labele + CV splitovi |
| `svm/DPC_SVM_dataset.xlsx` | FASTA + SMILES + ACTIVITY dataset |
| `svm/DPC_SVM_results.xlsx` | Rezultati evaluacije po foldovima |
| `svm/dpc_svm_fold_results.pkl` | Metrike po foldovima (za evaluacijske skripte) |
| `svm/models/dpc_svm_fold_{1..10}.pkl` | Naučeni SVM modeli + scaleri |

### Python ovisnosti
- `scikit-learn` (SVC, StandardScaler, metrike)
- `rdkit` (MolFromSmiles, MolFromSequence, kanonikalizacija)
- `pandas` (čitanje/pisanje XLSX)
- `numpy`

---

## 5. Redoslijed pokretanja

```bash
# 1. (Opcionalno) Priprema DPC_SVM_dataset.xlsx
python svm/prepare_dataset.py

# 2. Kreiranje usklađenih splitova
python svm/create_aligned_splits.py

# 3. Trening i evaluacija DPC + SVM modela
python svm/dpc_svm_model.py
```

---

## 6. Reference

1. Gupta S, Kapoor P, Chaudhary K, Gautam A, Kumar R, et al. (2013) *In Silico Approach for Predicting Toxicity of Peptides and Proteins.* PLoS ONE 8(9): e73957. [doi:10.1371/journal.pone.0073957](https://doi.org/10.1371/journal.pone.0073957)
2. Joachims T (1999) Making large-scale support vector machine learning practical. In: Schölkopf B, Burges C, Smola A, editors. Advances in Kernel Methods. Cambridge, MA: MIT Press. pp 169–184.
