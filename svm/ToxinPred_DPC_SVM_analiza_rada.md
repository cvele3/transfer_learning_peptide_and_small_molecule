# ToxinPred — DPC + SVM model: Detaljna analiza rada

**Rad:** Gupta S, Kapoor P, Chaudhary K, Gautam A, Kumar R, et al. (2013)
*In Silico Approach for Predicting Toxicity of Peptides and Proteins.*
PLoS ONE 8(9): e73957.
[doi:10.1371/journal.pone.0073957](https://doi.org/10.1371/journal.pone.0073957)

---

## 1. Dataset

### 1.1 Pozitivni primjeri (toksicni peptidi)

- Prikupljeno iz 6 baza podataka: **ATDB**, **Arachno-Server**, **ConoServer**, **DBETH**, **BTXpred**, **NTXpred** i **SwissProt** (kljucna rijec KW-0800 = toxin).
- Kriterij: duljina **<= 35 aminokiselinskih ostataka**, samo **prirodne aminokiseline**.
- Iz SwissProt-a pronadjeno 803 toksicnih proteina (<35 AA). Nakon uklanjanja duplikata s ostalim bazama, ostalo je 303 jedinstvena iz SwissProt-a.
- **Ukupno: 1805 jedinstvenih toksicnih peptida/proteina.**

### 1.2 Negativni primjeri (netoksicni peptidi)

- Izvor: **UniProt** (SwissProt + TrEMBL).
- Kljucne rijeci za pretragu: `NOT KW-0800 NOT KW-0020` (iskljuceni toksini i alergeni).
- Kriterij: duljina **<= 35 aminokiselinskih ostataka**.
- Nakon uklanjanja sekvenci koje sadrze nestandardne aminokiseline (B, J, O, U, X, Z): **3593 netoksicna peptida**.

### 1.3 Omjer klasa

| Dataset | Toksicni | Netoksicni | Omjer |
|---|---|---|---|
| **Glavni (nebalansirani)** | 1805 | 3593 | ~1:2 |
| **Glavni (balansirani)** | 1805 | 1805 (nasumicno odabrani) | 1:1 |
| **Alternativni (nebalansirani)** | 1805 | 12541 | ~1:7 |

### 1.4 Nezavisni testni skup

- **303 toksicna peptida** iz SwissProt-a (KW-0800) koji su uklonjeni iz trening skupa.
- **303 netoksicna peptida** nasumicno odabrana iz negativnog skupa.
- Koristi se za procjenu over-optimizacije modela.

---

## 2. Feature engineering — Dipeptide Composition (DPC)

### 2.1 Opis metode

DPC (Dipeptide Composition) kodira peptidnu sekvencu kao vektor normaliziranih frekvencija svih mogucih parova uzastopnih aminokiselina.

### 2.2 Postupak

1. Definira se skup od **20 standardnih aminokiselina**: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y.
2. Generira se svih **400 mogucih dipeptida** (20 x 20): AA, AC, AD, ..., YW, YY.
3. Za peptidnu sekvencu duljine **L**:
   - Kliznim prozorom velicine **2** prolazi se kroz sekvencu.
   - Broji se svaka dipeptidna kombinacija.
   - Ukupan broj dipeptida u sekvenci = **L - 1**.
4. Svaka komponenta vektora normalizira se:

   ```
   DPC(i) = count(dipeptid_i) / (L - 1)
   ```

5. Rezultat: **400-dimenzionalni vektor** normaliziranih frekvencija za svaku sekvencu.

### 2.3 Primjer

Za sekvencu `ACDEF` (L = 5):
- Dipeptidi: AC, CD, DE, EF (ukupno 4 = L-1)
- DPC(AC) = 1/4 = 0.25
- DPC(CD) = 1/4 = 0.25
- DPC(DE) = 1/4 = 0.25
- DPC(EF) = 1/4 = 0.25
- Svi ostali DPC(i) = 0

### 2.4 Zasto DPC

Rad navodi da DPC, za razliku od jednostavne aminokiselinske kompozicije (AAC, 20 dimenzija), **inkorporira i informaciju o redoslijedu aminokiselina** (lokalni kontekst susjednih parova), sto daje bolju diskriminativnu moc.

---

## 3. SVM model

### 3.1 Alat

- **SVMlight** (Joachims, 1999) — specijalizirana implementacija SVM-a optimizirana za velike skupove podataka.
- Referenca: Joachims T (1999) *Making large-scale support vector machine learning practical.* In: Scholkopf B, Burges C, Smola A, editors. Advances in Kernel Methods. Cambridge, MA: MIT Press. pp 169-184.

### 3.2 Kernel

- **RBF (Radial Basis Function)** kernel.
- RBF kernel implicitno mapira podatke u visoko-dimenzionalni prostor gdje linearno razdvajanje postaje moguce.

### 3.3 Hiperparametri

- Rad **ne navodi eksplicitne vrijednosti C i gamma** za dipeptidni model.
- SVMlight koristi vlastitu internu proceduru za optimizaciju parametara.
- **Cost factor: 1** — jednaki troskovi pogresne klasifikacije za obje klase (toksicno i netoksicno).

### 3.4 Trening

- Ulaz u SVM: 400-dimenzionalni DPC vektori za svaku sekvencu.
- Izlaz: binarna klasifikacija — **toksicno (1)** ili **netoksicno (0)**.

---

## 4. Evaluacija

### 4.1 Metoda evaluacije

- **5-fold unakrsna validacija (cross-validation)**:
  - Dataset se dijeli na 5 jednakih dijelova.
  - U svakoj iteraciji, 4 dijela su za trening, 1 za testiranje.
  - Postupak se ponavlja 5 puta, svaki dio jednom sluzi kao test skup.
  - Prijavljuju se prosjecne metrike preko svih 5 foldova.

### 4.2 Metrike

Rad koristi sljedece metrike za evaluaciju:

| Metrika | Formula | Opis |
|---|---|---|
| **Sensitivity (Sn)** | Sn = TP / (TP + FN) | Udio tocno prepoznatih toksicnih peptida |
| **Specificity (Sp)** | Sp = TN / (TN + FP) | Udio tocno prepoznatih netoksicnih peptida |
| **Accuracy (Ac)** | Ac = (TP + TN) / (TP + FP + TN + FN) | Ukupna tocnost klasifikacije |
| **MCC** | MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Matthews Correlation Coefficient — balansirana mjera koja uzima u obzir sve cetiri kategorije konfuzijske matrice |

Gdje je:
- **TP** = True Positive (toksicni tocno klasificirani kao toksicni)
- **TN** = True Negative (netoksicni tocno klasificirani kao netoksicni)
- **FP** = False Positive (netoksicni pogresno klasificirani kao toksicni)
- **FN** = False Negative (toksicni pogresno klasificirani kao netoksicni)

### 4.3 Dodatne napomene o metrikama

- **MCC** se smatra najinformativnijom pojedinacnom metrikom za binarnu klasifikaciju jer uzima u obzir nebalansiranost klasa. Raspon je od -1 (potpuno pogresno) do +1 (savrseno).
- Rad koristi **threshold** za klasifikaciju, pri cemu se SVM score usporeduje s pragom. Razliciti pragovi daju razlicite omjere Sensitivity/Specificity.

---

## 5. Rezultati DPC + SVM modela

### 5.1 Glavni dataset (nebalansirani, 1805 + 3593)

| Threshold | Sensitivity | Specificity | Accuracy | MCC |
|---|---|---|---|---|
| **-1.0** | 99.56 | 58.37 | 72.14 | 0.56 |
| **-0.5** | 97.62 | 80.74 | 86.38 | 0.73 |
| **-0.4** | 96.73 | 84.32 | 88.47 | 0.77 |
| **-0.3** | 95.18 | 87.69 | 90.19 | 0.80 |
| **-0.2** | 93.30 | 90.37 | 91.35 | 0.82 |
| **-0.1** | 90.86 | 92.32 | 91.83 | 0.82 |
| **0.0 (default)** | 87.09 | 98.22 | **94.50** | **0.88** |
| **0.1** | 82.88 | 98.78 | 93.47 | 0.86 |
| **0.2** | 77.62 | 99.14 | 91.95 | 0.83 |
| **0.3** | 72.13 | 99.53 | 90.37 | 0.79 |
| **0.4** | 65.98 | 99.64 | 88.39 | 0.75 |
| **0.5** | 59.89 | 99.72 | 86.42 | 0.70 |
| **1.0** | 34.90 | 99.89 | 78.17 | 0.50 |

**Optimalni rezultat (threshold = 0.0): Accuracy = 94.50%, MCC = 0.88**

### 5.2 Glavni dataset (balansirani, 1805 + 1805)

| Threshold | Sensitivity | Specificity | Accuracy | MCC |
|---|---|---|---|---|
| **0.0 (default)** | 87.09 | **100.0** | **93.88** | **0.88** |

Napomena: Balansirani dataset ima 1805 nasumicno odabranih netoksicnih peptida. Performanse su gotovo identicne nebalansiranom datasetu.

### 5.3 Nezavisni testni skup (303 + 303)

- Accuracy: **~90%**
- Ovo potvrdjuje da model nije pretjerano optimiziran (over-fitted) na trening podatke.

### 5.4 Usporedba s AAC modelom

| Model | Znacajke | Dimenzija | Accuracy | MCC |
|---|---|---|---|---|
| **AAC (Amino Acid Composition)** | 20 AA frekvencija | 20 | 91.64% | 0.81 |
| **DPC (Dipeptide Composition)** | 400 dipeptidnih frekvencija | 400 | **94.50%** | **0.88** |

DPC model nadmasuje AAC model za ~3% accuracy i 0.07 MCC, sto pokazuje vaznost ukljucivanja informacije o redoslijedu (susjednim parovima) aminokiselina.

### 5.5 Binary profile model

- SVM model na binarnim profilima sekvenci — nije bio uspjesniji od kompozicijskih modela.
- Razlog: cistein (Cys) je dominantan na vecini pozicija, sto smanjuje diskriminativnu moc profila.

### 5.6 Motif-based model

- Koristi MEME/MAST za ekstrakciju motiva iz toksicnih peptida.
- Performanse: razumne, ali slabije od DPC modela.
- **Ogranicenje**: motivi nisu ekstrahirani per-fold, vec iz svih toksicnih peptida odjednom.

### 5.7 Hybrid model (DPC + Motifs)

- Kombinira DPC kompoziciju s informacijom o motivima.
- Blago poboljsava performanse u odnosu na cisti DPC model.
- Ovo je **najbolji model** u radu, koristi se na ToxinPred web serveru.

---

## 6. Alternativni dataset (1805 + 12541)

Rad takodjer razvija modele na **alternativnom datasetu** s vecim omjerom nebalansiranosti (~1:7).

### 6.1 Rezultati DPC modela na alternativnom datasetu

| Threshold | Sensitivity | Specificity | Accuracy | MCC |
|---|---|---|---|---|
| **0.0 (default)** | 55.01 | 99.51 | 93.72 | 0.69 |
| **-0.5** | 95.18 | 83.81 | 85.29 | 0.57 |
| **-0.3** | 89.47 | 92.43 | 92.04 | 0.69 |
| **-0.1** | 76.34 | 96.60 | 93.97 | 0.72 |

Na ovom datasetu performanse su nize (posebno Sensitivity pri default thresholdu), sto se ocekuje jer je omjer klasa 1:7.

---

## 7. Kljucni zakljucci o DPC + SVM modelu

1. **DPC je najefikasnija single-feature metoda** u radu — nadmasuje AAC i binary profile pristupe.
2. **400-dimenzionalni vektor** normaliziranih dipeptidnih frekvencija dokazano dobro diskriminira toksicne od netoksicnih peptida.
3. **SVM s RBF kernelom** (SVMlight) s cost factor = 1 daje konzistentne rezultate na razlicitim datasetima.
4. **Threshold igra kljucnu ulogu** — promjenom praga mogu se balansirati Sensitivity i Specificity ovisno o potrebi primjene.
5. Model postize **~90% accuracy na nezavisnom testnom skupu**, sto potvrdjuje generalizabilnost.
6. Performanse na **balansiranom i nebalansiranom** glavnom datasetu su gotovo identicne (93.88% vs 94.50%), sto sugerira robusnost pristupa.

---

## 8. Reference

1. Gupta S, Kapoor P, Chaudhary K, Gautam A, Kumar R, et al. (2013) *In Silico Approach for Predicting Toxicity of Peptides and Proteins.* PLoS ONE 8(9): e73957. [doi:10.1371/journal.pone.0073957](https://doi.org/10.1371/journal.pone.0073957)
2. Joachims T (1999) Making large-scale support vector machine learning practical. In: Scholkopf B, Burges C, Smola A, editors. Advances in Kernel Methods. Cambridge, MA: MIT Press. pp 169-184.
