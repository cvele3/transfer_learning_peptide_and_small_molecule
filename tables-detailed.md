# Detaljne Tablice za Znanstveni Rad

---

## TABLICA 1: Karakteristike Skupova Podataka

| Svojstvo | Peptidni Skup (ToxinSequenceSMILES) | Skup Malih Molekula (MolToxPredDataset) |
|----------|-------------------------------------|----------------------------------------|
| Ukupan broj uzoraka | | |
| Toksični uzorci (klasa 1) | | |
| Netoksični uzorci (klasa 0) | | |
| Omjer klasa (toksični:netoksični) | | |
| Prosječan broj atoma po grafu | | |
| Prosječan broj veza po grafu | | |
| Min / Max broj atoma | | |
| Broj jedinstvenih elemenata | | |
| Elementi prisutni | C, N, O, S, ... | C, N, O, S, Cl, F, ... |

---

## TABLICA 2: Konfiguracija Modela (DeepGraphCNN)

| Parametar | Standard | Large Layers | Inflated | Extra Inflated |
|-----------|----------|--------------|----------|----------------|
| k (SortPooling) | 25 | 25 | 25 | 25 |
| GNN slojevi | [25, 25, 25, 1] | [125, 125, 125, 1] | [512, 256, 128, 1] | [1024, 512, 256, 1] |
| Conv1D filteri | [16, 32, 128] | [16, 32, 128] | [16, 32, 128] | [16, 32, 128] |
| Aktivacijska funkcija (GNN) | tanh | tanh | tanh | tanh |
| Aktivacijska funkcija (Dense) | ReLU, Sigmoid | ReLU, Sigmoid | ReLU, Sigmoid | ReLU, Sigmoid |
| Dropout | 0.2 | 0.2 | 0.2 | 0.2 |
| Ukupno parametara | | | | |
| Trenirajućih parametara | | | | |

---

## TABLICA 3: Opis Metoda Prijenosnog Učenja

| Metoda | Strategija | Zamrznuti slojevi | Trenirajući slojevi | Learning Rate | Broj faza |
|--------|------------|-------------------|---------------------|---------------|-----------|
| Baseline | Trening od nule | Nijedan | Svi | 1e-4 | 1 |
| Freeze GNN | Zamrzni GNN, treniraj klasifikator | DeepGraphCNN, GraphConv | Conv1D, Dense, Dropout | 1e-4 | 1 |
| Freeze Readout | Zamrzni klasifikator, treniraj GNN | Dense, Dropout, Flatten | GNN slojevi | 1e-5 | 1 |
| Freeze All | Zamrzni sve + novi izlazni sloj | Svi originalni slojevi | Novi Dense(1) sloj | 1e-4 | 1 |
| Gradual Unfreezing | Postupno odmrzavanje s različitim LR | Faza 1→2→3 odmrzavanje | Postupno svi | 1e-3→1e-4→1e-5 | 3 |

**Detalji Gradual Unfreezing metode:**
| Faza | Trenirajući slojevi | Learning Rate | Broj epoha |
|------|---------------------|---------------|------------|
| Faza 1 | Samo završni slojevi | 1e-3 | 10 |
| Faza 2 | + Readout slojevi | 1e-4 | 10 |
| Faza 3 | + GNN slojevi (svi) | 1e-5 | 10 |

---

## TABLICA 4: Rezultati - Prijenos Mala Molekula → Peptid

| Metoda | ROC-AUC | MCC | GM | F1 | Precision | Recall |
|--------|---------|-----|-----|-----|-----------|--------|
| Baseline | ±  | ± | ± | ± | ± | ± |
| Freeze GNN | ± | ± | ± | ± | ± | ± |
| Freeze Readout | ± | ± | ± | ± | ± | ± |
| Freeze All | ± | ± | ± | ± | ± | ± |
| Gradual Unfreezing | ± | ± | ± | ± | ± | ± |

**Legenda:** Vrijednosti prikazane kao srednja vrijednost ± standardna devijacija preko 10-fold CV. 
**Bold** = najbolja metoda. * p<0.05 vs baseline, ** p<0.01 vs baseline (Nemenyi post-hoc test)

---

## TABLICA 5: Rezultati - Prijenos Peptid → Mala Molekula

| Metoda | ROC-AUC | MCC | GM | F1 | Precision | Recall |
|--------|---------|-----|-----|-----|-----------|--------|
| Baseline | ± | ± | ± | ± | ± | ± |
| Freeze GNN | ± | ± | ± | ± | ± | ± |
| Freeze Readout | ± | ± | ± | ± | ± | ± |
| Freeze All | ± | ± | ± | ± | ± | ± |
| Gradual Unfreezing | ± | ± | ± | ± | ± | ± |

---

## TABLICA 6: Usporedba po Veličini Modela - Smjer SMT → Peptid

| Veličina Modela | Baseline ROC-AUC | Najbolja TL Metoda | TL ROC-AUC | Δ ROC-AUC | p-vrijednost |
|-----------------|------------------|--------------------|-----------| ----------|--------------|
| Standard | ± | | ± | | |
| Large Layers | ± | | ± | | |
| Inflated | ± | | ± | | |
| Extra Inflated | ± | | ± | | |

**Δ ROC-AUC** = (Najbolji TL) - (Baseline). Pozitivna vrijednost = poboljšanje od prijenosa.

---

## TABLICA 7: Usporedba po Veličini Modela - Smjer Peptid → SMT

| Veličina Modela | Baseline ROC-AUC | Najbolja TL Metoda | TL ROC-AUC | Δ ROC-AUC | p-vrijednost |
|-----------------|------------------|--------------------|-----------| ----------|--------------|
| Standard | ± | | ± | | |
| Large Layers | ± | | ± | | |
| Inflated | ± | | ± | | |
| Extra Inflated | ± | | ± | | |

---

## TABLICA 8: Friedman Test - Statistička Značajnost

| Metrika | Friedman χ² (SMT→P) | p-vrijednost | Friedman χ² (P→SMT) | p-vrijednost |
|---------|---------------------|--------------|---------------------|--------------|
| ROC-AUC | | | | |
| MCC | | | | |
| GM | | | | |
| F1 | | | | |
| Precision | | | | |
| Recall | | | | |

**Značajno pri α=0.05:** p < 0.05 označava statistički značajnu razliku između metoda.

---

## TABLICA 9: Nemenyi Post-hoc Test - Usporedba s Baseline (ROC-AUC)

### Smjer: Mala Molekula → Peptid

| Usporedba | p-vrijednost | Značajno? | Δ ROC-AUC |
|-----------|--------------|-----------|-----------|
| Baseline vs Freeze GNN | | Da/Ne | |
| Baseline vs Freeze Readout | | Da/Ne | |
| Baseline vs Freeze All | | Da/Ne | |
| Baseline vs Gradual Unfreezing | | Da/Ne | |

### Smjer: Peptid → Mala Molekula

| Usporedba | p-vrijednost | Značajno? | Δ ROC-AUC |
|-----------|--------------|-----------|-----------|
| Baseline vs Freeze GNN | | Da/Ne | |
| Baseline vs Freeze Readout | | Da/Ne | |
| Baseline vs Freeze All | | Da/Ne | |
| Baseline vs Gradual Unfreezing | | Da/Ne | |

---

## TABLICA 10: Potpuna Nemenyi Matrica (primjer za ROC-AUC, SMT→P)

|  | Baseline | Freeze GNN | Freeze Readout | Freeze All | Gradual Unfreeze |
|--|----------|------------|----------------|------------|------------------|
| Baseline | - | | | | |
| Freeze GNN | | - | | | |
| Freeze Readout | | | - | | |
| Freeze All | | | | - | |
| Gradual Unfreeze | | | | | - |

*Vrijednosti su p-vrijednosti. Bold = p < 0.05 (statistički značajna razlika)*

---

## TABLICA 11: Sažetak - Učinkovitost Prijenosnog Učenja

| Smjer Prijenosa | Najbolja Metoda | Δ ROC-AUC | Δ MCC | Statistički Značajno? | Preporuka |
|-----------------|-----------------|-----------|-------|----------------------|-----------|
| Mala Molekula → Peptid | | | | Da/Ne | Koristiti/Ne koristiti |
| Peptid → Mala Molekula | | | | Da/Ne | Koristiti/Ne koristiti |

---

## TABLICA 12: Utjecaj Kapaciteta Modela na Transfer

| Veličina | Parametri | SMT→P Učinkovit? | P→SMT Učinkovit? | Optimalan za Transfer? |
|----------|-----------|------------------|------------------|------------------------|
| Standard | ~X | Da/Ne (Δ=) | Da/Ne (Δ=) | |
| Large Layers | ~X | Da/Ne (Δ=) | Da/Ne (Δ=) | |
| Inflated | ~X | Da/Ne (Δ=) | Da/Ne (Δ=) | |
| Extra Inflated | ~X | Da/Ne (Δ=) | Da/Ne (Δ=) | |

**Zaključak:** [Koja veličina modela daje najbolji transfer?]

---

## TABLICA 13: Hiperparametri Treniranja

| Parametar | Vrijednost |
|-----------|------------|
| Optimizer | Adam |
| Learning rate (baseline) | 1e-4 |
| Batch size | 32 |
| Maksimalan broj epoha | 10000 |
| Early stopping patience | 7 (baseline), 3 (TL metode) |
| Validacijski split | 20% od trening skupa |
| Cross-validacija | 10-fold stratificirana |
| Funkcija gubitka | Binary Cross-Entropy |

---

## KOMPAKTNE TABLICE ZA OGRANIČENI PROSTOR

### Kompaktna Tablica A: Glavni Rezultati (sve u jednoj tablici)

| Smjer | Metoda | ROC-AUC | MCC | Δ vs Baseline |
|-------|--------|---------|-----|---------------|
| **SMT→P** | Baseline | ± | ± | - |
| | Freeze GNN | ± | ± | |
| | Freeze Readout | ± | ± | |
| | Freeze All | ± | ± | |
| | Gradual Unfreezing | ± | ± | |
| **P→SMT** | Baseline | ± | ± | - |
| | Freeze GNN | ± | ± | |
| | Freeze Readout | ± | ± | |
| | Freeze All | ± | ± | |
| | Gradual Unfreezing | ± | ± | |

---

### Kompaktna Tablica B: Kapacitet Modela (obje smjerove)

| Veličina | SMT→P Baseline | SMT→P Najbolji TL | Δ | P→SMT Baseline | P→SMT Najbolji TL | Δ |
|----------|----------------|-------------------|---|----------------|-------------------|---|
| Standard | | | | | | |
| Large | | | | | | |
| Inflated | | | | | | |
| Extra Inf. | | | | | | |

---

## NAPOMENE ZA POPUNJAVANJE

1. **Vrijednosti upisati kao:** `0.XXX ± 0.0XX` (3 decimale za srednju vrijednost, 2 za std)
2. **Bold za:** najbolju vrijednost u stupcu
3. **Zvjezdice:** * za p<0.05, ** za p<0.01, *** za p<0.001
4. **Δ izračunati kao:** TL_vrijednost - Baseline_vrijednost
5. **Pozitivan Δ:** transfer je poboljšao performanse
6. **Negativan Δ:** transfer je pogoršao performanse (negative transfer)
