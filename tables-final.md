# Tablice za Znanstveni Rad - Finalna Verzija

**Priča rada:** Ispitujemo prijenosno učenje između domena peptida i malih molekula. Testiramo 4 strategije prijenosa na 4 veličine modela u oba smjera.

---

## TABLICA 1: Skupovi Podataka

| Skup | Domena | Uzoraka | Toksičnih | Netoksičnih | Omjer |
|------|--------|---------|-----------|-------------|-------|
| ToxinSequenceSMILES | Peptidi | | | | |
| MolToxPredDataset | Male molekule | | | | |

---

## TABLICA 2: Arhitektura Modela

| Veličina | GNN Slojevi | k | Conv1D Filteri | Parametara |
|----------|-------------|---|----------------|------------|
| Standard | [25, 25, 25, 1] | 25 | [16, 32, 128] | ~X |
| Large Layers | [125, 125, 125, 1] | 25 | [16, 32, 128] | ~X |
| Inflated | [512, 256, 128, 1] | 25 | [16, 32, 128] | ~X |
| Extra Inflated | [1024, 512, 256, 1] | 25 | [16, 32, 128] | ~X |

---

## TABLICA 3: Metode Prijenosnog Učenja

| Metoda | Zamrznuto | Trenirano | Learning Rate |
|--------|-----------|-----------|---------------|
| Baseline | — | Sve | 1e-4 |
| Freeze GNN | GNN slojevi | Conv1D, Dense | 1e-4 |
| Freeze Readout | Dense, Flatten | GNN slojevi | 1e-5 |
| Freeze All | Sve + novi izlaz | Novi Dense(1) | 1e-4 |
| Gradual Unfreezing | Postupno | Faze 1→2→3 | 1e-3→1e-4→1e-5 |

---

## TABLICA 4: Rezultati - Mala Molekula → Peptid (ROC-AUC)

| Veličina modela | Baseline | Freeze GNN | Freeze Readout | Freeze All | Gradual Unfreeze | Najbolji Δ |
|-----------------|----------|------------|----------------|------------|------------------|------------|
| Standard | ± | ± | ± | ± | ± | |
| Large Layers | ± | ± | ± | ± | ± | |
| Inflated | ± | ± | ± | ± | ± | |
| Extra Inflated | ± | ± | ± | ± | ± | |

**Bold** = najbolja metoda za tu veličinu. * p<0.05 vs baseline.

---

## TABLICA 5: Rezultati - Peptid → Mala Molekula (ROC-AUC)

| Veličina modela | Baseline | Freeze GNN | Freeze Readout | Freeze All | Gradual Unfreeze | Najbolji Δ |
|-----------------|----------|------------|----------------|------------|------------------|------------|
| Standard | ± | ± | ± | ± | ± | |
| Large Layers | ± | ± | ± | ± | ± | |
| Inflated | ± | ± | ± | ± | ± | |
| Extra Inflated | ± | ± | ± | ± | ± | |

---

## TABLICA 6: Rezultati - MCC (oba smjera)

| Veličina | SMT→P Baseline | SMT→P Najbolji | Δ | P→SMT Baseline | P→SMT Najbolji | Δ |
|----------|----------------|----------------|---|----------------|----------------|---|
| Standard | ± | ± | | ± | ± | |
| Large Layers | ± | ± | | ± | ± | |
| Inflated | ± | ± | | ± | ± | |
| Extra Inflated | ± | ± | | ± | ± | |

---

## TABLICA 7: Sažetak - Učinkovitost Transfera po Veličini Modela

| Veličina | SMT→P: Najbolja metoda | SMT→P: Δ AUC | SMT→P: p-value | P→SMT: Najbolja metoda | P→SMT: Δ AUC | P→SMT: p-value |
|----------|------------------------|--------------|----------------|------------------------|--------------|----------------|
| Standard | | | | | | |
| Large Layers | | | | | | |
| Inflated | | | | | | |
| Extra Inflated | | | | | | |

**Zaključak:** Transfer je najučinkovitiji kod [X] veličine modela.

---

## TABLICA 8: Statistička Značajnost (Friedman Test)

| Veličina | Smjer | χ² | p-value | Značajno? |
|----------|-------|-----|---------|-----------|
| Standard | SMT→P | | | |
| Standard | P→SMT | | | |
| Large Layers | SMT→P | | | |
| Large Layers | P→SMT | | | |
| Inflated | SMT→P | | | |
| Inflated | P→SMT | | | |
| Extra Inflated | SMT→P | | | |
| Extra Inflated | P→SMT | | | |

---

## KOMPAKTNA VERZIJA ZA KRATKI RAD

### Tablica A: Glavni Rezultati (ROC-AUC) - Sve veličine, oba smjera

| Veličina | **SMT → Peptid** | | | **Peptid → SMT** | | |
|----------|------------------|-------|-----|------------------|-------|-----|
| | Baseline | Najbolji TL | Δ | Baseline | Najbolji TL | Δ |
| Standard | | | | | | |
| Large | | | | | | |
| Inflated | | | | | | |
| Extra Inf. | | | | | | |

---

### Tablica B: Koja Metoda Pobjeđuje?

| Veličina | SMT→P Pobjednik | P→SMT Pobjednik |
|----------|-----------------|-----------------|
| Standard | | |
| Large Layers | | |
| Inflated | | |
| Extra Inflated | | |

**Obrazac:** [Postoji li konzistentna najbolja metoda preko svih veličina?]

---

## PREPORUKA ZA RAD

**Minimalno potrebno (6 stranica):**
- Tablica 1+2+3 spojiti u "Experimental Setup" paragraf
- **Tablica 4** ili **Tablica 5** (odaberi glavni smjer) - puni rezultati
- **Tablica 7** - sažetak oba smjera
- Ostatak u Supplementary Materials

**Priča:**
1. Opisati eksperiment (T1-3 kao tekst)
2. Pokazati rezultate po veličini modela (T4 ili T5)
3. Zaključiti o optimalnoj veličini i metodi (T7)
