"""
SMILES Feature Extraction Utilities

This module provides feature extraction functions for SMILES strings,
similar to DPC (Dipeptide Composition) for peptide sequences.

Main functions:
    - compute_smiles_ngram_composition: Extract n-gram composition features
    - get_all_smiles_ngrams: Build vocabulary of all n-grams from a dataset
"""

import numpy as np
from collections import Counter


def get_all_smiles_ngrams(smiles_list, n=2):
    """
    Ekstrahira sve moguće n-grame iz liste SMILES stringova.
    
    Args:
        smiles_list: Lista SMILES stringova
        n: Veličina n-grama (default 2 za bigrame, kao DPC)
    
    Returns:
        Sorted list svih jedinstvenih n-grama (vokabular)
    
    Example:
        >>> smiles = ["CCO", "CCN"]
        >>> get_all_smiles_ngrams(smiles, n=2)
        ['CC', 'CN', 'CO']
    """
    all_ngrams = set()
    for smiles in smiles_list:
        if smiles and len(str(smiles)) >= n:
            smiles_str = str(smiles)
            for i in range(len(smiles_str) - n + 1):
                ngram = smiles_str[i:i+n]
                all_ngrams.add(ngram)
    return sorted(list(all_ngrams))


def compute_smiles_ngram_composition(smiles, ngram_vocab, n=2):
    """
    Računa N-gram Composition feature vektor za SMILES string.
    
    Slično kao DPC za peptide sekvence:
      - Za SMILES duljine L: ukupan broj n-grama = L - n + 1
      - N-gram Composition(i) = count(ngram_i) / (L - n + 1)
    
    Args:
        smiles: SMILES string (može biti string ili None)
        ngram_vocab: Lista svih mogućih n-grama (vokabular) - mora biti sortirana
        n: Veličina n-grama (default 2, mora odgovarati ngram_vocab)
    
    Returns:
        Normalizirani vektor frekvencija n-grama (numpy array)
    
    Example:
        >>> vocab = ['CC', 'CO', 'OC']
        >>> compute_smiles_ngram_composition("CCO", vocab, n=2)
        array([0.5, 0.5, 0.0])  # "CC"=1, "CO"=1, "OC"=0, normalizirano s 2
    """
    if not smiles or len(str(smiles)) < n:
        return np.zeros(len(ngram_vocab))
    
    smiles_str = str(smiles)
    
    # Brojanje n-grama
    counts = np.zeros(len(ngram_vocab))
    ngram_to_index = {ng: i for i, ng in enumerate(ngram_vocab)}
    
    total_ngrams = len(smiles_str) - n + 1
    for i in range(total_ngrams):
        ngram = smiles_str[i:i+n]
        if ngram in ngram_to_index:
            counts[ngram_to_index[ngram]] += 1
    
    # Normalizacija
    if total_ngrams > 0:
        counts = counts / total_ngrams
    
    return counts


def compute_smiles_ngram_composition_batch(smiles_list, ngram_vocab, n=2):
    """
    Računa N-gram Composition feature vektore za listu SMILES stringova.
    
    Args:
        smiles_list: Lista SMILES stringova
        ngram_vocab: Lista svih mogućih n-grama (vokabular)
        n: Veličina n-grama (default 2)
    
    Returns:
        Numpy array shape (len(smiles_list), len(ngram_vocab))
    """
    return np.array([
        compute_smiles_ngram_composition(smiles, ngram_vocab, n=n)
        for smiles in smiles_list
    ])


# ========================== USPOREDBA S DPC ==========================

def compare_with_dpc():
    """
    Demonstracija kako SMILES n-gram composition radi slično kao DPC.
    
    DPC za peptide "ACDEF":
      - Dipeptidi: AC, CD, DE, EF
      - 400-dimenzionalni vektor (20×20 kombinacija)
    
    SMILES n-gram za "CCO":
      - Bigrami: CC, CO
      - N-dimenzionalni vektor (N = broj jedinstvenih bigrama u datasetu)
    """
    print("DPC (Dipeptide Composition) za peptide:")
    print("  Sekvenca: 'ACDEF'")
    print("  Dipeptidi: AC, CD, DE, EF")
    print("  Vektor: 400 dimenzija (20×20 kombinacija)")
    print()
    print("SMILES N-gram Composition:")
    print("  SMILES: 'CCO'")
    print("  Bigrami: CC, CO")
    print("  Vektor: N dimenzija (N = broj jedinstvenih bigrama u datasetu)")
    print()
    print("Oba pristupa:")
    print("  1. Broje n-grame u stringu")
    print("  2. Normaliziraju frekvencije")
    print("  3. Stvaraju fiksno-dimenzionalne vektore za ML modele")


if __name__ == "__main__":
    # Test primjer
    print("=" * 70)
    print("  SMILES N-gram Composition - Test")
    print("=" * 70)
    
    test_smiles = ["CCO", "CCN", "CCOO", "CC(=O)N"]
    print(f"\nTest SMILES: {test_smiles}")
    
    # Izgradi vokabular
    vocab = get_all_smiles_ngrams(test_smiles, n=2)
    print(f"\nBigram vokabular ({len(vocab)} jedinstvenih):")
    print(f"  {vocab[:20]}..." if len(vocab) > 20 else f"  {vocab}")
    
    # Računaj feature vektore
    print("\nFeature vektori:")
    for smiles in test_smiles:
        features = compute_smiles_ngram_composition(smiles, vocab, n=2)
        non_zero = np.sum(features > 0)
        print(f"  {smiles:10s}: {non_zero}/{len(vocab)} non-zero features")
    
    print("\n" + "=" * 70)
    compare_with_dpc()
