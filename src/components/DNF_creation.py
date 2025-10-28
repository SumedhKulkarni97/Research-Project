"""
dnf_utils.py

This script contains utility functions for creating, evaluating,
and manipulating Disjunctive Normal Form (DNF) Boolean functions.
"""

from tqdm import tqdm

def create_initial_dnf(positive_examples):
    """
    Creates a 'rote-learner' DNF from all positive examples.
    Each term in the DNF is a frozenset of literals. A literal is an integer:
    - Positive integer `i` represents the variable x_i.
    - Negative integer `-i` represents the negated variable NOT(x_i).
    We use 1-based indexing for variables (e.g., pixel 0 is variable 1).
    """
    dnf = []
    print("\nCreating initial DNF from positive examples...")
    for example in tqdm(positive_examples, desc="Building DNF"):
        minterm = set()
        for i, val in enumerate(example):
            # Pixel i corresponds to variable i+1
            if val == 1:
                minterm.add(i + 1)  # Add literal x_{i+1}
            else:
                minterm.add(-(i + 1)) # Add literal NOT(x_{i+1})
        dnf.append(frozenset(minterm))
    return dnf

def get_dnf_length(dnf):
    """Calculates the total number of literals in a DNF."""
    return sum(len(term) for term in dnf)

def evaluate_term(term, example):
    """Checks if a binarized example satisfies a single term (AND-clause)."""
    for literal in term:
        var_index = abs(literal) - 1
        is_negated = literal < 0
        
        # If literal is x_i, example[i-1] must be 1
        if not is_negated and example[var_index] == 0:
            return False
        # If literal is NOT(x_i), example[i-1] must be 0
        if is_negated and example[var_index] == 1:
            return False
    return True

def evaluate_dnf(dnf, example):
    """Checks if a binarized example satisfies the full DNF (OR-of-ANDs)."""
    for term in dnf:
        if evaluate_term(term, example):
            return True
    return False

