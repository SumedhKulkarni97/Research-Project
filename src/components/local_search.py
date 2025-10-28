"""
local_search.py

This script contains the core local search algorithms for simplifying a DNF.
It includes the cost function and the Simulated Annealing implementation.
"""

import math
import random
from tqdm import tqdm
from DNF_creation import evaluate_dnf, get_dnf_length

def calculate_cost(dnf, E_pos, E_neg, lambda_val):
    """
    Calculates the cost of a DNF based on its error rate and length.
    Cost = ErrorRate + lambda * Length.
    """
    errors = 0
    # False negatives: positive examples the DNF fails to classify as positive
    for ex in E_pos:
        if not evaluate_dnf(dnf, ex):
            errors += 1
    # False positives: negative examples the DNF wrongly classifies as positive
    for ex in E_neg:
        if evaluate_dnf(dnf, ex):
            errors += 1
    
    total_examples = len(E_pos) + len(E_neg)
    error_rate = errors / total_examples if total_examples > 0 else 0
    length = get_dnf_length(dnf)
    
    cost = error_rate + lambda_val * length
    return cost, error_rate, length

def simulated_annealing_with_heuristic(initial_dnf, E_pos, E_neg, lambda_val, 
                                       initial_temp, cooling_rate, num_iterations, 
                                       priority_mask):
    """
    Simplifies a DNF using simulated annealing, guided by a heuristic mask.
    The mask informs which literals are less important and can be removed first.
    """
    current_dnf = list(initial_dnf)
    current_cost, _, _ = calculate_cost(current_dnf, E_pos, E_neg, lambda_val)
    
    # Keep track of the best solution found so far
    best_dnf = current_dnf
    best_cost = current_cost
    
    current_temp = initial_temp
    max_priority = max(priority_mask) if len(priority_mask) > 0 else 1.0

    print("\nStarting Simulated Annealing with Heuristic Guidance...")
    print(f"Initial Cost: {current_cost:.4f}, Initial Temp: {initial_temp:.4f}")

    for _ in tqdm(range(num_iterations), desc="Simulated Annealing"):
        # --- 1. Generate a biased random neighbor ---
        non_empty_terms_indices = [idx for idx, term in enumerate(current_dnf) if len(term) > 0]
        if not non_empty_terms_indices:
            print("All terms are empty. Stopping.")
            break
            
        term_to_modify_idx = random.choice(non_empty_terms_indices)
        term_to_modify = current_dnf[term_to_modify_idx]
        
        # --- Create weighted choice for literal removal ---
        literals = list(term_to_modify)
        removal_weights = []
        for lit in literals:
            pixel_index = abs(lit) - 1
            priority = priority_mask[pixel_index]
            # Inverse relationship: low priority -> high weight for removal
            weight = max_priority - priority + 1 
            removal_weights.append(weight)
        
        literal_to_remove = random.choices(literals, weights=removal_weights, k=1)[0]
        
        # --- Create the neighbor DNF ---
        neighbor_dnf = list(current_dnf)
        new_term = set(term_to_modify)
        new_term.remove(literal_to_remove)
        neighbor_dnf[term_to_modify_idx] = frozenset(new_term)
        
        # --- 2. Evaluate and decide ---
        neighbor_cost, _, _ = calculate_cost(neighbor_dnf, E_pos, E_neg, lambda_val)
        cost_delta = neighbor_cost - current_cost
        
        if cost_delta < 0: # Better solution
            current_dnf = neighbor_dnf
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_dnf = current_dnf
                best_cost = current_cost
        else: # Potentially accept a worse solution
            acceptance_probability = math.exp(-cost_delta / current_temp)
            if random.random() < acceptance_probability:
                current_dnf = neighbor_dnf
                current_cost = neighbor_cost

        # --- 3. Cool the temperature ---
        current_temp *= cooling_rate

    print(f"\nFinished. Best cost found: {best_cost:.4f}")
    return best_dnf
