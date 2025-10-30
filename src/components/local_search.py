"""
local_search.py

Contains the core logic for the DNF simplification algorithms,
including Hill Climbing and Simulated Annealing.
"""

import math
import random
from tqdm import tqdm
import numpy as np

# --- DNF Evaluation Functions ---

def evaluate_term(term, example):
    """Checks if a binarized example satisfies a single term (a frozenset of literals)."""
    for literal in term:
        var_index = abs(literal) - 1
        is_negated = literal < 0
        
        # If literal is x_i, example[var_index] must be 1
        if not is_negated and example[var_index] == 0:
            return False
        # If literal is not(x_i), example[var_index] must be 0
        if is_negated and example[var_index] == 1:
            return False
    # If the loop finishes without returning False, all literals are satisfied
    return True

def evaluate_dnf(dnf, example):
    """Checks if a binarized example satisfies the DNF (a list of terms)."""
    for term in dnf:
        if evaluate_term(term, example):
            return True
    # If no term is satisfied, the DNF is False for this example
    return False

def calculate_cost(dnf, E_pos, E_neg, lambda_val):
    """Calculates the cost = ErrorRate + lambda * Length."""
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
    
    # DNF length is the sum of all literals in all terms
    length = sum(len(term) for term in dnf)
    
    cost = error_rate + (lambda_val * length)
    return cost, error_rate, length

# --- Algorithm 1: Hill Climbing ---

def hill_climbing_simplify(initial_dnf, E_pos, E_neg, lambda_val, max_iterations=100, sample_size=2000):
    """
    Simplifies a DNF using a basic hill-climbing local search.
    It samples neighbors rather than checking all of them.
    """
    current_dnf = list(initial_dnf) # Make a mutable copy
    
    # Calculate initial cost
    current_cost, current_error, current_length = calculate_cost(current_dnf, E_pos, E_neg, lambda_val)
    print(f"Starting Hill Climbing...")
    print(f"Initial Cost: {current_cost:.4f}, Error Rate: {current_error:.4f}, Length: {current_length}")
    
    for i in range(max_iterations):
        best_neighbor = None
        best_neighbor_cost = current_cost # Start with current cost as the bar to beat

        # --- Generate and Evaluate a *sample* of neighbors (by removing one literal) ---
        search_space = []
        for term_idx, term in enumerate(current_dnf):
            if len(term) == 0: continue # Skip empty terms
            for literal in term:
                search_space.append((term_idx, literal))
        
        if not search_space:
            print("No more literals to remove. Stopping.")
            break
            
        # Randomly sample neighbors to check
        random.shuffle(search_space)
        
        for term_idx, literal_to_remove in search_space[:sample_size]:
            
            neighbor_dnf = list(current_dnf)
            original_term = neighbor_dnf[term_idx]
            
            # Create the new, simpler term
            new_term = set(original_term)
            new_term.remove(literal_to_remove)
            
            # Replace the old term with the new one
            neighbor_dnf[term_idx] = frozenset(new_term)
            
            # Evaluate this neighbor
            cost, _, _ = calculate_cost(neighbor_dnf, E_pos, E_neg, lambda_val)
            
            if cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor = neighbor_dnf
        
        # --- Make a move (Greedy Decision) ---
        if best_neighbor: # If we found a better neighbor
            current_dnf = best_neighbor
            current_cost, current_error, current_length = calculate_cost(current_dnf, E_pos, E_neg, lambda_val)
            print(f"  > Iter {i+1}: Found improvement. New Cost: {current_cost:.4f}, Length: {current_length}")
        else:
            print(f"\nNo better neighbor found after {i+1} iterations. Reached a local minimum.")
            break
            
    return current_dnf

# --- Algorithm 2: Simulated Annealing ---

def simulated_annealing_with_heuristic(initial_dnf, E_pos, E_neg, lambda_val, 
                                       initial_temp, cooling_rate, num_iterations, 
                                       priority_mask):
    """Simplifies a DNF using simulated annealing guided by a heuristic mask."""
    
    current_dnf = list(initial_dnf)
    current_cost, _, _ = calculate_cost(current_dnf, E_pos, E_neg, lambda_val)
    
    # Keep track of the best solution found so far
    best_dnf = current_dnf
    best_cost = current_cost
    
    current_temp = initial_temp
    
    max_priority = np.max(priority_mask)
    if max_priority <= 1: max_priority = 2 # Avoid division by zero if mask is all 1s

    print("Starting Simulated Annealing with Heuristic Guidance...")
    print(f"Initial Cost: {current_cost:.4f}, Initial Temp: {initial_temp:.4f}")

    for i in tqdm(range(num_iterations), desc="Simulated Annealing"):
        # --- 1. Generate a biased random neighbor ---
        
        # Find terms that are not empty
        non_empty_terms_indices = [idx for idx, term in enumerate(current_dnf) if len(term) > 0]
        if not non_empty_terms_indices:
            print("All terms are empty. Stopping.")
            break
            
        term_to_modify_idx = random.choice(non_empty_terms_indices)
        term_to_modify = current_dnf[term_to_modify_idx]
        
        # --- Heuristic-guided literal selection ---
        literals = list(term_to_modify)
        
        # Calculate removal weights: low priority pixels should have high removal weight
        removal_weights = []
        for lit in literals:
            pixel_index = abs(lit) - 1
            priority = priority_mask[pixel_index]
            # Inverse relationship: low priority -> high weight
            # Adding +1 to ensure non-zero weight
            weight = (max_priority - priority) + 1 
            removal_weights.append(weight)
        
        # Select a literal to remove based on the calculated weights
        literal_to_remove = random.choices(literals, weights=removal_weights, k=1)[0]
        # --- End of modified section ---
        
        neighbor_dnf = list(current_dnf)
        new_term = set(term_to_modify)
        new_term.remove(literal_to_remove)
        neighbor_dnf[term_to_modify_idx] = frozenset(new_term)
        
        # --- 2. Evaluate the neighbor ---
        neighbor_cost, _, _ = calculate_cost(neighbor_dnf, E_pos, E_neg, lambda_val)
        
        # --- 3. Decide whether to move to the neighbor ---
        cost_delta = neighbor_cost - current_cost
        
        # If the new solution is better, always accept it
        if cost_delta < 0:
            current_dnf = neighbor_dnf
            current_cost = neighbor_cost
            # If this is the best solution we've seen, save it
            if current_cost < best_cost:
                best_dnf = current_dnf
                best_cost = current_cost
        # If the new solution is worse, accept it with a certain probability
        else:
            acceptance_probability = math.exp(-cost_delta / current_temp)
            if random.random() < acceptance_probability:
                current_dnf = neighbor_dnf
                current_cost = neighbor_cost

        # --- 4. Cool the temperature ---
        current_temp *= cooling_rate

    print(f"\nFinished. Best cost found: {best_cost:.4f}")
    return best_dnf

