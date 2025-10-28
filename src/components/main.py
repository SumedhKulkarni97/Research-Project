"""
main.py

This is the main script to run the DNF learning and simplification experiment.
It orchestrates the data loading, initial DNF creation, and the local search
process. It then prints the final results.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
import data_ingestion
import DNF_creation
import local_search

def create_priority_mask_for_digit(digit_str, width=28, height=28):
    """
    Creates a simple heuristic mask for a given digit.
    This should be adapted based on the visual structure of the digit.
    """
    mask = np.ones(width * height) # Base priority of 1 for all pixels
    
    # --- Example Heuristic for Digit '7' ---
    if digit_str == '7':
        # High priority for the top horizontal bar
        for col in range(5, 20):
            for row in range(4, 8):
                mask[row * width + col] = 100
        # High priority for the diagonal bar
        for i in range(5, 23):
             # A simple diagonal line
            col = 20 - int(i * 0.7)
            row = i
            if 0 <= col < width and 0 <= row < height:
                mask[row * width + col] = 100
    
    # --- Example Heuristic for Digit '1' ---
    elif digit_str == '1':
        center_cols = [12, 13, 14, 15] 
        for row in range(4, 24):
            for col in center_cols:
                mask[row * width + col] = 100 # High priority
    
    # Add more heuristics for other digits if needed
    
    print(f"Priority mask for digit '{digit_str}' created.")
    return mask

def main():
    """Main function to run the entire experiment."""
    
    # --- 1. Configuration ---
    TARGET_DIGIT = '7'
    N_POSITIVE_SAMPLES = 50   # Number of '7's to learn from
    N_NEGATIVE_SAMPLES = 200  # Number of other digits for validation

    # --- Simulated Annealing Hyperparameters ---
    LAMBDA = 0.001          # Balances error vs. length. Higher = more simplification.
    INITIAL_TEMP = 1.0        # Higher temp allows more exploration.
    COOLING_RATE = 0.9995   # Rate at which temperature decreases. Closer to 1 is slower.
    NUM_ITERATIONS = 20000    # Total steps for the algorithm.
    
    # --- 2. Data Loading ---
    E_pos, E_neg, n_features = data_ingestion.load_and_prepare_data(
        TARGET_DIGIT, N_POSITIVE_SAMPLES, N_NEGATIVE_SAMPLES
    )
    
    # --- 3. Initial DNF Creation ---
    initial_dnf = DNF_creation.create_initial_dnf(E_pos)
    initial_length = DNF_creation.get_dnf_length(initial_dnf)
    print(f"\nInitial DNF created.")
    print(f" - Number of terms: {len(initial_dnf)}")
    print(f" - Total literals (length): {initial_length}")

    # --- 4. Heuristic Mask Creation ---
    priority_mask = create_priority_mask_for_digit(TARGET_DIGIT)
    
    # Optional: Visualize the mask
    plt.imshow(priority_mask.reshape(28, 28), cmap='hot')
    plt.title(f"Priority Mask for Digit '{TARGET_DIGIT}'")
    plt.show()

    # --- 5. Run Simplification ---
    final_dnf = local_search.simulated_annealing_with_heuristic(
        initial_dnf, E_pos, E_neg,
        lambda_val=LAMBDA,
        initial_temp=INITIAL_TEMP,
        cooling_rate=COOLING_RATE,
        num_iterations=NUM_ITERATIONS,
        priority_mask=priority_mask
    )
    
    # --- 6. Show Results ---
    final_cost, final_error, final_length = local_search.calculate_cost(
        final_dnf, E_pos, E_neg, LAMBDA
    )
    final_accuracy = 1.0 - final_error

    print("\n==================== RESULTS ====================")
    print(f"Initial DNF Length: {initial_length} literals")
    print(f"Final DNF Length:   {final_length} literals")
    print("-" * 50)
    if initial_length > 0:
        reduction = (initial_length - final_length) / initial_length * 100
        print(f"Length Reduction: {reduction:.2f}%")
    print(f"Final DNF Accuracy on Subset: {final_accuracy:.2%}")
    print(f"Final Cost: {final_cost:.4f}")
    print("===============================================")

if __name__ == '__main__':
    main()
