"""
main.py

This is the main script to run the DNF learning and simplification experiment.
It orchestrates the data loading, initial DNF creation, and runs TWO local
search algorithms (Hill Climbing and Simulated Annealing) to compare them.
"""

import numpy as np
import matplotlib.pyplot as plt
import time # Import the time module

# Import our custom modules using their new names
import data_ingestion
import DNF_creation
import local_search # This now contains both algorithms

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
    
    print(f"Priority mask for digit '{digit_str}' created.")
    return mask

# --- MODIFIED VISUALIZATION FUNCTION ---
def visualize_dnf_heatmap(dnf, n_features, title, width=28, height=28):
    """
    Visualizes the "learned concept" of the DNF as a pixel heatmap.
    
    - Positive values (Blue) = Pixels the DNF requires to be ON.
    - Negative values (Red)  = Pixels the DNF requires to be OFF.
    - Zero values (White)    = "Don't Care" pixels (simplified away).
    """
    heatmap = np.zeros(n_features)
    
    # "Vote" for each pixel's importance
    for term in dnf:
        for literal in term:
            pixel_index = abs(literal) - 1
            if literal > 0:
                heatmap[pixel_index] += 1 # "Must be ON"
            else:
                heatmap[pixel_index] -= 1 # "Must be OFF"
                
    plt.figure(figsize=(7, 6))
    plt.imshow(heatmap.reshape((height, width)), cmap='coolwarm')
    plt.colorbar(label="Pixel Importance (Blue=ON, Red=OFF)")
    # Use the title passed as an argument
    plt.title(title, fontsize=16)
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    plt.show()


def main():
    """Main function to run the entire experiment."""
    
    # --- 1. Configuration ---
    TARGET_DIGIT = '7'
    N_POSITIVE_SAMPLES = 50   # Number of '7's to learn from
    N_NEGATIVE_SAMPLES = 200  # Number of other digits for validation

    # --- Simulated Annealing Hyperparameters ---
    LAMBDA = 0.01          # Balances error vs. length.
    INITIAL_TEMP = 1.0        # Higher temp allows more exploration.
    COOLING_RATE = 0.9995   # Rate at which temperature decreases.
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
    
    plt.figure(figsize=(6, 5))
    plt.title(f"Heuristic Priority Mask for Digit '{TARGET_DIGIT}'")
    plt.show()

    # --- 5. RUN ALGORITHM 1: SIMPLE HILL CLIMBING ---
    print("\n--- Starting Simple Hill Climbing Search ---")
    start_time = time.time() # Start timer
    # We pass a copy (list()) so the original dnf isn't modified
    hc_final_dnf = local_search.hill_climbing_simplify(
        list(initial_dnf), 
        E_pos, E_neg,
        lambda_val=LAMBDA,
        max_iterations=100, # Will likely stop much earlier
        sample_size=2000
    )
    hc_time = time.time() - start_time # End timer
    hc_cost, hc_error, hc_length = local_search.calculate_cost(
        hc_final_dnf, E_pos, E_neg, LAMBDA
    )
    hc_accuracy = 1.0 - hc_error
    print(f"Hill Climbing Finished. Time: {hc_time:.2f}s, Cost: {hc_cost:.4f}, Length: {hc_length}, Accuracy: {hc_accuracy:.2%}")


    # --- 6. RUN ALGORITHM 2: SIMULATED ANNEALING ---
    print("\n--- Starting Simulated Annealing Search ---")
    start_time = time.time() # Start timer
    sa_final_dnf = local_search.simulated_annealing_with_heuristic(
        list(initial_dnf), # Pass a fresh copy
        E_pos, E_neg,
        lambda_val=LAMBDA,
        initial_temp=INITIAL_TEMP,
        cooling_rate=COOLING_RATE,
        num_iterations=NUM_ITERATIONS,
        priority_mask=priority_mask
    )
    sa_time = time.time() - start_time # End timer
    sa_cost, sa_error, sa_length = local_search.calculate_cost(
        sa_final_dnf, E_pos, E_neg, LAMBDA
    )
    sa_accuracy = 1.0 - sa_error
    print(f"Simulated Annealing Finished. Time: {sa_time:.2f}s, Cost: {sa_cost:.4f}, Length: {sa_length}, Accuracy: {sa_accuracy:.2%}")

    
    # --- 7. Show Combined Results ---
    print("\n==================== FINAL COMPARISON ====================")
    print(f"Initial DNF:     Length = {initial_length}, Accuracy = 100.00%, Time = N/A")
    print(f"Hill Climbing:   Length = {hc_length}, Accuracy = {hc_accuracy:.2%}, Time = {hc_time:.2f}s")
    print(f"Sim. Annealing:  Length = {sa_length}, Accuracy = {sa_accuracy:.2%}, Time = {sa_time:.2f}s")
    print("==========================================================")

    # --- 8. VISUALIZE THE RESULTS ---
    
    # Plot 1: Bar chart for length reduction
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Initial DNF", "Hill Climbing", "Simulated Annealing"],
        [initial_length, hc_length, sa_length],
        color=["#FF6B6B", "#FFD16B", "#6BFFB8"] # Red, Yellow, Green
    )
    plt.title("DNF Simplification Results: Length", fontsize=16)
    plt.ylabel("Total Number of Literals")
    plt.text(0, initial_length + 1000, f"{initial_length}", ha='center', fontsize=12)
    plt.text(1, hc_length + 1000, f"{hc_length}", ha='center', fontsize=12)
    plt.text(2, sa_length + 1000, f"{sa_length}", ha='center', fontsize=12)
    plt.show()

    # Plot 2: Bar chart for Time
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Hill Climbing", "Simulated Annealing"],
        [hc_time, sa_time],
        color=["#FFD16B", "#6BFFB8"] # Yellow, Green
    )
    plt.title("Algorithm Runtimes", fontsize=16)
    plt.ylabel("Time (seconds)")
    plt.text(0, hc_time + (sa_time * 0.05), f"{hc_time:.2f}s", ha='center', fontsize=12)
    plt.text(1, sa_time + (sa_time * 0.05), f"{sa_time:.2f}s", ha='center', fontsize=12)
    plt.show()

    # Plot 3: Heatmap for Hill Climbing (was Plot 2)
    visualize_dnf_heatmap(
        hc_final_dnf, n_features, 
        title="Learned Concept (Hill Climbing)"
    )
    
    # Plot 4: Heatmap for Simulated Annealing (was Plot 3)
    visualize_dnf_heatmap(
        sa_final_dnf, n_features, 
        title="Learned Concept (Simulated Annealing)"
    )

if __name__ == '__main__':
    main()

