import numpy as np
import matplotlib.pyplot as plt

class LaserCA:
    """
    A Cellular Automata model for simulating laser dynamics based on the paper
    "Application of Cellular Automata Algorithms to the Parallel Simulation of Laser Dynamics".
    """
    def __init__(self, L, tau_c, tau_a, lambda_p, M=10, stim_threshold=1, noise_photons=1):
        """
        Initializes the Laser CA simulation.

        Args:
            L (int): The side length of the square cellular lattice. 
            tau_c (int): The lifetime of photons in time steps. [cite: 421]
            tau_a (int): The lifetime of excited electrons in time steps. [cite: 425]
            lambda_p (float): The pumping probability (0 to 1). 
            M (int, optional): Maximum number of photons per cell. Defaults to 10. 
            stim_threshold (int, optional): Threshold for stimulated emission. Defaults to 1. 
            noise_photons (int, optional): Number of noise photons to add each step. 
        """
        # --- System Parameters ---
        self.L = L
        self.tau_c = tau_c
        self.tau_a = tau_a
        self.lambda_p = lambda_p
        self.M = M
        self.stim_threshold = stim_threshold
        self.noise_photons = noise_photons
        
        # --- CA State Variables ---
        # a_ij(t): State of the electron (0=ground, 1=excited) 
        self.electron_states = np.zeros((L, L), dtype=np.int8)
        
        # tla_ij: Remaining lifetime of an excited electron [cite: 426]
        self.electron_lifetimes = np.zeros((L, L), dtype=np.int32)
        
        # c_ij(t): Number of photons in the cell 
        self.photon_counts = np.zeros((L, L), dtype=np.int32)
        
        # tlc_ijk: Remaining lifetime of each photon 'k' in cell {i,j} [cite: 422]
        # A 3D array: (L, L, M) where M is the max number of photons per cell
        self.photon_lifetimes = np.zeros((L, L, M), dtype=np.int32)

    def _get_moore_neighbors_sum(self, i, j):
        """
        Calculates the sum of photons in the Moore neighborhood of cell (i, j)
        using periodic boundary conditions. [cite: 394, 479]
        """
        total = 0
        # Loop over the 3x3 neighborhood
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                # Skip the center cell itself
                if di == 0 and dj == 0:
                    continue
                
                # Apply periodic boundary conditions using modulo operator
                ni, nj = (i + di) % self.L, (j + dj) % self.L
                total += self.photon_counts[ni, nj]
        return total

    def _apply_stimulated_emission(self):
        """
        Applies the stimulated emission rule. This must be done in a way that
        simulates a parallel update for all cells. 
        """
        # A temporary array to store changes, as per the paper [cite: 419]
        new_photons_this_step = np.zeros((self.L, self.L), dtype=np.int32)

        for i in range(self.L):
            for j in range(self.L):
                # Condition 1: Electron is in the excited state 
                if self.electron_states[i, j] == 1:
                    # Condition 2: Photon count in neighborhood is above threshold 
                    neighbor_photons = self._get_moore_neighbors_sum(i, j)
                    if neighbor_photons >= self.stim_threshold:
                        # Find the first available photon "slot" in this cell
                        k = np.where(self.photon_lifetimes[i, j, :] == 0)[0]
                        if len(k) > 0:
                            slot = k[0]
                            # A new photon is emitted 
                            self.photon_lifetimes[i, j, slot] = self.tau_c + 1 # +1 to account for decay in the same step
                            new_photons_this_step[i, j] += 1
                            # The electron decays to the ground level 
                            self.electron_states[i, j] = 0
                            self.electron_lifetimes[i, j] = 0
        
        # Update the main photon counts array
        self.photon_counts += new_photons_this_step

    def _apply_decays_and_pumping(self):
        """
        Applies the photon decay, electron decay, and pumping rules. [cite: 424]
        """
        # --- Photon Decay Rule --- [cite: 421]
        decaying_photons_mask = (self.photon_lifetimes > 0)
        self.photon_lifetimes[decaying_photons_mask] -= 1
        # Count how many photons just decayed in each cell
        newly_decayed_photons = np.sum((self.photon_lifetimes == 0) & decaying_photons_mask, axis=2)
        self.photon_counts -= newly_decayed_photons
        
        # --- Electron Decay Rule --- [cite: 425]
        decaying_electrons_mask = (self.electron_lifetimes > 0)
        self.electron_lifetimes[decaying_electrons_mask] -= 1
        # If lifetime reaches zero, electron decays
        self.electron_states[(self.electron_lifetimes == 0) & decaying_electrons_mask] = 0

        # --- Pumping Rule --- 
        ground_state_mask = (self.electron_states == 0)
        # Generate random numbers only for electrons in the ground state
        rand_values = np.random.rand(self.L, self.L)
        # Find which of those electrons get pumped
        pumped_mask = (rand_values < self.lambda_p) & ground_state_mask
        
        self.electron_states[pumped_mask] = 1
        self.electron_lifetimes[pumped_mask] = self.tau_a

    def _apply_noise(self):
        """
        Adds a small number of noise photons to random cells. 
        """
        for _ in range(self.noise_photons):
            i, j = np.random.randint(0, self.L, size=2)
            # Find an empty photon slot
            k = np.where(self.photon_lifetimes[i, j, :] == 0)[0]
            if len(k) > 0 and self.photon_counts[i, j] < self.M:
                slot = k[0]
                self.photon_lifetimes[i, j, slot] = self.tau_c
                self.photon_counts[i, j] += 1

    def run_simulation(self, max_time_steps):
        """
        Runs the main simulation loop.

        Args:
            max_time_steps (int): The number of time steps to simulate.

        Returns:
            tuple: A tuple containing lists of total photons and population inversion over time.
        """
        total_photons_history = []
        population_inversion_history = []
        
        for t in range(max_time_steps):
            # Apply rules in the order specified by Algorithm 19.1 [cite: 398]
            self._apply_stimulated_emission()
            self._apply_decays_and_pumping()
            self._apply_noise()

            # Calculate macroscopic magnitudes [cite: 482, 483, 484]
            n_t = np.sum(self.photon_counts)
            N_t = np.sum(self.electron_states)
            
            total_photons_history.append(n_t)
            population_inversion_history.append(N_t)
            
            if (t + 1) % 100 == 0:
                print(f"Time Step: {t+1}/{max_time_steps}, Photons: {n_t}, Population Inversion: {N_t}")
                
        return total_photons_history, population_inversion_history

if __name__ == '__main__':
    # --- Simulation Parameters ---
    # These parameters are chosen to demonstrate oscillatory behavior, similar to Fig 19.4 (right)
    LATTICE_SIZE = 50       # L x L grid
    SIMULATION_STEPS = 1000
    TAU_C = 8               # Photon lifetime
    TAU_A = 100             # Electron lifetime
    LAMBDA_P = 0.05         # Pumping probability
    MAX_PHOTONS_PER_CELL = 20
    NOISE_PHOTONS = 5

    # --- Initialize and Run Simulation ---
    laser_sim = LaserCA(
        L=LATTICE_SIZE,
        tau_c=TAU_C,
        tau_a=TAU_A,
        lambda_p=LAMBDA_P,
        M=MAX_PHOTONS_PER_CELL,
        noise_photons=NOISE_PHOTONS
    )
    
    print("Starting Laser CA Simulation...")
    photons, inversion = laser_sim.run_simulation(SIMULATION_STEPS)
    print("Simulation finished.")

    # --- Plotting Results ---
    time_steps = np.arange(SIMULATION_STEPS)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Time evolution of photons and population inversion
    ax1.plot(time_steps, inversion, label='Population Inversion ($N(t)$)', color='gray')
    ax1.plot(time_steps, photons, label='Laser Photons ($n(t)$)', color='black', linewidth=2)
    ax1.set_ylabel('Population')
    ax1.set_title('Laser Dynamics Simulation via Cellular Automata')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Phase space plot (Population Inversion vs. Laser Photons)
    ax2.plot(inversion, photons, color='black', marker='.', linestyle='', markersize=2)
    ax2.set_xlabel('Population Inversion ($N(t)$)')
    ax2.set_ylabel('Laser Photons ($n(t)$)')
    ax2.set_title('Phase Space Plot')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.show()
