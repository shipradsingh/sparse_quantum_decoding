import unittest
import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
from src.belief_propagation_decoder import apply_freezing, apply_random_perturbations

class TestLDPCHeuristics(unittest.TestCase):

    def setUp(self):
        """
        Setup function to initialize parameters and generate LDPC matrices.
        """
        self.n = 15  # Length of the codeword
        self.d_v = 4  # Degree of variable nodes
        self.d_c = 5  # Degree of check nodes
        self.signal_to_noise_ratio = 10  # SNR in decibels
        self.threshold = 0.95  # Threshold for freezing
        self.epsilon = 0.05  # Epsilon for random perturbations

        # Generate sparse LDPC matrices
        self.H, self.G = make_ldpc(self.n, self.d_v, self.d_c, systematic=True, sparse=True)

    def test_apply_freezing(self):
        """
        Test the apply_freezing function to ensure it properly freezes high-confidence LLRs.
        """
        Lc = np.array([0.5, 2.0, -1.0, 0.1, -3.0])  # Example LLRs
        frozen_Lc = apply_freezing(Lc.copy(), threshold=self.threshold)

        # Check if high-confidence nodes are frozen
        for i, val in enumerate(Lc):
            if abs(val) > np.log(self.threshold / (1 - self.threshold)):
                self.assertTrue(np.isinf(frozen_Lc[i]))
            else:
                self.assertEqual(Lc[i], frozen_Lc[i])

    def test_apply_random_perturbations(self):
        """
        Test the apply_random_perturbations function to ensure it introduces stochastic variations correctly.
        """
        Lc = np.zeros(10)  # Example LLRs
        perturbed_Lc = apply_random_perturbations(Lc.copy(), epsilon=self.epsilon)

        # Check if all perturbations are within the epsilon range
        perturbations = perturbed_Lc - Lc
        self.assertTrue(np.all(perturbations >= -self.epsilon))
        self.assertTrue(np.all(perturbations <= self.epsilon))

    def test_encode_and_decode(self):
        """
        Test encoding and decoding with LDPC matrices.
        """
        k = self.G.shape[1]
        original_message = np.random.randint(0, 2, size=k)
        encoded_message = encode(self.G, original_message, self.signal_to_noise_ratio)

        # Add noise
        noise = np.random.normal(0, 10**(-self.signal_to_noise_ratio / 20), encoded_message.shape)
        noisy_message = encoded_message + noise

        # Decode message
        decoded_codeword = decode(self.H, noisy_message, self.signal_to_noise_ratio, maxiter=100)
        decoded_message = get_message(self.G, decoded_codeword)

        # Ensure the decoded message matches the original
        self.assertTrue(np.array_equal(original_message, decoded_message))

    def test_encode_and_decode_with_heuristics(self):
        """
        Test encoding and decoding with heuristics applied.
        """
        k = self.G.shape[1]
        original_message = np.random.randint(0, 2, size=k)
        encoded_message = encode(self.G, original_message, self.signal_to_noise_ratio)

        # Add noise
        noise = np.random.normal(0, 10**(-self.signal_to_noise_ratio / 20), encoded_message.shape)
        noisy_message = encoded_message + noise

        # Apply heuristics
        Lc = 2 * noisy_message / (10**(-self.signal_to_noise_ratio / 10))
        Lc = apply_freezing(Lc, threshold=self.threshold)
        Lc = apply_random_perturbations(Lc, epsilon=self.epsilon)

        # Decode message
        decoded_codeword = decode(self.H, noisy_message, self.signal_to_noise_ratio, maxiter=100)
        decoded_message = get_message(self.G, decoded_codeword)

        # Ensure the decoded message matches the original
        self.assertTrue(np.array_equal(original_message, decoded_message))

if __name__ == "__main__":
    unittest.main()
