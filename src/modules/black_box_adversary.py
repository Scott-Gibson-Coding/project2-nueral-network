# Approach: Based on the paper: https://arxiv.org/pdf/1712.04248
# DECISION-BASED ADVERSARIAL ATTACKS: RELIABLE ATTACKS AGAINST BLACK-BOX MACHINE LEARNING MODELS
# Wieland Brendel, Jonas Rauber & Matthias Bethge
#
# 1. Pick an image to use as reference. Denote this x. It should be classified correctly, f(x) = y. ~x_k will be the adversarially perturbed image at the k-th step of the attack.
# 2. Pick a point, can be random noise, which is already adversarial to x. f(~x_0) != y.
#   2a. For instance, random sampling on a uniform distribution of pixel values from [0, 1].
# 3. For up to k_max iterations, perform a random walk along the boundary between the adversarial and non-adversarial region.
#   3a*. As a cool idea it would be neat to save images along the walk. Show how we go from adversarial random noise to a reasonable replication of an existing data point.
#   3b. Each step is controlled by two hyper parameters. The size of the total perterbation "delta", and the reduced distance between the adversary and the original image after the step, "epsilon".
#
#

import numpy as np
from src.modules.dense_net import DenseNet
from src.modules.save_data import save_points

class BlackBoxAdversary():
    # Takes in a starting image which we wish to mock with a new image that is as close as possible to the original
    # (in the L2 norm) but misclassified.
    def __init__(self, net: DenseNet, x, y):
        self.net = net
        
        self.x = x
        self.y = y

    def _classify(self, x_k):
        """Returns true if x_k is classified as the same output class as the original."""
        y_k = self.net.predict(x_k)
        return np.argmax(y_k) == np.argmax(self.y)

    def _dist(self, x_k):
        """Returns the distance in the 2-norm between x and x_k."""
        return np.linalg.norm(self.x - x_k, ord=2)

    def generate(self, img_offs: int = None):
        """
        Main attacking method, generates a new image as close to the original "x" provided
        as possible while keeping it misclassified.
        """

        # Hyper-parameters
        max_steps = 1000
        delta = 1
        epsilon = 0.1

        closeness_threshold = 1e-5

        # Step 1. Generate an image of random noise that is misclassified.
        x_k = np.random.random(self.x.shape)
        while self._classify(x_k):
            x_k = np.random.random(self.x.shape)

        # Store image points to display later
        xk_points = []

        for step in range(max_steps):
            diff_v = self.x - x_k   # Vector x - x_k
            dist = self._dist(x_k)  # Distance between x, x_k in 2-norm
            print(f"Step {step}: {round(dist, 5)}")
            # Exit condition
            if dist < closeness_threshold:
                break

            # Save image, note ignore offsets if it would save over 50 images
            if img_offs and len(xk_points) < 50:
                if step % img_offs == 0:
                    xk_points.append(x_k)

            # Step 2. Sample a random step, and project it orthogonally onto a hpyer-sphere centered 
            # around x with radius ||x - x_k||.
            noise = np.random.randn(*x_k.shape)
            proj = np.dot(noise, diff_v) / (dist ** 2) * diff_v
            noise_orth = noise - proj

            # Normalize step by delta and current distance
            noise_orth = noise_orth / np.linalg.norm(noise_orth)
            orth_step = noise_orth * delta * dist

            # Move along dist sphere
            orth_candidate = x_k + orth_step

            # Re-project onto the sphere centered at x_original
            orth_direction = orth_candidate - self.x
            orth_candidate = self.x + dist * \
                (orth_direction / np.linalg.norm(orth_direction))

            # Step 3. Nudge the adversary a small distance (epsilon) towards x.
            forward_step = epsilon * dist * (self.x - orth_candidate) / dist
            x_candidate = orth_candidate + forward_step

            # Clip candidate to ensure it's in the output range
            x_candidate = np.clip(x_candidate, 0.0, 1.0)

            is_adversarial = not self._classify(x_candidate)

            # Accept or Reject candidate, and adjust step sizes accordingly
            if is_adversarial:
                x_k = x_candidate
                # Try to accelerate progress
                delta *= 1.05
                epsilon *= 1.05
            else:
                # Decrease step sizes, we stepped across the boundary into a non-adversarial zone
                delta *= 0.95
                epsilon *= 0.95

        # Save images if any are stored
        if xk_points:
            save_points(xk_points, "spook-step")
        return x_k
    