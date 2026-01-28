# Fix for random.shuffle compatibility issue in pysc2
import random
import sys

# Monkey patch the random.shuffle function to handle the old signature
original_shuffle = random.shuffle

def patched_shuffle(sequence, random_func=None):
    """Patched version of random.shuffle that handles both old and new signatures"""
    if random_func is not None:
        # Old signature: shuffle(sequence, random_func)
        # In newer Python versions, random.shuffle only takes one argument
        # So we need to implement the shuffle ourselves using the random_func
        for i in range(len(sequence)-1, 0, -1):
            j = int(random_func() * (i+1))
            sequence[i], sequence[j] = sequence[j], sequence[i]
    else:
        # New signature: shuffle(sequence)
        original_shuffle(sequence)

# Apply the patch
random.shuffle = patched_shuffle

print("Applied random.shuffle compatibility patch")
