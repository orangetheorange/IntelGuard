import numpy as np
import pandas as pd
import os
from pathlib import Path

script_path = Path(__file__).resolve()
parent_dir = script_path.parent
os.chdir(parent_dir)


def generate_interaction_based_dataset(num_samples=10000, random_state=42):
    np.random.seed(random_state)

    file_size = np.random.lognormal(mean=15, sigma=2, size=num_samples)
    file_size = np.clip(file_size, 1_000, 5_000_000_000)
    file_size = np.round(file_size).astype(np.int64)

    mean_sections = np.clip(np.log10(file_size) - 4, 1, 12)
    num_sections = np.random.poisson(mean_sections)
    num_sections = np.clip(num_sections, 1, 15)

    base_imports = np.random.normal(loc=20, scale=10, size=num_samples)
    import_scale = (file_size / 1e9) * 300
    num_imports = base_imports + import_scale + num_sections * 5
    num_imports = np.clip(num_imports, 0, 450).round().astype(int)

    packed = np.random.binomial(1, 0.2, size=num_samples)
    entropy = np.random.normal(loc=4.5, scale=1.0, size=num_samples) * (1 - packed) + \
              np.random.normal(loc=7.0, scale=0.5, size=num_samples) * packed
    entropy = np.clip(entropy, 0, 8)

    normal_ep = (file_size * np.random.uniform(0.05, 0.2, size=num_samples)).astype(int)
    suspicious_ep = (file_size * np.random.uniform(0.8, 0.95, size=num_samples)).astype(int)
    entry_point = np.where(packed == 1, suspicious_ep, normal_ep)
    entry_point = np.clip(entry_point, 1000, file_size - 1)

    # Calculate ratios & interactions
    size_per_section = file_size / np.maximum(num_sections, 1)
    imports_per_section = num_imports / np.maximum(num_sections, 1)
    entropy_per_1MB = entropy / (file_size / 1e6 + 1)  # +1 to avoid div zero
    ep_ratio = entry_point / file_size

    # Suspiciousness score with weights & interactions
    score = (
            2.0 * (file_size < 20_000).astype(int) +  # tiny file suspicious
            2.5 * (file_size > 1_000_000_000).astype(int) +  # huge file suspicious
            3.0 * (entropy > 6.5).astype(int) +  # high entropy suspicious
            3.0 * (ep_ratio > 0.75).astype(int) +  # late entry point suspicious
            1.5 * (num_sections < 2).astype(int) +  # very few sections suspicious
            1.5 * (num_sections > 10).astype(int) +  # very many sections suspicious
            2.0 * (num_imports > 300).astype(int) +  # many imports suspicious
            2.0 * packed +  # packed files suspicious

            # Interaction heuristics:
            4.0 * ((size_per_section > 200_000_000) & (num_sections < 5)).astype(int) +  # huge sections + few sections
            3.0 * ((imports_per_section > 30) & (num_sections < 5)).astype(
        int) +  # many imports packed into few sections
            2.5 * (entropy_per_1MB > 1.5).astype(int)  # very high entropy per MB suspicious
    )

    # Normalize to 0-100 and add noise
    suspiciousness = (score / score.max()) * 100
    suspiciousness = suspiciousness.round().astype(int)

    noise_mask = np.random.rand(num_samples) < 0.02
    suspiciousness[noise_mask] = np.random.randint(0, 101, size=noise_mask.sum())

    df = pd.DataFrame({
        'File Size (bytes)': file_size,
        'Number of Sections': num_sections,
        'Number of Imports': num_imports,
        'Entropy': entropy.round(3),
        'Entry Point Offset (bytes)': entry_point,
        'Packed (bool)': packed,
        'Suspiciousness Score (0-100)': suspiciousness
    })

    return df


if __name__ == '__main__':
    dataset = generate_interaction_based_dataset(num_samples=10000, random_state=42)
    dataset.to_csv('malware_suspiciousness_dataset.csv', index=False)
