# Online Detection of AI-Generated Images: A Cross-Generator Generalization Study

This repository contains the PyTorch implementation for a simulated online learning pipeline designed to detect AI-Generated images. The project investigates whether a Convolutional Neural Network (CNN) can generalize to unseen generative architectures by maintaining a cumulative historical training pool, extending the methodologies proposed by Epstein et al. (2023).

## 1. Directory Structure and Datasets
Because the datasets utilized in this project (CIFAKE and subsets of GenImage) consist of hundreds of thousands of high-resolution images, they are not included in this repository and cannot be automatically downloaded via script due to size and hosting constraints. To run this code, you must manually download the datasets and replicate the following directory structure:

```text
/content/
│
├── train/                        # CIFAKE (Stable Diffusion 1.4) Train Split
│   ├── FAKE/
│   └── REAL/
├── test/                         # CIFAKE (Stable Diffusion 1.4) Test Split
│   ├── FAKE/
│   └── REAL/
│
└── local_genimage/               # GenImage Validation Subsets
    ├── stable_diffusion_v_1_5/
    │   └── val/
    │       ├── nature/           # (Real Images)
    │       └── ai/               # (Fake Images)
    ├── Midjourney/
    │   └── val/
    └── glide/
        └── val/
```

*Note: For the GenImage datasets, only the val folders were extracted to manage storage constraints. The codebase automatically applies a deterministic torch.utils.data.random_split to these folders to safely isolate training updates from evaluation data without leakage.*

## 2. Dependencies
To run this project, ensure you have the following libraries installed:
* torch and torchvision (PyTorch framework)
* scikit-learn (For AuC metric calculations)
* matplotlib (For ROC curve visualizations)
* Python 3.8+

You can install the required packages via:
`pip install torch torchvision scikit-learn matplotlib`

## 3. Instructions to Run
1. Ensure your directory structure perfectly matches the outline in Section 1.
2. Execute the main pipeline script. If running in a Jupyter Notebook/Colab, execute the cell containing the pipeline.
3. **Resuming Training:** The script features a built-in Resume Controller. If execution is interrupted, change `start_step = 0` to the failed step index (e.g., `start_step = 1` for Stable Diffusion V1.5). The script will automatically load the .pth weights from the previous successful step and rebuild the historical training pool.
4. **Final Validation:** The loop will automatically halt before Phase B of the GLIDE dataset, as it is strictly reserved for final Phase A (Zero-Shot) evaluation.

## 4. Code Attribution
In accordance with course policies, the code in this repository is attributed as follows:

* **Adapted from Prior Code (Standard Boilerplate):**
    * The initialization of the resnet50 backbone and standard PyTorch image transformations were adapted from official PyTorch Transfer Learning documentation.
    * The standard PyTorch optimizer.zero_grad(), loss.backward(), and optimizer.step() loop logic within Phase B was adapted from standard PyTorch documentation.
* **Written by Me:**
    * The Online Learning Pipeline architecture (chronological_timeline iteration).
    * The Phase A, Phase B, and Phase C evaluation separation logic.
    * The Resume Controller mechanism for state recovery and dynamic pool rebuilding.
    * The deterministic validation splitting function utilizing fixed seeds to prevent dataset leakage across phases.
    * The dynamic evaluation loader that maps folder hierarchies to the active pipeline step.

## 5. LLM Acknowledgments
In the interest of academic transparency, I acknowledge the use of Google's Gemini as an assistive tool during the development of this project. Specifically, Gemini was utilized to aid in drafting the software implementation and troubleshooting pipeline logic (such as diagnosing catastrophic forgetting, addressing data leakage bugs in validation splits, and handling sklearn import constraints). Additionally, the model was employed during the preparation of the final manuscript and this README.md file for copy-editing purposes to enhance clarity and formatting. I have thoroughly reviewed, edited, and take full responsibility for all code and final content presented in this repository and the accompanying report.