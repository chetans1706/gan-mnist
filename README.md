# Adversarial Image Synthesis on MNIST (PyTorch)

A Generative Adversarial Network (GAN) implemented from scratch in PyTorch to synthesize $28\times 28$ grayscale handwritten digits trained on the MNIST benchmark.

---

## Model Architecture

### Generator
Maps a latent noise vector $z \in \mathbb{R}^{100} \sim \mathcal{N}(0, I)$ to pixel space ($1 \times 28 \times 28$):
* **Linear(100 $\rightarrow$ 128)** $\rightarrow$ ReLU $\rightarrow$ BatchNorm1d(128)
* **Linear(128 $\rightarrow$ 256)** $\rightarrow$ ReLU $\rightarrow$ BatchNorm1d(256)
* **Linear(256 $\rightarrow$ 512)** $\rightarrow$ ReLU $\rightarrow$ BatchNorm1d(512)
* **Linear(512 $\rightarrow$ 784)** $\rightarrow$ Tanh (scaled to $[-1, 1]$)

### Discriminator
Binary classifier scoring realness vs. generated validity:
* **Linear(784 $\rightarrow$ 512)** $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.3)
* **Linear(512 $\rightarrow$ 256)** $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.3)
* **Linear(256 $\rightarrow$ 128)** $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.3)
* **Linear(128 $\rightarrow$ 1)** $\rightarrow$ Sigmoid

---

## Hyperparameters & Training

| Parameter | Value |
|---|---|
| **Loss Function** | Binary Cross Entropy (`BCELoss`) |
| **Optimizer** | Adam ($\alpha = 0.0002$, $\beta_1 = 0.5$, $\beta_2 = 0.999$) |
| **Batch Size** | 128 (469 batches/epoch) |
| **Epochs** | 50 |
| **Latent Vector ($z$)** | 100-dimensional standard normal noise |
| **Normalization** | Mean = 0.5, Std = 0.5 (maps images to $[-1, 1]$) |

---

## Results & Visualization

* Evaluated generator output at epoch 50 across an $8\times 8$ grid (64 generated samples).
* Stable adversarial equilibrium maintained across 50 epochs without mode collapse using LeakyReLU and intermediate Batch Normalization.

---

## Tech Stack
* Python 3.10+
* PyTorch & Torchvision
* Matplotlib & NumPy

## How to Run
```bash
git clone [https://github.com/chetans1706/gan-mnist.git](https://github.com/chetans1706/gan-mnist.git)
cd gan-mnist
jupyter notebook GanMnist.ipynb
