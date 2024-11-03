# Whale Vocalization Clustering using Spectral Clustering and K-means

This repository contains two Jupyter notebooks that explore clustering techniques for whale vocalization data. These notebooks apply spectral clustering and K-means algorithms to audio recordings with varying levels of noise, utilizing key acoustic features such as MFCCs, chroma, and spectral contrast.

## Contents

- **`clustering_short_recordings.ipynb`** - This notebook processes recordings at 250 Hz and is used to focus on audio preprocessing and spectral clustering without added noise.
- **`clustering_long_recordings.ipynb`** - This notebook handles longer recordings at 2000 Hz, with Additive White Gaussian Noise (AWGN) added for noise robustness testing. It also compares Spectral clustering with K-means clustering across different SNR levels.

## Project Overview

The aim of this project is to develop a clustering model that is robust to noise and displays good performance across different signal to noise ratios. This study seeks to explore the effect of different audio features (MFCCs, delta, chroma and spectral contrasts) on spectral clustering performance and to try find the optimal configuration of the spectral clustering model for the purpose of clustering whale vocalisations. The goal is to determine the optimal combination of preprocessing steps and audio features that lead to accurate and reliable clustering of whale calls. The performance of a K-measn clustering and spectral clstering is assessed under different SNR levels through the use of AWGN.

### Key Features Extracted

1. **MFCCs**: Mel Frequency Cepstral Coefficients, a popular feature in audio processing.
2. **Delta MFCCs**: Temporal derivatives of MFCCs.
3. **Chroma**: Represents energy distribution across the 12 distinct pitch classes.
4. **Spectral Contrast**: Emphasizes differences between spectral peaks and valleys.

## Notebooks

1. **whale_clustering_short_recordings.ipynb**
   - **Input**: Audio files sampled at 250 Hz.
   - **Preprocessing**: Bandpass filtering and noise reduction.
   - **Clustering**: Spectral clustering with RBF, Polynomial, Cosine and dot product affinity matrices.
   - **Objective**: Identify best performing spectral clustering algorithm.
  
2. **whale_clustering_long_recordings_awgn.ipynb**
   - **Input**: Audio files sampled at 2000 Hz.
   - **Preprocessing**: Bandpass filtering and AWGN.
   - **Clustering**: Spectral clustering and K-means under varying SNRs using AWGN.
   - **Objective**: Evaluate clustering robustness between K-means clustering and spectral clustering across high-noise environments.

## Setup and Installation

### Prerequisites

1. **Python 3.7 or above**
2. **Jupyter Notebook**
3. Required packages:
   - `librosa`
   - `matplotlib`
   - `scipy`
   - `scikit-learn`
   - `maad` (for advanced audio processing)




