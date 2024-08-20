
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

# Loading the audio file
start = 10
duration = 300

audio_data, sampling_rate = librosa.load('data/ElephantIsland2013Aural/wav/20130117_090000_AWI251-01_AU0231_250Hz.wav', sr=250, offset=start, duration=duration)
duration = librosa.get_duration(y=audio_data, sr=sampling_rate)

# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(audio_data, sr=sampling_rate)
# plt.show()

audio_data = librosa.effects.preemphasis(audio_data)

window_length = 314
hop_length = 85
fft_size = 314
n_mels=80
# window_length = 256
# hop_length = 64
# fft_size = 256

D = librosa.stft(audio_data, n_fft=fft_size, hop_length=hop_length, win_length=window_length, window='hann')
spectrogram = librosa.amplitude_to_db(np.abs(D))
power_spectrum=np.abs(D)**2

num_mfcc = 10
mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=num_mfcc, hop_length=hop_length, win_length=window_length)

# Standardize the MFCC features
scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs.T)

pca = PCA(n_components=4)  
mfccs_pca = pca.fit_transform(mfccs_scaled)

sigma = np.median(np.std(mfccs_pca, axis=0))
affinity_matrix = rbf_kernel(mfccs_pca, gamma=1/(2*(sigma)**2))

#Degree matrix
degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

sqrt_deg_matrix = np.diag(1.0 / np.sqrt(np.sum(affinity_matrix, axis=1)))
norm_laplacian_matrix = np.eye(degree_matrix.shape[0]) - sqrt_deg_matrix @ affinity_matrix @ sqrt_deg_matrix

#Eigenvalue decomp
eig_values, eig_vectors = eigh(norm_laplacian_matrix)
idx = eig_values.argsort()
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:,idx]

k = 2  # Number of clusters
feature_vector = eig_vectors[:, :k]
feature_vector = StandardScaler().fit_transform(feature_vector)

# Apply k-means clustering
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(feature_vector)

time_axis = np.arange(len(labels)) * (duration / len(labels))  # Time in seconds

# Plot the cluster labels over time
plt.figure(figsize=(15, 5))
plt.scatter(time_axis, labels, c=labels, cmap='viridis', s=10)
plt.xlabel('Time (s)')
plt.ylabel('Cluster Label')
# plt.title('Spectral Clustering of 5-Second Overlapping Audio Intervals')
plt.show()


