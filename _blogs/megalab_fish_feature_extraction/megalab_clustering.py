import re
import polars as pl
from umap import UMAP
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import PIL.Image as Image
import cv2
from tqdm import tqdm

sns.set_style("whitegrid")


def extract_crop(frame, bbox):
    x, y, w, h = bbox
    orig_h, orig_w, _ = frame.shape
    x = int(x * orig_w / 1024)
    y = int(y * orig_h / 1024)
    w = int(w * orig_w / 1024)
    h = int(h * orig_h / 1024)
    crop = frame[y:y+h, x:x+w]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


def create_vit_small_patch16_224_in21k():
    model = timm.create_model('vit_small_patch16_224_in21k', pretrained=True)
    model = model.eval().to("cuda")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    transform = create_transform(
        **resolve_data_config({}, model=model)
    )

    def process(frames: np.ndarray):
        images = []
        for frame in frames:
            image = Image.fromarray(frame)
            images.append(transform(image))
        images = torch.stack(images).to("cuda")
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = model.forward_features(images)
        del images
        return features.view(features.size(0), -1).cpu().numpy().tolist()
    
    return process


def create_xcit_small_24_p16_224():
    model = timm.create_model('xcit_small_24_p16_224', pretrained=True)
    model = model.eval().to("cuda")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    transform = create_transform(
        **resolve_data_config({}, model=model)
    )

    def process(frames: np.ndarray):
        images = []
        for frame in frames:
            image = Image.fromarray(frame)
            images.append(transform(image))
        images = torch.stack(images).to("cuda")
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = model.forward_features(images)
        del images
        return features.view(features.size(0), -1).cpu().numpy().tolist()
    
    return process


def create_mobilenetv3_large_100_miil_in21k():
    model = timm.create_model('mobilenetv3_large_100_miil_in21k', pretrained=True)
    model = model.eval().to("cuda")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    transform = create_transform(
        **resolve_data_config({}, model=model)
    )

    def process(frames: np.ndarray):
        images = []
        for frame in frames:
            image = Image.fromarray(frame)
            images.append(transform(image))
        images = torch.stack(images).to("cuda")
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = model.forward_features(images)
        del images
        return features.view(features.size(0), -1).cpu().numpy().tolist()
    
    return process


def extract_features(df: pl.DataFrame):

    process_vit = create_vit_small_patch16_224_in21k()
    process_xcit = create_xcit_small_24_p16_224()
    process_mnetv3 = create_mobilenetv3_large_100_miil_in21k()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_c = torch.cuda.Stream()

    loader = tqdm(total=len(df))

    results = []
    crops = []
    metadatas = []

    for video_file, gdf in df.group_by("video_file"):
        video_file = video_file[0]

        capture = cv2.VideoCapture(f"../../../megalab_recordings/recordings/{video_file}")

        for frame_id, ggdf in gdf.group_by("frame_id"):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id[0])
            _, frame = capture.read()

            for row in ggdf.iter_rows(named=True):
                crops.append(extract_crop(frame, row["box"]))
                metadatas.append({ "video_file": video_file, "frame_id": frame_id[0] })

                if len(crops) >= 32:
                    loader.update(len(crops))
                    with torch.cuda.stream(stream_a):
                        xcit_features = process_xcit(crops)
                    with torch.cuda.stream(stream_b):
                        vit_features = process_vit(crops)
                    with torch.cuda.stream(stream_c):
                        mnetv3_features = process_mnetv3(crops)
                    stream_a.synchronize()
                    stream_b.synchronize()
                    stream_c.synchronize()
                    for m, (xcit_feature, vit_feature, mnetv3_feature) in zip(metadatas, zip(xcit_features, vit_features, mnetv3_features)):
                        results.append({**m, "xcit_feature": xcit_feature, "vit_feature": vit_feature, "mnetv3_feature": mnetv3_feature})
                    crops = []
                    metadatas = []

        capture.release()

        if len(results) >= 10000:
            rdf = pl.DataFrame(results)
            rdf.write_delta("megalab_features", mode="append")
            results = []

    if len(crops) > 0:
        loader.update(len(crops))
        with torch.cuda.stream(stream_a):
            xcit_features = process_xcit(crops)
        with torch.cuda.stream(stream_b):
            vit_features = process_vit(crops)
        with torch.cuda.stream(stream_c):
            mnetv3_features = process_mnetv3(crops)
        stream_a.synchronize()
        stream_b.synchronize()
        stream_c.synchronize()
        for m, (xcit_feature, vit_feature, mnetv3_feature) in zip(metadatas, zip(xcit_features, vit_features, mnetv3_features)):
            results.append({**m, "xcit_feature": xcit_feature, "vit_feature": vit_feature, "mnetv3_feature": mnetv3_feature})
        crops = []
        metadatas = []

    if len(results) > 0:
        rdf = pl.DataFrame(results)
        rdf.write_delta("megalab_features", mode="append")
        results = []

    loader.close()
    torch.cuda.empty_cache()

    return "megalab_features"


def reduce_dimensions(rdf: pl.DataFrame, column: str, n_umap_components: int = 4):
    scaler = StandardScaler()

    print("Scaling features...")

    features = np.vstack(rdf.select(column).to_series().to_list())
    features = scaler.fit_transform(features)

    print("Performing PCA dimensionality reduction to 95% variance...")

    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(features)

    print(f"Performing UMAP dimensionality reduction to {n_umap_components} components...")

    umap = UMAP(n_neighbors=15, n_components=n_umap_components, metric='euclidean')
    umap_features = umap.fit_transform(features)

    print("Plotting results...")

    rdf = rdf.with_columns([
        pl.Series(f"{column}_umap", umap_features.tolist()),
        pl.Series(f"{column}_pca", pca_features.tolist())
    ])

    plt.figure(figsize=(5, 3))
    sns.lineplot(
        x=np.arange(1, len(pca.explained_variance_ratio_)+1),
        y=np.cumsum(pca.explained_variance_ratio_)
    )
    plt.title(f"PCA Explained Variance Ratio on '{column}' Features")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

    percent_reduction = (1 - pca.n_components_ / features.shape[1]) * 100
    print(f"Original feature dimension: {features.shape[1]}")
    print(f"Reduced feature dimension: {pca.n_components_}")
    print(f"Percent reduction in feature dimension: {percent_reduction:.2f}%")

    return rdf


def find_optimal_clusters(rdf: pl.DataFrame, feature_column: str, max_k: int, sample_size: int = 10_000):

    def find_optimal_k(data, max_k):
        intertias = []
        scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans = kmeans.fit(data)
            intertias.append(kmeans.inertia_)
            scores.append(silhouette_score(data, kmeans.labels_))

        return intertias, scores
    
    scaler = StandardScaler()
    
    features = np.vstack(
        rdf\
            .select(feature_column)\
            .sample(sample_size)\
            .to_series()\
            .to_list()
    )

    features = scaler.fit_transform(features)

    intertias, scores = find_optimal_k(features, max_k=max_k)

    plt.figure(figsize=(10, 5))
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(2, max_k + 1), intertias, 'b-', label='Inertia')
    ax2.plot(range(2, max_k + 1), scores, 'r-', label='Silhouette Score')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax2.set_ylabel('Silhouette Score')

    plt.title(f"KMeans Elbow Method on '{feature_column}' Features")
    plt.tight_layout()
    plt.show()


def pairplot_features(rdf: pl.DataFrame, feature_column: str, cluster_column: str):
    scaler = StandardScaler()
    features = np.vstack(rdf.select(feature_column).to_series().to_list())
    features = scaler.fit_transform(features)

    cluster_labels = rdf.select(cluster_column).to_series().to_list()

    plot_samples = pd.DataFrame({
        [feature_column]: list(features),
        [cluster_column]: cluster_labels
    })

    plot_samples = plot_samples\
        .join(plot_samples[feature_column].apply(pd.Series).add_prefix("feature_"))\
        .drop(columns=[feature_column])

    sns.pairplot(
        plot_samples,
        hue=cluster_column,
        palette="viridis",
        kind="hist",
        diag_kind="kde",
    )
    plt.suptitle(f"Components of '{feature_column}' Pairplot Colored by Cluster")
    plt.show()