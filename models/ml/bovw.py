import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


NUM_WORDS = 200
MAX_KEYPOINTS = 200
MAX_DESCRIPTORS = 200000
RANDOM_SEED = 42


def extract_sift(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS)
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def build_vocabulary(image_paths):
    all_descriptors = []
    print("Extracting SIFT descriptor...")

    for i, image_path in enumerate(image_paths):
        descriptors = extract_sift(image_path)

        if descriptors is not None:
            all_descriptors.append(descriptors)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images")

    all_descriptors = np.vstack(all_descriptors)

    if len(all_descriptors) > MAX_DESCRIPTORS:
        rng = np.random.default_rng(RANDOM_SEED)
        selected = rng.choice(
            len(all_descriptors),
            MAX_DESCRIPTORS,
            replace=False
        )
        all_descriptors = all_descriptors[selected]

    print(f"Using {len(all_descriptors)} descriptors")

    kmeans = MiniBatchKMeans(
        n_clusters=NUM_WORDS,
        batch_size=4096,
        random_state=RANDOM_SEED,
        n_init=3
    )

    kmeans.fit(all_descriptors)
    print("Visual vocabulary completed")

    return kmeans


def create_histogram(image_path, kmeans):
    descriptors = extract_sift(image_path)
    histogram = np.zeros(NUM_WORDS, dtype=np.float32)

    if descriptors is None:
        return histogram

    words = kmeans.predict(descriptors)

    for word in words:
        histogram[word] += 1

    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def create_bovw_features(image_paths, kmeans):
    features = []

    for i, image_path in enumerate(image_paths):
        histogram = create_histogram(image_path, kmeans)
        features.append(histogram)

        if (i + 1) % 1000 == 0:
            print(f"Created features for {i + 1}/{len(image_paths)} images")

    return np.array(features, dtype=np.float32)