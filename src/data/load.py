from pathlib import Path
import numpy as np
import cv2

DATA_DIR  = Path('../data/raw/Data')

IMG_SIZE = (224, 224)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ImageLoader:
    data_classes =[]
    data_paths = []

    @classmethod
    def loadClasses(cls):
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {DATA_DIR.resolve()}"
            )

        cls.data_paths = sorted(
            path 
            for path in DATA_DIR.iterdir()
            if path.is_dir()
        )

        for path in cls.data_paths:
            files = list(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS)
            print(f"class: {path.name} containes {len(files)} files")

        cls.data_classes = [path.name for path in cls.data_paths]

    @classmethod
    def loadImages(cls):
        if not ( cls.data_paths or cls.data_classes ): cls.loadClasses()

        images = []
        labels = []

        for path in cls.data_paths:
            for imgPath in path.rglob("*"):
                if not imgPath.is_file():
                    continue
                if imgPath.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                try:
                    img = cv2.imread(str(imgPath))
                    if img is None:
                        print(f"image is unreadable : {imgPath}")
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    images.append(img)
                    labels.append(path.name)

                except (cv2.error, OSError) as e:
                    print(f"Failed to load {imgPath}: {e}")

        if not images:
            raise ValueError(
                f"No readable images found in {DATA_DIR.resolve()}"
            )

        np_images  =  np.asarray(images, dtype=np.uint8)
        np_labels = np.asarray(labels)

        print(f"Loaded images: {np_images.shape}")
        print(f"Loaded labels: {np_labels.shape}")
        print(f"Classes: {cls.data_classes}")

        return np_images, np_labels, cls.data_classes
