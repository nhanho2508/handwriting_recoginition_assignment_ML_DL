from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm
import os
from mltu.tensorflow.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.annotations.images import CVImage

def fetch_and_extract(url, target_dir="Datasets", buffer_size=1024*1024):
    response = urlopen(url)

    content = b""
    total_chunks = response.length // buffer_size + 1
    for _ in tqdm(range(total_chunks)):
        content += response.read(buffer_size)

    with ZipFile(BytesIO(content)) as zip_file:
        zip_file.extractall(path=target_dir)

def process_word_data(words, dataset_path):
    dataset = []
    vocab = set()
    max_len = 0

    for line in tqdm(words):
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[1] == "err":
            continue

        folder1 = line_split[0][:3]
        folder2 = "-".join(line_split[0].split("-")[:2])
        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip("\n")

        rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
        if not os.path.exists(rel_path):
            print(f"File not found: {rel_path}")
            continue

        dataset.append([rel_path, label])
        vocab.update(label)
        max_len = max(max_len, len(label))

    return dataset, sorted(vocab), max_len

def create_data_providers(dataset, configs):
    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
    )

    return data_provider.split(split=0.9)
