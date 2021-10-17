import argparse, time, torch, os, logging, warnings, sys

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from model import BertMetaLearning
from datapath import loc, get_loc
from utils.logger import Logger

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--hidden_dims", type=int, default=768, help="")  # 768

# bert-base-multilingual-cased
# xlm-roberta-base
parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)

parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--qa_labels", type=int, default=2, help="")
parser.add_argument("--tc_labels", type=int, default=10, help="")
parser.add_argument("--po_labels", type=int, default=18, help="")
parser.add_argument("--pa_labels", type=int, default=2, help="")

parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")  # 32
parser.add_argument("--tc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--po_batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--pa_batch_size", type=int, default=8, help="batch size")

parser.add_argument("--seed", type=int, default=63, help="seed for numpy and pytorch")
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--tpu", action="store_true", help="use TPU")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="model.pt", help="")
parser.add_argument("--log_file", type=str, default="main_output.txt", help="")
parser.add_argument("--grad_clip", type=float, default=5.0)
parser.add_argument("--datasets", type=str, default="sc_en")

parser.add_argument(
    "--sampler", type=str, default="uniform_batch", choices=["uniform_batch"]
)
parser.add_argument("--temp", type=float, default=1.0)

parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--n_best_size", default=20, type=int)  # 20
parser.add_argument("--max_answer_length", default=30, type=int)  # 30

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))
print(args)

task_types = args.datasets.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
    if tt[:2] in task_types:
        list_of_tasks.append(tt)

for tt in task_types:
    if "_" in tt:
        list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
print(list_of_tasks)

if torch.cuda.is_available():
    if not args.cuda:
        args.cuda = True

    torch.cuda.manual_seed_all(args.seed)

DEVICE = torch.device("cuda" if args.cuda else "cpu")


def main():
    # loader
    dataloaders = []

    for k in list_of_tasks:
        data = None
        batch_size = 32

        if "qa" in k:
            data = CorpusQA(
                *get_loc("train", k, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.qa_batch_size
        elif "sc" in k:
            data = CorpusSC(
                *get_loc("train", k, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.sc_batch_size
        elif "tc" in k:
            data = CorpusTC(
                get_loc("train", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.tc_batch_size
        elif "po" in k:
            data = CorpusPO(
                get_loc("train", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.po_batch_size
        elif "pa" in k:
            data = CorpusPA(
                get_loc("train", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.pa_batch_size
        else:
            continue

        dataloader = DataLoader(data, shuffle=False, batch_size=batch_size)
        dataloaders.append(dataloader)

    # model = BertMetaLearning(args).to(DEVICE)

    print(f"loading model {args.load}...")
    model = torch.load(args.load)

    global_time = time.time()

    print("\n------------------ Generate Embeddings ------------------\n")

    embeddings = []
    for dataloader in dataloaders:
        embedding = embed(model, dataloader)
        embeddings.append(embedding)

    embeddings = torch.stack(embeddings)

    print("save embeddings to embeddings.pt...")
    torch.save(embeddings, "embeddings.pt")

    print(f"{time.time() - global_time}s")

    print("\n----------------------- Apply PCA -----------------------\n")

    pca = PCA(n_components=0.99, svd_solver="full", copy=False)
    principalComponents = pca.fit_transform(embeddings)

    print(principalComponents.shape)

    del embeddings

    print(f"{time.time() - global_time}s")

    print("\n---------------------- Clustering -----------------------\n")

    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
    labels = np.array([label_dict[label] for label in data["label"].to_list()])

    embeddings_0 = principalComponents[labels == 0]
    embeddings_1 = principalComponents[labels == 1]
    embeddings_2 = principalComponents[labels == 2]

    print(embeddings_0.shape)
    print(embeddings_1.shape)
    print(embeddings_2.shape)

    kmeans_0 = KMeans(n_clusters=2, random_state=0).fit(embeddings_0)
    kmeans_1 = KMeans(n_clusters=2, random_state=0).fit(embeddings_1)
    kmeans_2 = KMeans(n_clusters=2, random_state=0).fit(embeddings_2)

    print(f"{time.time() - global_time}s")

    print("\n-------------------- Plot Clusters ----------------------\n")

    plt.rcParams["figure.figsize"] = 15, 10

    def plot_clusters(embeddings, kmeans, marker="o", label=""):
        cluster_1 = embeddings[kmeans.labels_ == 0]
        cluster_2 = embeddings[kmeans.labels_ == 1]

        plt.scatter(
            cluster_1[:, 0], cluster_1[:, 1], marker=marker, label=f"{label}: cluster 1"
        )
        plt.scatter(
            cluster_2[:, 0], cluster_2[:, 1], marker=marker, label=f"{label}: cluster 2"
        )

    plot_clusters(embeddings_0, kmeans_0, marker="+", label="entailment")
    plot_clusters(embeddings_1, kmeans_1, marker="x", label="contradiction")
    plot_clusters(embeddings_2, kmeans_2, marker=".", label="neutral")
    plt.legend()
    plt.show()

    print(f"{time.time() - global_time}s")

    print("\n------ Save all label-cluster combinations subsets ------\n")

    ids = data.index.to_numpy()

    def get_cluster_idx(kmeans, ids):
        return ids[kmeans.labels_ == 0], ids[kmeans.labels_ == 1]

    e = get_cluster_idx(kmeans_0, ids[labels == 0])
    c = get_cluster_idx(kmeans_1, ids[labels == 1])
    n = get_cluster_idx(kmeans_2, ids[labels == 2])

    for i, combination in enumerate(list(itertools.product(e, c, n))):
        idx_list = np.concatenate(combination)
        data_subset = data.loc[idx_list, :].sort_index()
        data_subset.to_csv(f"train-{i}.csv", sep="\t", header=None, index=None)


def embed(model, dataloader):
    model.eval()
    with torch.no_grad():
        embeddings = torch.tensor([]).to(DEVICE)

        for pair_token_ids, mask_ids, seg_ids, y in tqdm(dataloader):
            pair_token_ids = pair_token_ids.to(DEVICE)
            mask_ids = mask_ids.to(DEVICE)
            seg_ids = seg_ids.to(DEVICE)
            labels = y.to(DEVICE)
            outputs = model(
                pair_token_ids,
                token_type_ids=seg_ids,
                attention_mask=mask_ids,
                labels=labels,
                output_hidden_states=True,
            )

            # embeddding at the [CLS] token:
            cls_embedding = outputs.hidden_states[0][:, 0, :]

            # mean pooled representation of premise & hypothesis
            pooled_representation = outputs.hidden_states[-1]
            premise_representation = torch.stack(
                [i[j == 0].mean(axis=0) for i, j in zip(pooled_representation, seg_ids)]
            )
            hypothesis_representation = torch.stack(
                [i[j == 1].mean(axis=0) for i, j in zip(pooled_representation, seg_ids)]
            )

            embedding = torch.cat(
                (cls_embedding, premise_representation, hypothesis_representation), 1
            )
            embeddings = torch.cat((embeddings, embedding), 0)

    return embeddings


if __name__ == "__main__":
    main()

