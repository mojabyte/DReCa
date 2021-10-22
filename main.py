import argparse, time, torch, os, logging, warnings, sys

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from data import CorpusQA, CorpusSC

# from model import BertMetaLearning
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
    default="bert-base-multilingual-cased",
    help="name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)

parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--qa_labels", type=int, default=2, help="")

parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")

parser.add_argument("--seed", type=int, default=63, help="seed for numpy and pytorch")
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--tpu", action="store_true", help="use TPU")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="model.pt", help="")
parser.add_argument("--load_embeddings", type=str, default="", help="")
parser.add_argument("--load_pca", type=str, default="", help="")
parser.add_argument("--log_interval", type=int, default=100, help="")
parser.add_argument("--log_file", type=str, default="main_output.txt", help="")
parser.add_argument("--grad_clip", type=float, default=5.0)
parser.add_argument("--datasets", type=str, default="sc_en")
parser.add_argument("--num_workers", type=int, default=0, help="")

parser.add_argument("--embed", action="store_true", help="Generate embeddings")
parser.add_argument("--pca", action="store_true", help="Calculate PCA")
parser.add_argument("--cluster", action="store_true", help="Clustering")

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
    print("********************\n", "cuda available", "\n********************")
    if not args.cuda:
        args.cuda = True

    torch.cuda.manual_seed_all(args.seed)

DEVICE = torch.device("cuda" if args.cuda else "cpu")


def main():
    global_time = time.time()

    if args.load_pca != "":
        print(f"load pca {args.load_pca}...")
        principalComponents = torch.load(args.load_pca)
    else:
        embeddings = []

        if args.load_embeddings != "":
            print(f"load embeddings {args.load_embeddings}...")
            embeddings = torch.load(args.load_embeddings)
        elif args.embed:
            # loader
            data_list = []
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
                else:
                    continue

                data_list.append(data)
                dataloader = DataLoader(data, shuffle=False, batch_size=batch_size)
                dataloaders.append(dataloader)

            print(f"loading model {args.load}...")
            model = torch.load(args.load).to(DEVICE)

            global_time = time.time()

            print("\n------------------ Load Datasets ------------------\n")

            data = None
            header = ["premise", "hypothesis", "label"]
            for task in list_of_tasks:
                path = get_loc("train", k, args.data_dir)[0]
                df = pd.read_csv(path, sep="\t", header=None, names=header)
                if data == None:
                    data = df
                else:
                    data.append(df)

            print(f"{time.time() - global_time:.0f}s")

            print("\n------------------ Generate Embeddings ------------------")

            for task, dataloader in zip(list_of_tasks, dataloaders):
                print(f"\nEmbedding {task} dataset")
                print("======" * 10)
                embedding = embed(model, dataloader, task)
                embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim=0).cpu()

            print("save embeddings...")
            torch.save(embeddings, os.path.join(args.save, "embeddings.pt"))

        print(f"{time.time() - global_time:.0f}s")

        if args.pca:
            print("\n----------------------- Apply PCA -----------------------\n")

            pca = PCA(n_components=0.99, svd_solver="full", copy=False)
            principalComponents = pca.fit_transform(embeddings)
            del embeddings
            print(f"PCA output shape: {principalComponents.shape}")
            torch.save(principalComponents, os.path.join(args.save, "pca.pt"), pickle_protocol=4)

            print(f"{time.time() - global_time:.0f}s")

    if args.cluster:

        print("\n---------------------- Clustering -----------------------\n")

        # TODO: loop over all task_list labels
        labels = np.array(
            [data_list[0].label_dict[label] for label in data["label"].to_list()]
        )

        embeddings_0 = principalComponents[labels == 0]
        embeddings_1 = principalComponents[labels == 1]
        embeddings_2 = principalComponents[labels == 2]

        print(embeddings_0.shape)
        print(embeddings_1.shape)
        print(embeddings_2.shape)

        kmeans_0 = KMeans(n_clusters=2, random_state=0).fit(embeddings_0)
        kmeans_1 = KMeans(n_clusters=2, random_state=0).fit(embeddings_1)
        kmeans_2 = KMeans(n_clusters=2, random_state=0).fit(embeddings_2)

        print(f"{time.time() - global_time:.0f}s")

        print("\n-------------------- Plot Clusters ----------------------\n")

        plt.rcParams["figure.figsize"] = 15, 10

        def plot_clusters(embeddings, kmeans, marker="o", label=""):
            cluster_1 = embeddings[kmeans.labels_ == 0]
            cluster_2 = embeddings[kmeans.labels_ == 1]

            plt.scatter(
                cluster_1[:, 0],
                cluster_1[:, 1],
                marker=marker,
                label=f"{label}: cluster 1",
            )
            plt.scatter(
                cluster_2[:, 0],
                cluster_2[:, 1],
                marker=marker,
                label=f"{label}: cluster 2",
            )

        plot_clusters(embeddings_0, kmeans_0, marker="+", label="entailment")
        plot_clusters(embeddings_1, kmeans_1, marker="x", label="contradiction")
        plot_clusters(embeddings_2, kmeans_2, marker=".", label="neutral")
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(args.save, "clusters.png"))

        print(f"{time.time() - global_time:.0f}s")

        print("\n------ Save all label-cluster combinations subsets ------\n")

        ids = data.index.to_numpy()

        def get_cluster_idx(kmeans, ids):
            return ids[kmeans.labels_ == 0], ids[kmeans.labels_ == 1]

        e = get_cluster_idx(kmeans_0, ids[labels == 0])
        c = get_cluster_idx(kmeans_1, ids[labels == 1])
        n = get_cluster_idx(kmeans_2, ids[labels == 2])

        tasks_path = os.path.join(args.save, "tasks")
        if not os.path.exists(tasks_path):
            os.makedirs(tasks_path)

        for i, combination in enumerate(list(itertools.product(e, c, n))):
            idx_list = np.concatenate(combination)
            data_subset = data.loc[idx_list, :].sort_index()
            data_subset.to_csv(
                os.path.join(tasks_path, f"task-{i}.csv"),
                sep="\t",
                header=None,
                index=None,
            )

        print(f"{time.time() - global_time:.0f}s")


def embed(model, dataloader, task):
    model.eval()
    with torch.no_grad():
        embeddings = torch.tensor([]).to(DEVICE)

        timer = time.time()
        for i, batch in enumerate(dataloader):
            outputs = model.forward(task, batch, output_hidden_states=True)

            last_hidden_state = outputs[2][-1]

            # embeddding at the [CLS] token:
            cls_embedding = last_hidden_state[:, 0, :]

            # mean pooled representation of premise & hypothesis
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            seg_ids = token_type_ids + attention_mask
            seg_ids[:, 0] = 0

            premise_seg = (seg_ids == 1).unsqueeze(-1).float()
            premise_representation = torch.sum(
                last_hidden_state * premise_seg, dim=1
            ) / torch.sum(premise_seg, dim=1)

            hypothesis_seg = (seg_ids == 2).unsqueeze(-1).float()
            hypothesis_representation = torch.sum(
                last_hidden_state * hypothesis_seg, dim=1
            ) / torch.sum(hypothesis_seg, dim=1)

            embedding = torch.cat(
                (cls_embedding, premise_representation, hypothesis_representation), 1
            )
            embeddings = torch.cat((embeddings, embedding), 0)

            if (i + 1) % args.log_interval == 0 or (i + 1) == len(dataloader):
                print(
                    f"{time.time() - timer:.2f}s | batch#{i + 1} | {(i + 1) / len(dataloader) * 100:.2f}% completed"
                )
                timer = time.time()

    return embeddings


if __name__ == "__main__":
    main()
