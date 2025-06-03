import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=6.0, epsilon=2.0):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.embedding_range = nn.Parameter(
            torch.Tensor([(margin + epsilon) / dim]), requires_grad=False
        )
        self.entity_embedding = nn.Parameter(torch.zeros(num_entities, dim * 2))
        self.relation_embedding = nn.Parameter(torch.zeros(num_relations, dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.entity_embedding, -self.embedding_range.item(), self.embedding_range.item())
        nn.init.uniform_(self.relation_embedding, -self.embedding_range.item(), self.embedding_range.item())

    def forward(self, heads, relations, tails):
        re_head, im_head = torch.chunk(self.entity_embedding[heads], 2, dim=1)
        re_tail, im_tail = torch.chunk(self.entity_embedding[tails], 2, dim=1)
        phase_relation = self.relation_embedding[relations] / (self.embedding_range.item() / np.pi)
        re_rel = torch.cos(phase_relation)
        im_rel = torch.sin(phase_relation)
        re_rot = re_head * re_rel - im_head * im_rel
        im_rot = re_head * im_rel + im_head * re_rel
        score = (re_rot - re_tail) ** 2 + (im_rot - im_tail) ** 2
        return self.margin - torch.norm(score, dim=1)

def load_triples(triple_file):
    triples = []
    with open(triple_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]  # 去掉括号
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                print(f"⚠️ 跳过格式错误行: {line}")
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def build_mappings(triples):
    ent2id, rel2id = {}, {}
    for h, r, t in triples:
        for e in (h, t):
            if e not in ent2id:
                ent2id[e] = len(ent2id)
        if r not in rel2id:
            rel2id[r] = len(rel2id)
    return ent2id, rel2id

def triple_tensor(triples, ent2id, rel2id):
    heads = torch.LongTensor([ent2id[h] for h, _, _ in triples])
    rels = torch.LongTensor([rel2id[r] for _, r, _ in triples])
    tails = torch.LongTensor([ent2id[t] for _, _, t in triples])
    return heads, rels, tails

def train_rotate(model, triples_tensor, epochs=100, lr=0.01, batch_size=64):
    heads, rels, tails = triples_tensor
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(heads)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            h, r, t = heads[idx], rels[idx], tails[idx]
            optimizer.zero_grad()
            pos = model(h, r, t)
            neg_t = torch.randint_like(t, 0, model.entity_embedding.shape[0])
            neg = model(h, r, neg_t)
            loss = torch.relu(neg + model.margin - pos).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def load_api_sequences(csv_file):
    sequences = []
    labels = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)  # skip header
        for row in reader:
            label = int(row[1])
            apis = row[2:]  # 从第3列开始是API
            sequences.append(apis)
            labels.append(label)
    return sequences, np.array(labels)

def sequences_to_vectors(sequences, ent2id, entity_embedding):
    embedding = entity_embedding.detach().cpu().numpy()
    dim = embedding.shape[1]
    vectors = []
    for seq in sequences:
        vecs = []
        for api in seq:
            if api in ent2id:
                vecs.append(embedding[ent2id[api]])
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors, dtype=np.float32)

def main():
    # 路径
    triple_file = r"C:\Users\盖乐\Desktop\三元组.txt"
    sample_file = r"G:\网络安全学院创新资助计划\结项\标注数据集\label_dataset.csv"

    # 参数
    dim = 100
    epochs = 100
    lr = 0.01

    # 加载知识图谱
    triples = load_triples(triple_file)
    ent2id, rel2id = build_mappings(triples)
    triple_tensor_data = triple_tensor(triples, ent2id, rel2id)

    # 训练RotatE
    model = RotatE(len(ent2id), len(rel2id), dim)
    train_rotate(model, triple_tensor_data, epochs=epochs, lr=lr)

    # 加载API序列样本
    sequences, labels = load_api_sequences(sample_file)
    X = sequences_to_vectors(sequences, ent2id, model.entity_embedding)

    # SVM分类
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X, labels)
    preds = clf.predict(X)

    # 评估
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

if __name__ == '__main__':
    main()
