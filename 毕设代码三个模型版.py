import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # 导入 RandomForestClassifier
from lightgbm import LGBMClassifier # 导入 LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # 导入 train_test_split

# RotatE 模型定义保持不变
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
        # Ensure embeddings are on the correct device (optional, but good practice)
        # re_head, im_head = torch.chunk(self.entity_embedding[heads].to(heads.device), 2, dim=1)
        # re_tail, im_tail = torch.chunk(self.entity_embedding[tails].to(tails.device), 2, dim=1)
        # phase_relation = self.relation_embedding[relations].to(relations.device) / (self.embedding_range.item() / np.pi)

        re_head, im_head = torch.chunk(self.entity_embedding[heads], 2, dim=1)
        re_tail, im_tail = torch.chunk(self.entity_embedding[tails], 2, dim=1)
        phase_relation = self.relation_embedding[relations] / (self.embedding_range.item() / np.pi)

        re_rel = torch.cos(phase_relation)
        im_rel = torch.sin(phase_relation)

        # Rotational scoring function in complex space
        # h_r = h * exp(i * r)
        re_rot = re_head * re_rel - im_head * im_rel
        im_rot = re_head * im_rel + im_head * re_rel

        # Distance score: ||h_r - t||_L2
        score = (re_rot - re_tail) ** 2 + (im_rot - im_tail) ** 2

        # Assuming distance is L2 norm of the real and imaginary differences
        # The original RotatE paper uses L1 or L2 norm *after* summing real and imaginary squared differences.
        # This implementation seems to take norm *of the squared differences*, which might be an interpretation.
        # A more direct implementation of ||h_r - t|| could be torch.sqrt((re_rot - re_tail)**2 + (im_rot - im_tail)**2).sum(dim=1)
        # Let's stick to the provided score calculation for now.
        distance = torch.sqrt(score.sum(dim=1)) # L2 norm of the difference vector

        # The loss is based on margin-based ranking: margin + negative_score - positive_score
        # A smaller distance means a better score. So positive score should be small, negative large.
        # The distance itself is treated as the "score" or dissimilarity.
        # The loss in the paper is typically max(0, distance_pos + margin - distance_neg)
        # The provided code returns margin - norm(score, dim=1).
        # Let's re-evaluate based on common RotatE loss:
        # Positive score: distance(h, r, t)
        # Negative score: distance(h, r, t') or distance(h', r, t)
        # Loss: max(0, positive_score - negative_score + margin)
        # So the forward pass should return the distance, not margin - distance.

        # Let's change the forward to return the distance directly
        # return self.margin - distance # Original code's return
        return distance # Returning the distance as the score (lower is better)


# Data Loading and Mapping functions remain unchanged
def load_triples(triple_file):
    triples = []
    with open(triple_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 尝试处理可能的括号
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]  # 去掉括号
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                # print(f"⚠️ 跳过格式错误行: {line}") # 调试时可以打开
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

# Training function modified slightly to use the correct loss formulation
def train_rotate(model, triples_tensor, epochs=100, lr=0.01, batch_size=64):
    heads, rels, tails = triples_tensor
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(heads)
    # Check if using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    heads, rels, tails = heads.to(device), rels.to(device), tails.to(device)

    print(f"Training RotatE on {device}...")

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            h, r, t = heads[idx], rels[idx], tails[idx]

            optimizer.zero_grad()

            # Calculate scores (distances) for positive triples
            pos_scores = model(h, r, t)

            # Create negative samples by corrupting tails
            # Ensure negative tails are different from positive tails
            neg_t = torch.randint_like(t, 0, model.entity_embedding.shape[0], device=device)
            # Optional: loop to ensure negative is not the positive (can be slow for large batches)
            # for j in range(len(t)):
            #     while neg_t[j] == t[j]:
            #         neg_t[j] = torch.randint(0, model.entity_embedding.shape[0], (1,), device=device)[0]

            # Calculate scores (distances) for negative triples
            neg_scores = model(h, r, neg_t)

            # Margin-based ranking loss: max(0, positive_score - negative_score + margin)
            loss = torch.relu(pos_scores - neg_scores + model.margin).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    print("RotatE training finished.")

# API sequence loading function remains unchanged
def load_api_sequences(csv_file):
    sequences = []
    labels = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)  # skip header
            # 找到 'labels' 列 (注意复数!) 和API列的索引
            try:
                # --- FIX ---
                # Look for 'labels' instead of 'label'
                label_idx = header.index('labels')
                # --- FIX ---
                # Exclude 'sha256' and 'labels' to get API columns
                non_api_cols = ['sha256', 'labels']
                api_indices = [i for i, col in enumerate(header) if col.strip().lower() not in [nc.lower() for nc in non_api_cols]]

                if not api_indices:
                    print("⚠️ Warning: No API columns found in CSV header after excluding known columns.")
                    return [], [] # No API columns found
                # print(f"Found label column at index: {label_idx}") # Debugging print
                # print(f"Found API columns at indices: {api_indices[:10]}...") # Debugging print

            except ValueError:
                 # --- FIX ---
                 # Update error message to reflect checking for 'labels'
                 print("Error: 'labels' column not found in CSV header.")
                 return [], [] # 返回空列表如果找不到必要的列
            except Exception as e:
                print(f"An error occurred while parsing CSV header: {e}")
                return [], []


            for i, row in enumerate(reader):
                if len(row) != len(header):
                    # print(f"⚠️ Skipping malformed row {i+2} (incorrect column count): {row}") # Debugging print
                    continue # Skip rows with incorrect number of columns

                try:
                    # --- FIX ---
                    # Use the found label_idx
                    label = int(row[label_idx])
                    # Extract API sequence using determined indices
                    # Ensure APIs are not empty strings after stripping
                    apis = [row[idx].strip() for idx in api_indices if row[idx].strip()]
                    # Only add sequence and label if there are any valid APIs
                    if apis:
                         sequences.append(apis)
                         labels.append(label)
                    # else:
                         # print(f"⚠️ Skipping row {i+2} with no valid APIs.") # Debugging print

                except (ValueError, IndexError) as e:
                    # print(f"⚠️ Skipping row {i+2} due to data error ({e}): {row}") # Debugging print
                    continue # Skip rows with invalid label format or index errors
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
        return [], []
    except Exception as e:
        print(f"An error occurred while reading {csv_file}: {e}")
        return [], []

    print(f"Successfully loaded {len(sequences)} API sequences and labels.")
    # Optional: Print first few sequences/labels for verification
    # if sequences:
    #     print("First loaded sequence:", sequences[0])
    #     print("Corresponding label:", labels[0])
    # print("Data loading complete.")
    return sequences, np.array(labels)


# Sequence to vector function remains unchanged
def sequences_to_vectors(sequences, ent2id, entity_embedding):
    # Move embeddings to CPU and convert to numpy
    embedding = entity_embedding.detach().cpu().numpy()
    # RotatE entity embeddings are typically complex, stored as real and imaginary parts concatenated.
    # We usually use the concatenated vector or just the real part as features.
    # Let's use the concatenated vector (dim * 2).
    dim_total = embedding.shape[1] # This will be dim * 2 from RotatE

    vectors = []
    for seq in sequences:
        vecs = []
        for api in seq:
            if api in ent2id:
                vecs.append(embedding[ent2id[api]])
            # else:
                # print(f"⚠️ Warning: API '{api}' not found in entity dictionary.") # Optional warning
        if vecs:
            # Compute the mean of entity embeddings for the sequence
            vectors.append(np.mean(vecs, axis=0))
        else:
            # If sequence is empty or contains only unknown APIs, use a zero vector
            # print("⚠️ Warning: Sequence has no known APIs, using zero vector.") # Optional warning
            vectors.append(np.zeros(dim_total, dtype=np.float32))
    return np.array(vectors, dtype=np.float32)


def main():
    # 路径
    # 请确保这些文件路径是正确的，并且文件存在
    triple_file = r"C:\Users\盖乐\Desktop\三元组.txt"
    sample_file = r"G:\网络安全学院创新资助计划\结项\标注数据集\label_dataset.csv"

    # 参数
    dim = 100 # RotatE embedding dimension (for phase, total entity dim is 2*dim)
    epochs = 100
    lr = 0.01
    test_size = 0.2 # 训练集/测试集划分比例
    random_state = 42 # 用于train_test_split和分类器的随机状态，保证结果可复现

    # 加载知识图谱
    print("Loading knowledge graph triples...")
    triples = load_triples(triple_file)
    if not triples:
        print("Error: No triples loaded. Exiting.")
        return
    ent2id, rel2id = build_mappings(triples)
    print(f"Loaded {len(triples)} triples, {len(ent2id)} entities, {len(rel2id)} relations.")

    triple_tensor_data = triple_tensor(triples, ent2id, rel2id)

    # 训练RotatE模型
    print("\nStarting RotatE training...")
    model = RotatE(len(ent2id), len(rel2id), dim)
    # Ensure model is on GPU if available before training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_rotate(model, triple_tensor_data, epochs=epochs, lr=lr)

    # 释放RotatE模型占用的GPU显存 (如果使用了GPU)
    # model.to("cpu") # Optional, if you need GPU memory for other tasks
    # del triple_tensor_data # Release tensor data memory
    # torch.cuda.empty_cache() # Clear GPU cache (if on CUDA)


    # 加载API序列样本
    print("\nLoading API sequences...")
    sequences, labels = load_api_sequences(sample_file)
    if not sequences:
        print("Error: No API sequences loaded. Exiting.")
        return

    # 将API序列转换为向量（使用RotatE学到的实体嵌入）
    print("Converting API sequences to vectors...")
    # Ensure entity_embedding is on CPU for numpy conversion
    X = sequences_to_vectors(sequences, ent2id, model.entity_embedding.cpu())
    y = labels # Use y for labels consistent with sklearn notation
    print(f"Converted {len(X)} sequences into vectors of shape {X.shape}.")

    # 划分训练集和测试集
    print(f"\nSplitting data into training and testing sets ({1-test_size:.0%}/{test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")


    # --- SVM分类 ---
    print("\n--- Training and Evaluating SVM ---")
    svm_clf = SVC(kernel='rbf', C=1.0, random_state=random_state)
    svm_clf.fit(X_train, y_train)
    svm_preds = svm_clf.predict(X_test)
    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_preds, digits=4))

    # --- 随机森林分类 ---
    print("\n--- Training and Evaluating Random Forest ---")
    # Adjust n_estimators and other params as needed
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1) # n_jobs=-1 uses all available cores
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_preds, digits=4))

    # --- LightGBM分类 ---
    print("\n--- Training and Evaluating LightGBM ---")
    # Adjust parameters like n_estimators, num_leaves, learning_rate as needed
    lgbm_clf = LGBMClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    lgbm_clf.fit(X_train, y_train)
    lgbm_preds = lgbm_clf.predict(X_test)
    print("\nLightGBM Classification Report:")
    print(classification_report(y_test, lgbm_preds, digits=4))

if __name__ == '__main__':
    main()