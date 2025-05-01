# text_analyzer.py
import re
import random
import heapq
from collections import defaultdict, deque
import networkx as nx

class DirectedGraph:
    def __init__(self):
        self.adj = defaultdict(lambda: defaultdict(int))

    def add_edge(self, u, v):
        self.adj[u][v] += 1
        if v not in self.adj:
            self.adj[v] = defaultdict(int)

    def vertices(self):
        return list(self.adj.keys())

    def neighbors(self, u):
        return dict(self.adj[u])

    def weight(self, u, v):
        return self.adj[u].get(v, 0)

class TextGraphAnalyzer:
    def __init__(self, damping=0.85):
        self.graph = DirectedGraph()
        self.damping = damping

    def build_graph_from_file(self, filepath):
        words = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                clean = re.sub(r'[^A-Za-z]', ' ', line).lower()
                words.extend(clean.split())
        for i in range(len(words) - 1):
            self.graph.add_edge(words[i], words[i+1])

    def adjacency_str(self):
        lines = []
        for u in sorted(self.graph.vertices()):
            nbrs = self.graph.neighbors(u)
            if not nbrs:
                lines.append(f"{u} -> []")
            else:
                lst = [f"{v}({w})" for v, w in nbrs.items()]
                lines.append(f"{u} -> [{', '.join(lst)}]")
        return '\n'.join(lines)

    def query_bridge_words(self, w1, w2):
        w1, w2 = w1.lower(), w2.lower()
        verts = set(self.graph.vertices())
        if w1 not in verts or w2 not in verts:
            return f"No {w1} or {w2} in the graph!"
        bridges = [mid for mid in self.graph.neighbors(w1) if self.graph.weight(mid, w2) > 0]
        if not bridges:
            return f"No bridge words from “{w1}” to “{w2}”!"
        return f"The bridge words from “{w1}” to “{w2}” are: {', '.join(bridges)}."

    def generate_new_text(self, input_text):
        clean = re.sub(r'[^A-Za-z\s]', ' ', input_text)
        ws = clean.split()
        output = []
        for i in range(len(ws)):
            output.append(ws[i])
            if i < len(ws) - 1:
                w, nxt = ws[i].lower(), ws[i+1].lower()
                bridges = [mid for mid in self.graph.neighbors(w) if self.graph.weight(mid, nxt) > 0]
                if bridges:
                    output.append(random.choice(bridges))
        return ' '.join(output)

    def calc_shortest_path(self, source, target):
        source, target = source.lower(), target.lower()
        verts = set(self.graph.vertices())
        if source not in verts or target not in verts:
            return f"{source} or {target} not in graph!"
        dist = {v: float('inf') for v in verts}
        prev = {}
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == target:
                break
            if d > dist[u]:
                continue
            for v, w in self.graph.neighbors(u).items():
                alt = d + w
                if alt < dist[v]:
                    dist[v], prev[v] = alt, u
                    heapq.heappush(pq, (alt, v))
        if target not in prev and source != target:
            return f"{source} cannot reach {target}!"
        path = deque()
        cur = target
        while True:
            path.appendleft(cur)
            if cur == source:
                break
            cur = prev[cur]
        return f"Shortest path: {' -> '.join(path)} (length={int(dist[target])})"

    def cal_pagerank(self, word, iterations=100, tol=1e-6):
        """
        计算所有节点的 PageRank 并返回指定 word 的 PR 值。
        公式：
          PR(u) = (1-d)/N + d * ( ∑_{v∈B_u} PR(v)/L(v)  +  ∑_{v∈D} PR(v)/N )
        其中 D 表示所有出度为 0 的“悬挂节点”集合。
        """
        word = word.lower()
        verts = self.graph.vertices()
        N = len(verts)
        if word not in verts:
            return None

        d = self.damping
        # 初始均分
        pr = {v: 1.0/N for v in verts}

        # 预先构建反向邻接：incoming[u] = 所有指向 u 的前驱节点列表
        incoming = {v: [] for v in verts}
        for u in verts:
            for v in self.graph.neighbors(u):
                incoming[v].append(u)

        for _ in range(iterations):
            new_pr = {}
            # 先计算悬挂节点贡献之和
            dangling_sum = sum(pr[u] for u in verts if len(self.graph.neighbors(u)) == 0)

            for u in verts:
                # 1）阻尼项
                rank = (1 - d) / N
                # 2）悬挂节点平均分给所有节点
                rank += d * dangling_sum / N
                # 3）正常入边贡献
                s = 0.0
                for v in incoming[u]:
                    L_v = len(self.graph.neighbors(v))
                    if L_v > 0:
                        s += pr[v] / L_v
                rank += d * s
                new_pr[u] = rank

            # （可选）判断收敛：如果两次差距小于 tol，则提前退出
            delta = sum(abs(new_pr[v] - pr[v]) for v in verts)
            pr = new_pr
            if delta < tol:
                break

        return pr.get(word)


    def random_walk(self, max_steps=1000):
        verts = self.graph.vertices()
        if not verts:
            return ""
        curr = random.choice(verts)
        walk = [curr]
        seen = set()
        for _ in range(max_steps):
            nbrs = self.graph.neighbors(curr)
            if not nbrs:
                break
            total = sum(nbrs.values())
            r = random.randint(1, total)
            s = 0
            for v, w in nbrs.items():
                s += w
                if s >= r:
                    nxt = v
                    break
            if (curr, nxt) in seen:
                break
            seen.add((curr, nxt))
            curr = nxt
            walk.append(curr)
        return ' '.join(walk)

    def draw_to_ax(self, ax):
        """
        使用 networkx 在传入的 ax 上绘制有权重的有向图，
        确保每条边的标签都是实际的 edge weight。
        """
        # 1. 构建 DiGraph 并显式存储 weight 属性
        G = nx.DiGraph()
        for u in self.graph.vertices():
            for v, w in self.graph.neighbors(u).items():
                if w > 0:
                    G.add_edge(u, v, weight=w)

        # 2. 选布局：少节点用 shell，多节点用 kamada_kawai
        pos = nx.shell_layout(G) if G.number_of_nodes() < 15 else nx.kamada_kawai_layout(G)

        # 3. 节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=800,
            node_color="#4B9CD3",
            alpha=0.9,
            linewidths=1.5,
            edgecolors="#2A2A2A",
            ax=ax
        )

        # 4. 边（仅画线，不画标签）
        nx.draw_networkx_edges(
            G, pos,
            width=1.2,
            edge_color="#666666",
            arrowstyle="->",
            arrowsize=20,
            connectionstyle="arc3,rad=0.2",
            ax=ax
        )

        # 5. 节点标签
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_color="white",
            font_weight="bold",
            verticalalignment="center_baseline",
            ax=ax
        )

        # 6. 边权重标签：从 G.edges(data=True) 中读取各自的 weight
        edge_labels = {
            (u, v): data.get('weight', 0)
            for u, v, data in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                boxstyle="round,pad=0.1"
            ),
            rotate=False,
            label_pos=0.5,
            ax=ax
        )

        # 7. 美化
        ax.set_facecolor("#F5F5F5")
        ax.set_title("Directed Text Graph with Weights", pad=20, fontweight="bold")
