"""Constrained k-means clustering"""

# constrained k-means clustering implementation of
# Bradley, Paul S., Kristin P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering." Microsoft Research, Redmond 20.0 (2000): 0.

import numpy as np
from ortools.graph.python import min_cost_flow
from scipy.spatial.distance import cdist


class ConstrainedKMeans:
    def __init__(
        self,
        n_clusters,  # クラスタ数
        min_membership,  # クラスタあたりの最小データ数
        max_membership=None,  # クラスタあたりの最大データ数
        random_state=0,  # シード値
        max_iter=300,  # 最大イテレーション数
        tol=1e-4,  # 中心点の更新に伴う変動がこの値よりも小さければクラスタリングを終了する
    ):
        self.valid(
            n_clusters,
            min_membership,
            max_membership,
            random_state,
            max_iter,
            tol,
        )
        self.n_clusters = n_clusters
        self.min_membership = min_membership
        self.max_membership = max_membership
        self.random_state = np.random.RandomState(random_state)
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.labels = None

    # sklearnのkmeans_plusplusメソッドとほぼ一緒
    # 二乗ユークリッド距離の計算にscipyを用いている点が異なる
    def kmeans_plusplus(self, X):
        n_samples, n_features = X.shape
        # クラスタの中心点の集合
        centers = np.empty((self.n_clusters, n_features), dtype=np.float32)
        # 1イテレーションあたりの初期中心点の候補数
        n_local_trials = 2 + int(np.log(self.n_clusters))
        # 初期中心点候補のインデックスを1つ選ぶ
        center_id = self.random_state.choice(n_samples)
        # 初期値に選んだデータのインデックスを格納する配列を-1で初期化
        indices = np.full(self.n_clusters, -1, dtype=int)
        # 1つ目の初期中心点を代入
        centers[0] = X[center_id]
        indices[0] = center_id
        # 最近傍中心点との二乗ユークリッド距離を求める
        closest_dist_sq = cdist(
            centers[0, np.newaxis], X, "sqeuclidean"
        )  # (1, n_samples)
        # 最近傍中心点との二乗ユークリッド距離の総和
        current_pot = np.sum(closest_dist_sq)
        # 2つ目以降の初期中心点を逐次選ぶ
        for c in range(1, self.n_clusters):
            # 一様分布からn_local_trials個の値をサンプリングし、最近傍中心点との二乗ユークリッド距離の総和を係数としてかける
            rand_vals = self.random_state.uniform(size=n_local_trials) * current_pot
            # すでに中心点に選ばれたデータは最近傍中心点との二乗ユークリッド距離が0になるので、累積和を取ると同じ値が連続することになり、
            # searchsortedによってそれらのインデックスを避けて中心点候補をサンプリングできる　(a[i-1] < v <= a[i])
            # 一様分布を変数変換し、最近傍中心点との二乗ユークリッド距離によって重み付けした確率分布からサンプリングするのと同義
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            # out of range防止
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
            # 中心点候補とデータの距離を計算する
            distance_to_candidates = cdist(
                X[candidate_ids], X, "sqeuclidean"
            )  # (n_candidate_ids, n_samples)

            # 最近傍中心点との二乗ユークリッド距離を求める
            np.minimum(
                closest_dist_sq, distance_to_candidates, out=distance_to_candidates
            )
            candidates_pot = np.sum(distance_to_candidates, axis=-1)
            # 二乗ユークリッド距離が最小となるデータを中心点とする
            best_candidate = np.argmin(candidates_pot)
            # 各値を更新
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]
            centers[c] = X[best_candidate]
            indices[c] = best_candidate

        return centers, indices

    # smcfのハイパラを定義する
    def get_smcf_params(self, n_samples):
        # データのノード番号
        X_nodes = np.arange(n_samples)
        # 中心点のノード番号
        cluster_nodes = np.arange(n_samples, n_samples + self.n_clusters)
        # 最終的な需要を記述するために人工のノードを用意する
        artificial_demand_node = np.array([n_samples + self.n_clusters])
        # エッジの始点
        start_nodes = np.concatenate(
            [np.repeat(X_nodes, self.n_clusters), cluster_nodes]
        )
        # エッジの終点
        end_nodes = np.concatenate(
            [
                np.tile(cluster_nodes, n_samples),
                np.repeat(artificial_demand_node, self.n_clusters),
            ]
        )
        # エッジの容量
        # クラスタあたりの最大データ数 (max_membership) を指定可能
        capacities = np.concatenate(
            [
                np.ones(
                    self.n_clusters * n_samples,
                ),
                np.full(
                    self.n_clusters,
                    n_samples - self.n_clusters * self.min_membership
                    if self.max_membership is None
                    else self.max_membership - self.min_membership,
                ),
            ]
        )
        # ノードの供給 (マイナスは需要)
        # データの供給は1
        # 中心点の需要はクラスタあたりの最小データ数 (min_membership)
        supplies = np.concatenate(
            [
                np.ones(
                    n_samples,
                ),
                -1 * np.full(self.n_clusters, self.min_membership),
                np.array([-n_samples + self.n_clusters * self.min_membership]),
            ]
        ).tolist()
        return start_nodes, end_nodes, capacities, supplies

    # エッジのコスト（中心点との二乗ユークリッド距離）を計算する
    def calc_unit_costs(self, X, centers):
        dist_sq = cdist(X, centers, "sqeuclidean")  # (n_samples, n_clusters)
        unit_costs = np.concatenate([dist_sq.flatten(), np.zeros(self.n_clusters)])
        return unit_costs

    # 最小コストフロー問題を解き、各データが所属するクラスタをマスクの形で得る
    def clustering(self, start_nodes, end_nodes, capacities, supplies, unit_costs):
        smcf = min_cost_flow.SimpleMinCostFlow()
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
            start_nodes, end_nodes, capacities, unit_costs
        )
        smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)
        status = smcf.solve()
        solution_flows = smcf.flows(all_arcs)
        mask = solution_flows[: -self.n_clusters].reshape(
            (-1, self.n_clusters)
        )  # (n_samples, n_clusters)
        return mask

    # クラスタリングする
    def fit(self, X):
        self.valid_tr(X)
        # Xが1-Dの場合2-Dに変換する
        if X.ndim == 1:
            X = X[:, np.newaxis]
        n_samples, _ = X.shape
        # k-means++によってクラスタの初期中心を選ぶ
        centers, self.init_indices = self.kmeans_plusplus(X)
        # smcfのハイパラを得る
        params = self.get_smcf_params(n_samples)
        # エッジのコスト（中心点との二乗ユークリッド距離）を計算する
        unit_costs = self.calc_unit_costs(X, centers)
        for iter_ in range(self.max_iter):
            # 各データが所属するクラスタをマスクの形で得る
            mask = self.clustering(*params, unit_costs)  # (n_samples, n_clusters)
            # 中心点を計算する
            centers_ = np.dot(mask.T, X) / np.sum(mask, axis=0)[:, np.newaxis]
            # コストを更新する
            unit_costs = self.calc_unit_costs(X, centers_)
            # 中心点の更新前後の差の平方和を計算し、tol以下なら終了する
            centers_squared_diff = np.sum((centers_ - centers) ** 2)
            # 中心点を更新する
            centers = centers_
            if centers_squared_diff <= self.tol:
                break
        dist_sq = unit_costs[: -self.n_clusters].reshape(-1, self.n_clusters)
        self.inertia = np.sum(mask * dist_sq)  # クラスタ内二乗ユークリッド距離の総和
        self.centers = centers  # 中心点
        self.labels = np.argmax(mask, axis=-1)  # 各データの所属クラスタのラベル
        self.iter_ = iter_  # クラスタリングに要したイテレーション数
        return self

    # 未知のデータを既知の中心点によってクラスタリングし、ラベルを得る
    def predict(self, X):
        self.valid_pr(X)
        # Xが1-Dの場合2-Dに変換する
        if X.ndim == 1:
            X = X[:, np.newaxis]
        n_samples, _ = X.shape
        params = self.get_smcf_params(n_samples)
        unit_costs = self.calc_unit_costs(X, self.centers)
        mask = self.clustering(*params, unit_costs)
        labels = np.argmax(mask, axis=1)
        return labels

    # 初期化の引数を検証する
    def valid(
        self,
        n_clusters,
        min_membership,
        max_membership,
        random_state,
        max_iter,
        tol,
    ):
        if not isinstance(n_clusters, int):
            raise TypeError("n_clusters must be an integer.")
        if not isinstance(min_membership, int):
            raise TypeError("min_membership must be an integer.")
        if not isinstance(max_membership, int) and max_membership is not None:
            raise TypeError("max_membership must be an integer or None.")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")
        if not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if not isinstance(tol, int) and not isinstance(tol, float):
            raise TypeError("tol must be an integer or float.")

        if n_clusters < 2:
            raise ValueError("n_clusters must be 2 or more.")
        if min_membership < 0:
            raise ValueError("min_membership must be 0 or more.")
        if max_membership is not None:
            if max_membership < min_membership or max_membership < 0:
                raise ValueError("max_membership must be at least min_membership.")
        if random_state < 0:
            raise ValueError("random_state must be 0 or more.")
        if max_iter < 1:
            raise ValueError("max_iter must be 1 or more.")
        if tol < 0:
            raise ValueError("tol must be 0 or more.")

    # fitの引数を検証する
    def valid_tr(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be numpy.ndarray.")
        if X.ndim < 1 or X.ndim > 2:
            raise ValueError("X must be 1-D or 2-D.")
        if (
            X.shape[0] < self.n_clusters
            or X.shape[0] < self.n_clusters * self.min_membership
        ):
            raise ValueError("X must have at least n_clusters * min_membership rows.")
        if self.max_membership is not None:
            if X.shape[0] > self.n_clusters * self.max_membership:
                raise ValueError(
                    "X must have at most n_clusters * max_membership rows."
                )

    # predictの引数を検証する
    def valid_pr(self, X):
        if self.centers is None or self.labels is None:
            raise RuntimeError("fit method hasn't been called.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be numpy.ndarray.")
        if X.ndim < 1 or X.ndim > 2:
            raise ValueError("X must be 1-D or 2-D.")
        if (
            X.shape[0] < self.n_clusters
            or X.shape[0] < self.n_clusters * self.min_membership
        ):
            raise ValueError("X must have at least n_clusters * min_membership rows.")
        if self.max_membership is not None:
            if X.shape[0] > self.n_clusters * self.max_membership:
                raise ValueError(
                    "X must have at most n_clusters * max_membership rows."
                )
        if X.ndim == 2:
            if X.shape[-1] != self.centers.shape[-1]:
                raise ValueError("input shape does not match the centers shape.")
