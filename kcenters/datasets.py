import numpy as np
from sklearn import datasets as sk_datasets


def gerar_colecao_sklearn(random_state=0):
    """Gera vários tipos de datasets. Retorna lista de tuplas (X, k, meta, y).
    """
    res = []
    rng = np.random.RandomState(random_state)

    # 1) BLOBS
    for i in range(5):
        centers = rng.randint(2, 8)
        X, y = sk_datasets.make_blobs(
            n_samples=800,
            centers=centers,
            cluster_std=0.5 + 0.5 * i,
            random_state=rng,
        )
        k = len(np.unique(y))
        meta = {
            'tipo': 'blobs',
            'var': i,
            'name': f'blobs_{i}',
        }
        res.append((X.astype(float), k, meta, y))

    # 2) MOONS
    for i in range(5):
        X, y = sk_datasets.make_moons(
            n_samples=800,
            noise=0.05 + 0.05 * i,
            random_state=rng,
        )
        k = len(np.unique(y))
        meta = {
            'tipo': 'moons',
            'var': i,
            'name': f'moons_{i}',
        }
        res.append((X.astype(float), k, meta, y))

    # 3) CIRCLES
    for i in range(5):
        X, y = sk_datasets.make_circles(
            n_samples=800,
            noise=0.01 + 0.02 * i,
            factor=0.3 + 0.1 * (i % 2),
            random_state=rng,
        )
        k = len(np.unique(y))
        meta = {
            'tipo': 'circles',
            'var': i,
            'name': f'circles_{i}',
        }
        res.append((X.astype(float), k, meta, y))

    # 4) VARIED
    for i in range(5):
        centers = rng.randint(3, 7)
        stds = 0.3 + rng.rand(centers) * (0.5 + 0.2 * i)
        X, y = sk_datasets.make_blobs(
            n_samples=800,
            centers=centers,
            cluster_std=stds,
            random_state=rng,
        )
        k = len(np.unique(y))
        meta = {
            'tipo': 'varied',
            'var': i,
            'name': f'varied_{i}',
        }
        res.append((X.astype(float), k, meta, y))

    # 5) ANISOTROPIC
    for i in range(5):
        X, y = sk_datasets.make_blobs(
            n_samples=800,
            centers=4,
            random_state=rng,
        )
        transformation = np.array([
            [0.6 + 0.2 * i, -0.4],
            [-0.3, 0.8 + 0.1 * i],
        ])
        X = X.dot(transformation)
        k = len(np.unique(y))
        meta = {
            'tipo': 'anisotropic',
            'var': i,
            'name': f'anisotropic_{i}',
        }
        res.append((X.astype(float), k, meta, y))

    # 6) UNIFORM (sem rótulos verdadeiros)
    for i in range(5):
        X = rng.rand(800, 2) * (10 + i)
        k = 3
        meta = {
            'tipo': 'uniform',
            'var': i,
            'name': f'uniform_{i}',
        }
        # não existe rótulo “real” natural -> usamos None
        y = None
        res.append((X.astype(float), k, meta, y))

    return res


def gerar_normais_multivariadas(
    k=4,
    n_por_cluster=200,
    dims=2,
    overlap_factor=1.0,
    rng=None,
):
    """Gera dataset em 2D com k clusters gaussianos, controlando sobreposição e anisotropia.
    Retorna X, labels
    """
    if rng is None:
        rng = np.random.RandomState(0)

    mus = []
    covs = []
    for i in range(k):
        angle = rng.rand() * 2 * np.pi
        r = 5.0 + rng.randn() * 0.5 + i * 0.2
        mu = np.array([r * np.cos(angle), r * np.sin(angle)])
        mus.append(mu)

        A = rng.randn(dims, dims)
        base = np.dot(A, A.T)
        cov = base / np.trace(base) * (0.5 + overlap_factor)
        covs.append(cov)

    points = []
    labels = []
    for idx, (mu, cov) in enumerate(zip(mus, covs)):
        pts = rng.multivariate_normal(mu, cov, size=n_por_cluster)
        points.append(pts)
        labels.append(np.full(n_por_cluster, idx, dtype=int))

    X = np.vstack(points)
    y = np.concatenate(labels)
    return X, y
