"""Small plotting helpers for graph views (Plotly)."""
import numpy as np
import plotly.graph_objects as go


def pca_layout(vectors) -> np.ndarray:
    """2-D positions from the first two principal components (deterministic)."""
    x = np.asarray(vectors, dtype=np.float64)
    x = x - x.mean(axis=0)
    if len(x) < 2:
        return np.zeros((len(x), 2))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords


def _lines(coords, edges):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    xs = np.full(3 * len(edges), np.nan)
    ys = np.full(3 * len(edges), np.nan)
    xs[0::3], xs[1::3] = coords[edges[:, 0], 0], coords[edges[:, 1], 0]
    ys[0::3], ys[1::3] = coords[edges[:, 0], 1], coords[edges[:, 1], 1]
    return xs, ys


def network_figure(coords, edges, highlight=None, node_colors="#22D3EE",
                   node_hover=None, node_size=10, height=560):
    """Draw nodes and edges. `highlight` is a boolean array over `edges`."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    highlight = np.zeros(len(edges), dtype=bool) if highlight is None else np.asarray(highlight, dtype=bool)

    fig = go.Figure()
    faint_x, faint_y = _lines(coords, edges[~highlight])
    fig.add_trace(go.Scatter(x=faint_x, y=faint_y, mode="lines", hoverinfo="skip",
                             line=dict(color="rgba(148,163,184,0.35)", width=1)))
    if highlight.any():
        hx, hy = _lines(coords, edges[highlight])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", hoverinfo="skip",
                                 line=dict(color="#22D3EE", width=3.5)))
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers",
        marker=dict(size=node_size, color=node_colors, line=dict(width=1, color="#0B1220")),
        hovertext=node_hover, hoverinfo="text" if node_hover is not None else "skip"))
    fig.update_layout(template="plotly_dark", showlegend=False, height=height,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    return fig