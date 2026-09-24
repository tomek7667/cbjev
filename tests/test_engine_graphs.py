"""CUDA-graph cache eviction must never corrupt the graphs that stay (GPU only)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_many_shapes_with_a_tiny_graph_cache():
    import cbjev
    ag = cbjev.load(device="cuda")
    ag.engine.max_graphs = 2
    q = {"t": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "tech": "bugs"}},
         "n": {"type": "noul", "instructions": "Is the customer upset?"}}
    first = None
    for i in range(120):
        text = "I was charged twice for my order, please refund it. " * (1 + i % 17)
        qs = dict(list(q.items())[: 1 + i % 2])
        res = ag.predict(text, qs)
        torch.cuda.synchronize()
        if i == 0:
            first = res["answers"]["t"]["probabilities"]
    again = ag.predict("I was charged twice for my order, please refund it. ", {"t": q["t"]})
    assert abs(again["answers"]["t"]["probabilities"]["billing"] - first["billing"]) < 1e-3
