import pytest
import torch

from finetune.loss import compute_loss_with_mask


def loss(logits, target, mask):
    return compute_loss_with_mask(logits, target, mask, mode="audio")


def test_masked_nan_has_finite_gradients():
    logits = torch.tensor([[[[1., 2.], [float('nan'), float('nan')]]]], requires_grad=True)
    target = torch.tensor([[[1, -1]]])
    result = loss(logits, target, target != -1)
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 1].eq(0).all()


def test_empty_mask_returns_differentiable_zero():
    logits = torch.full((1, 1, 2, 3), float('nan'), requires_grad=True)
    result = loss(logits, torch.full((1, 1, 2), -1), torch.zeros(1, 1, 2, dtype=torch.bool))
    assert result == 0
    result.backward()
    assert logits.grad.eq(0).all()


def test_unmasked_nan_fails_loudly():
    with pytest.raises(FloatingPointError):
        loss(torch.full((1, 1, 1, 2), float('nan')), torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 1, 1, dtype=torch.bool))


def test_evaluation_counts_processed_batches_and_empty_error(monkeypatch):
    from types import SimpleNamespace
    import finetune.eval as evaluation
    from finetune.data.interleaver import Batch
    class Model(torch.nn.Module):
        audio_offset = 1
        dep_q = 1
        text_padding_token_id = 3
        end_of_text_padding_id = 0
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def forward(self, codes, condition_tensors=None):
            return SimpleNamespace(text_logits=torch.zeros(1, 1, 2, 4),
                                   logits=torch.zeros(1, 1, 2, 4),
                                   text_mask=torch.ones(1, 1, 2, dtype=torch.bool),
                                   mask=torch.ones(1, 1, 2, dtype=torch.bool))
    monkeypatch.setattr(evaluation, 'get_world_size', lambda: 1)
    monkeypatch.setattr(evaluation, 'get_rank', lambda: 0)
    monkeypatch.setattr(evaluation.dist, 'all_reduce', lambda *args, **kwargs: None)
    args = SimpleNamespace(text_padding_weight=.01, first_codebook_weight_multiplier=2.)
    model, state = Model(), SimpleNamespace()
    batch = Batch(torch.ones(1, 2, 2, dtype=torch.long))
    for _ in range(2):
        evaluation.evaluate(model, iter([batch] * 45), state, args)
        assert state.this_eval_loss == pytest.approx(2 * torch.log(torch.tensor(4.)).item())
        assert state.this_eval_perplexity == pytest.approx(16., rel=1e-5)
        assert model.training
    with pytest.raises(ValueError, match='no batches'):
        evaluation.evaluate(model, iter([]), state, args)
    assert model.training
