import torch

from test_model.final.model.loss import YOLODetectionLoss


def _run_loss(loss_fn, cls_seed, reg_seed, mask):
    cls = [value.detach().clone().requires_grad_(True) for value in cls_seed]
    reg = [value.detach().clone().requires_grad_(True) for value in reg_seed]
    gt = {
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
        "classes": torch.tensor([0]),
    }
    if mask is not None:
        gt["class_valid_mask"] = torch.tensor(mask)
    losses = loss_fn({"cls": cls, "reg": reg}, [gt])
    losses["total"].backward()
    return losses, torch.cat([value.grad.flatten() for value in cls])


def test_class_valid_mask_blocks_invalid_class_bce_and_gradients():
    torch.manual_seed(7)
    loss_fn = YOLODetectionLoss(num_classes=3, reg_max=16, strides=(8, 16, 32))
    cls_seed = [torch.randn(1, 3, 1, 1) for _ in range(3)]
    reg_seed = [torch.randn(1, 64, 1, 1) for _ in range(3)]

    all_losses, all_grads = _run_loss(loss_fn, cls_seed, reg_seed, [1, 1, 1])
    masked_losses, masked_grads = _run_loss(loss_fn, cls_seed, reg_seed, [1, 0, 0])

    assert masked_losses["det_cls"] < all_losses["det_cls"]
    assert torch.allclose(masked_grads.view(3, 3)[:, 1:], torch.zeros(3, 2))
    assert not torch.allclose(all_grads, masked_grads)


def test_all_ones_class_valid_mask_preserves_detection_loss():
    torch.manual_seed(11)
    loss_fn = YOLODetectionLoss(num_classes=3, reg_max=16, strides=(8, 16, 32))
    cls_seed = [torch.randn(1, 3, 1, 1) for _ in range(3)]
    reg_seed = [torch.randn(1, 64, 1, 1) for _ in range(3)]

    baseline, baseline_grads = _run_loss(loss_fn, cls_seed, reg_seed, None)
    masked, masked_grads = _run_loss(loss_fn, cls_seed, reg_seed, [1, 1, 1])

    assert torch.allclose(baseline["total"], masked["total"])
    assert torch.allclose(baseline["det_cls"], masked["det_cls"])
    assert torch.allclose(baseline_grads, masked_grads)


def test_class_mask_metrics_report_effective_supervision():
    torch.manual_seed(13)
    loss_fn = YOLODetectionLoss(num_classes=3, reg_max=16, strides=(8, 16, 32))
    cls_seed = [torch.randn(2, 3, 1, 1) for _ in range(3)]
    reg_seed = [torch.randn(2, 64, 1, 1) for _ in range(3)]
    gt = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
            "classes": torch.tensor([0]),
            "class_valid_mask": torch.tensor([1, 1, 0]),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
            "classes": torch.tensor([1]),
            "class_valid_mask": torch.tensor([0, 1, 1]),
        },
    ]

    losses = loss_fn({"cls": cls_seed, "reg": reg_seed}, gt)
    metrics = losses["_class_metrics"]

    assert torch.equal(metrics["valid_images"], torch.tensor([1.0, 2.0, 1.0]))
    assert torch.equal(metrics["valid_logits"], torch.tensor([3.0, 6.0, 3.0]))
    assert metrics["pos_anchors"][2] == 0
    assert torch.allclose(metrics["cls"].sum(), losses["det_cls"])
