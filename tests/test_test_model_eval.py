import numpy as np
import torch

from test_model.eval import compute_all_metrics, compute_pose_ap, evaluate


def test_pose_ap_ignores_person_gt_without_visible_keypoints():
    visible_kpts = np.zeros((17, 3), dtype=np.float32)
    visible_kpts[0] = [20.0, 20.0, 2.0]
    invisible_kpts = np.zeros((17, 3), dtype=np.float32)

    predictions = [{
        'person_boxes': np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        'person_scores': np.array([0.9], dtype=np.float32),
        'person_kpts': visible_kpts.reshape(1, 17, 3),
    }]
    ground_truths = [{
        'boxes': np.array([
            [10.0, 10.0, 50.0, 50.0],
            [60.0, 60.0, 100.0, 100.0],
        ], dtype=np.float32),
        'classes': np.array([0, 0], dtype=np.int32),
        'kpts': np.stack([visible_kpts, invisible_kpts], axis=0),
    }]

    assert compute_pose_ap(predictions, ground_truths) == 1.0


def test_evaluate_handles_person_predictions_without_keypoints():
    class NoKeypointModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6):
            return [{
                'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=images.device),
                'scores': torch.tensor([0.9], device=images.device),
                'classes': torch.tensor([0], dtype=torch.long, device=images.device),
            }]

    batch = {
        'image': torch.zeros(1, 3, 64, 64),
        'boxes': [torch.tensor([[10.0, 10.0, 50.0, 50.0]])],
        'classes': [torch.tensor([0], dtype=torch.long)],
        'kpts': [torch.zeros(1, 17, 3)],
    }

    preds, gts = evaluate(NoKeypointModel(), [batch], device='cpu')

    assert preds[0]['person_kpts'].shape == (1, 17, 3)
    assert gts[0]['classes'].tolist() == [0]


def test_evaluate_selects_aligned_person_keypoints():
    class AlignedKeypointModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def predict_val(self, images, score_thresh=0.01, iou_thresh=0.6, max_det=300):
            kpts = torch.zeros(2, 17, 3, device=images.device)
            kpts[0, 0] = torch.tensor([20.0, 20.0, 0.8], device=images.device)
            kpts[1, 0] = torch.tensor([40.0, 40.0, 0.2], device=images.device)
            return [{
                'boxes': torch.tensor([
                    [10.0, 10.0, 50.0, 50.0],
                    [60.0, 60.0, 90.0, 90.0],
                ], device=images.device),
                'scores': torch.tensor([0.9, 0.8], device=images.device),
                'classes': torch.tensor([0, 2], dtype=torch.long, device=images.device),
                'kpts': kpts,
            }]

    batch = {
        'image': torch.zeros(1, 3, 64, 64),
        'boxes': [torch.tensor([[10.0, 10.0, 50.0, 50.0]])],
        'classes': [torch.tensor([0], dtype=torch.long)],
        'kpts': [torch.zeros(1, 17, 3)],
    }

    preds, _ = evaluate(AlignedKeypointModel(), [batch], device='cpu')

    assert preds[0]['person_kpts'].shape == (1, 17, 3)
    assert np.isclose(preds[0]['person_kpts'][0, 0, 2], 0.8)


def test_compute_all_metrics_includes_yolo_style_aliases():
    predictions = [{
        'boxes': np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        'scores': np.array([0.9], dtype=np.float32),
        'classes': np.array([1], dtype=np.int32),
        'person_boxes': np.zeros((0, 4), dtype=np.float32),
        'person_scores': np.zeros((0,), dtype=np.float32),
        'person_kpts': np.zeros((0, 17, 3), dtype=np.float32),
    }]
    ground_truths = [{
        'boxes': np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        'classes': np.array([1], dtype=np.int32),
        'kpts': np.zeros((1, 17, 3), dtype=np.float32),
    }]

    metrics = compute_all_metrics(predictions, ground_truths)

    assert metrics['mAP@0.5'] == 1.0
    assert metrics['mAP50'] == metrics['mAP@0.5']
    assert metrics['metrics/mAP50(B)'] == metrics['mAP@0.5']
    assert metrics['mAP50-95'] == metrics['mAP@0.5:0.95']
    assert 'per_class_AP@0.5:0.95' in metrics
