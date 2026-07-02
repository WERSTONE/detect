import numpy as np
import torch

from test_model.eval import compute_pose_ap, evaluate


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
