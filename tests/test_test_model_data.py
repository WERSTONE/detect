from test_model.data.prepare_coco20 import CLASS_MAP
from test_model.dataset import COCO_CATEGORY_ID_TO_20, COCO_CATEGORY_ID_TO_80, COCO80_CLASSES


def test_prepare_coco20_mapping_matches_training_dataset():
    assert CLASS_MAP == COCO_CATEGORY_ID_TO_20


def test_prepare_coco20_person_is_internal_zero():
    assert CLASS_MAP[1] == 0


def test_coco80_mapping_person_and_last_class():
    assert len(COCO80_CLASSES) == 80
    assert COCO_CATEGORY_ID_TO_80[1] == 0
    assert COCO_CATEGORY_ID_TO_80[90] == 79
