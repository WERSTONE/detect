"""Final three-task model construction."""

from test_model.final.model.bifpn import DomainAttrBiFPN


def create_final_model(name="final_three_head", **kwargs):
    if name not in ("final_three_head", "bifpn_final_three_head"):
        raise ValueError(
            'Supported final models: "final_three_head", '
            '"bifpn_final_three_head".'
        )
    model = DomainAttrBiFPN(**kwargs)
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
