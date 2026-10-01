import torch

from . import prompt as prompt_data


def build_prompts(
    items: list[dict],
    image_size: int,
    mask_size: tuple[int, int],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
    prompts = [
        prompt_data.build_prompt(
            item["points"],
            item["point_labels"],
            item["box"],
            item["mask"],
            (image_size, image_size),
            image_size,
            mask_size,
            device,
        )
        for item in items
    ]
    points = (
        torch.cat([prompt[0][0] for prompt in prompts]),
        torch.cat([prompt[0][1] for prompt in prompts]),
    )
    if prompts[0][1] is None:
        masks = None
    else:
        masks = torch.cat([prompt[1] for prompt in prompts])
    return points, masks
