import torch
from timm.layers import trunc_normal_
from torch import nn

from ...blocks.video.features import VideoFeatures
from ...blocks.video.memory import VideoMemory
from ...blocks.video.tracking import NUM_MULTIMASK_OUTPUTS, VideoTracking
from ...components.video.tracker.decoder import heads as decoder_heads
from ...components.video.tracker.frame import (
    features as frame_features,
    inference as frame_inference,
    memory as frame_memory,
    output as frame_output,
)
from ...components.video.tracker.interaction import dynamic_masks, mask_as_output
from ...components.video.tracker.memory import (
    conditioning as memory_conditioning,
    encoding as memory_encoding,
)
from ...components.video.tracker.multiplex.state import MultiplexController
from ...components.video.tracker.runtime import step as runtime_step
from . import masks, objects, propagate
from .heads import build_sam_heads
from .state import add_object, create_state

IMAGE_SIZE = 1008
BACKBONE_STRIDE = 14


class VideoRuntime(nn.Module):
    def __init__(self, features, memory, tracking, multiplex_controller):
        super().__init__()
        self.backbone = features
        self.use_high_res_features_in_sam = True
        self.num_feature_levels = 3
        self.use_obj_ptrs_in_encoder = True
        self.max_obj_ptrs_in_encoder = 16
        self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.multiplex_controller = multiplex_controller
        self.save_image_features = True
        self.multiplex_count = multiplex_controller.multiplex_count

        assert (
            tracking.transformer.decoder is None
        ), "transformer should be encoder-only"
        self.transformer = tracking.transformer
        self.hidden_dim = self.transformer.d_model
        self.maskmem_backbone = memory.encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            assert (
                self.maskmem_backbone.out_proj.weight.shape[0] == self.hidden_dim
            ), "there should be no compression of memory embeddings"
        self.num_maskmem = 7
        self.sincos_tpos_enc = True
        self.use_maskmem_tpos_v2 = True
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.interactivity_no_mem_embed = nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim)
        )
        trunc_normal_(self.interactivity_no_mem_embed, std=0.02)
        self.directly_add_no_mem_embed = True

        self.pred_obj_scores = True
        self.pred_obj_scores_mlp = True
        self.fixed_no_obj_ptr = True
        self.use_no_obj_ptr = True
        self.use_linear_no_obj_ptr = True
        self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.use_mlp_for_obj_ptr_proj = True
        self.no_obj_embed_spatial = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        self.add_output_suppression_embeddings = True
        self.output_valid_embed = tracking.output_valid_embed
        self.output_invalid_embed = tracking.output_invalid_embed
        self.add_object_conditional_embeddings = False
        self.add_object_unconditional_embeddings = False
        self.condition_as_mask_input = True
        self.condition_as_mask_input_fg = 1.0
        self.condition_as_mask_input_bg = 0.0

        self.image_size = IMAGE_SIZE
        self.backbone_stride = BACKBONE_STRIDE
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        self.input_mask_size = self.low_res_mask_size * 4
        self.add_tpos_enc_to_obj_ptrs = True
        self.proj_tpos_enc_in_obj_ptrs = True
        self.use_signed_tpos_enc_to_obj_ptrs = False
        self.only_obj_ptrs_in_the_past_for_eval = False
        self.apply_sigmoid_to_mask_logits_for_mem_enc = True
        self.sigmoid_scale_for_mem_enc = 2.0
        self.sigmoid_bias_for_mem_enc = -1.0
        self.binarize_mask_from_pts_for_mem_enc = False
        self.non_overlap_masks_for_mem_enc = False
        self.memory_temporal_stride_for_eval = 1
        self.use_mask_input_as_output_without_sam = True
        self.multimask_output_in_sam = True
        self.multimask_min_pt_num = 0
        self.multimask_max_pt_num = 1
        self.multimask_output_for_tracking = True
        self.use_best_iou_mask_for_mem_enc = False
        self.iou_prediction_use_sigmoid = False
        self.object_score_logit_threshold = 0.0
        self.stability_score_attentuation = False
        self.iter_use_prev_mask_pred = False
        self.interactive_sam_mask_decoder_extra_args = {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }
        self.sam_mask_decoder_extra_args = {
            **self.interactive_sam_mask_decoder_extra_args,
            "dynamic_multimask_via_stability": False,
        }
        self.num_multimask_outputs = NUM_MULTIMASK_OUTPUTS
        self.decode_mask_with_shared_tokens = False
        self.decode_mask_attribute_with_shared_tokens = False
        self.share_necks = False
        self.offload_output_to_cpu_for_eval = False
        self.trim_past_non_cond_mem_for_eval = False
        self.max_cond_frames_in_attn = 4
        self.keep_first_cond_frame = False
        self.add_all_frames_to_correct_as_cond = False
        self.use_memory_selection = False
        self.mf_threshold = 0.01
        self.is_dynamic_model = True
        self.image_pe_layer = tracking.image_pe
        self.sam_mask_decoder = tracking.mask_decoder
        build_sam_heads(self)
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.fill_hole_area = 0
        self.always_start_from_first_ann_frame = False
        self.non_overlap_masks_for_output = True

    def _get_interactive_pix_mem(self, *args, **kwargs):
        return frame_features.get_interactive_pix_mem(self, *args, **kwargs)

    def _forward_sam_heads(self, *args, **kwargs):
        return decoder_heads.forward_sam_heads(self, *args, **kwargs)

    def _use_mask_as_output(self, *args, **kwargs):
        return mask_as_output.use_mask_as_output(self, *args, **kwargs)

    def forward_image(self, *args, **kwargs):
        return frame_features.forward_image(self, *args, **kwargs)

    def _prepare_backbone_features(self, *args, **kwargs):
        return frame_features.prepare_backbone_features(self, *args, **kwargs)

    def _prepare_backbone_features_per_frame(self, *args, **kwargs):
        return frame_features.prepare_backbone_features_per_frame(self, *args, **kwargs)

    def _prepare_memory_conditioned_features(self, *args, **kwargs):
        return memory_conditioning.prepare_memory_conditioned_features(
            self, *args, **kwargs
        )

    def _encode_new_memory(self, *args, **kwargs):
        return memory_encoding.encode_new_memory(self, *args, **kwargs)

    def _trim_output_and_memory(self, *args, **kwargs):
        return frame_output.trim_output_and_memory(self, *args, **kwargs)

    def score_memory(self, *args, **kwargs):
        return frame_output.score_memory(self, *args, **kwargs)

    def add_new_masks_to_existing_state(self, *args, **kwargs):
        return dynamic_masks.add_new_masks_to_existing_state(self, *args, **kwargs)

    def recondition_masks_in_existing_state(self, *args, **kwargs):
        return dynamic_masks.recondition_masks_in_existing_state(self, *args, **kwargs)

    def track_step(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        image,
        mask_inputs,
        output_dict,
        num_frames,
        run_mem_encoder=True,
        multiplex_state,
    ):
        current_out = runtime_step.run_track_step(
            self,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            run_mem_encoder=run_mem_encoder,
            multiplex_state=multiplex_state,
        )
        return self._trim_output_and_memory(
            frame_idx=frame_idx,
            output_dict=output_dict,
            current_out=current_out,
            memory_encoder_was_used=run_mem_encoder,
        )

    def _use_multimask(self, is_init_cond_frame):
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and self.multimask_min_pt_num <= 0 <= self.multimask_max_pt_num
            and self.num_multimask_outputs > 0
        )

    def _apply_non_overlapping_constraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks
        max_indices = torch.argmax(pred_masks, dim=0, keepdim=True)
        indices = torch.arange(batch_size, device=pred_masks.device)[
            :, None, None, None
        ]
        return torch.where(
            max_indices == indices,
            pred_masks,
            torch.clamp(pred_masks, max=-10.0),
        )

    def init_state(
        self,
        video_height,
        video_width,
        num_frames,
        cached_features=None,
        device="cuda",
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        state = create_state(
            cached_features=cached_features,
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            device=device,
        )
        return state

    def _obj_id_to_idx(self, state, obj_id, error_if_new=False):
        index = state["obj_id_to_idx"].get(obj_id)
        if index is not None:
            return index
        if error_if_new:
            raise RuntimeError(f"object id {obj_id} is not registered")
        return add_object(state, obj_id)

    def _get_obj_num(self, state):
        multiplex_state = state["multiplex_state"]
        return 0 if multiplex_state is None else multiplex_state.total_valid_entries

    def add_masks(self, *args, **kwargs):
        return masks.add_masks(self, *args, **kwargs)

    def _get_orig_video_res_output(self, state, masks):
        masks = masks.to(state["device"], non_blocking=True)
        size = (state["video_height"], state["video_width"])
        if masks.shape[-2:] != size:
            video_masks = torch.nn.functional.interpolate(
                masks,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            video_masks = masks
        if self.non_overlap_masks_for_output:
            video_masks = self._apply_non_overlapping_constraints(video_masks)
        return masks, video_masks

    def propagate_in_video_preflight(self, *args, **kwargs):
        return propagate.preflight(self, *args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        return propagate.propagate(self, *args, **kwargs)

    def _add_output_per_object(self, state, frame_idx, current_out, storage_key):
        local_indices = current_out.get(
            "local_obj_id_to_idx",
            state["obj_id_to_idx"],
        )
        for obj_id, obj_idx in state["obj_id_to_idx"].items():
            local_idx = local_indices.get(obj_id)
            if local_idx is None:
                continue
            output = state["output_dict_per_obj"][obj_idx]
            item = {
                "pred_masks": current_out["pred_masks"][local_idx : local_idx + 1],
                "object_score_logits": current_out["object_score_logits"][
                    local_idx : local_idx + 1
                ],
            }
            if self.use_memory_selection:
                item["iou_score"] = current_out["iou_score"][local_idx : local_idx + 1]
            output[storage_key][frame_idx] = item

    def _get_image_feature(self, *args, **kwargs):
        return frame_features.get_image_feature(self, *args, **kwargs)

    def _run_single_frame_inference(self, *args, **kwargs):
        return frame_inference.run_single_frame_inference(self, *args, **kwargs)

    def _run_memory_encoder(self, *args, **kwargs):
        return frame_memory.run_memory_encoder(self, *args, **kwargs)

    def _get_maskmem_pos_enc(self, *args, **kwargs):
        return frame_features.get_maskmem_pos_enc(self, *args, **kwargs)

    def remove_objects(self, *args, **kwargs):
        return objects.remove_objects(self, *args, **kwargs)

    def remove_object(self, state, obj_id, **kwargs):
        return self.remove_objects(state, [obj_id], **kwargs)

    def _clear_non_cond_mem_around_input(self, *args, **kwargs):
        return objects.clear_non_cond_mem_around_input(self, *args, **kwargs)


def create_runtime():
    features = VideoFeatures()
    memory = VideoMemory()
    tracking = VideoTracking()
    controller = MultiplexController(16, eval_multiplex_count=16)
    return VideoRuntime(features, memory, tracking, controller)
