import os
import traceback
from typing import Dict

import re
import torch
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer
from .gsplat_renderer import GSPlatRenderer, DEFAULT_ANTI_ALIASED_STATUS, DEFAULT_BLOCK_SIZE
from .gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer
from ..cameras import Camera, Cameras
from ..models.gaussian import GaussianModel
from internal.utils.seganygs import ScaleGateUtils, SegAnyGSUtils
import torchvision
import numpy as np
import viser.transforms as vtf
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.rotation import rotmat2qvec
import torch.nn.functional as F
import math
from plyfile import PlyData, PlyElement

class SegAnyGSRenderer(Renderer):
    def __init__(
            self,
            semantic_features: torch.Tensor,
            scale_gate: torch.nn.Module,
            anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS,
            train_cam_poses: list = None,
            gaussian_model: torch.Tensor = None,
            bg_color: torch.Tensor = None,
            dataset_path: str = None,
    ):
        super().__init__()

        self.bg_color = bg_color

        # Load train camera poses to be used for finding which camera can see the segmented object
        if dataset_path is not None:
            from internal.dataparsers.colmap_dataparser import ColmapDataParser, Colmap

            parser_config = Colmap(
                image_dir="images",
                split_mode="reconstruction",
                scene_scale=1.0,
                force_pinhole=True
            )

            parser = ColmapDataParser(
                path=dataset_path,
                output_path="",
                global_rank=0,
                params=parser_config
            )

            outputs = parser.get_outputs()

            self.all_train_cameras = outputs.train_set.cameras
            self.original_img_names = outputs.train_set.image_names

        self.anti_aliased = anti_aliased

        self.initial_scale = 1.
        self.gaussian_model = gaussian_model
        self.duplicated_indices = None
        self.original_duplicated_means = None
        self.original_duplicated_scales = None
        self.last_rotation_matrix = None

        # move to cuda first
        self.semantic_features = semantic_features.cuda()
        self.scale_gate = ScaleGateUtils(scale_gate.cuda())

        self.scale_conditioned_semantic_features = SegAnyGSUtils.get_scale_conditioned_semantic_features(self.semantic_features, self.scale_gate, self.initial_scale)

        # PCA
        normalized_semantic_features = torch.nn.functional.normalize(self.semantic_features, dim=-1)
        self.pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(normalized_semantic_features)
        self.pca_colors = SegAnyGSUtils.get_pca_projected_colors(normalized_semantic_features, self.pca_projection_matrix)
        # scale conditioned PCA
        self.scale_conditioned_pca_projection_matrix = torch.nn.functional.normalize(self.pca_projection_matrix * self.scale_gate(self.initial_scale).unsqueeze(-1).to(self.pca_projection_matrix.device), dim=-1)
        self.scale_gated_pca_colors = SegAnyGSUtils.get_pca_projected_colors(
            self.scale_conditioned_semantic_features,
            self.scale_conditioned_pca_projection_matrix,
        )

        self.segment_mask = None
        self.similarities = None

        self.cluster_color = None
        self.cluster_result = None

        # reduce CUDA memory consumption
        # self.semantic_features = self.semantic_features.cpu()  # slow scale update a little
        self.scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cpu()
        # torch.cuda.empty_cache()

        self.color_producers = {
            "rgb": self._shs_to_rgb,
            "depth": self._depth_as_color,
            "pca2d": self._semantic_features_as_color,
            "pca3d": self._pca_as_color,
            "scale_gated_pca2d": self._scale_gated_semantic_features_as_color,
            "scale_gated_pca3d": self._scale_gated_pca_as_color,
            "cluster3d": self._cluster_as_color,
            "segment3d": self._segment_as_color,
            "segment3d_out": self._segment_out,
            "segment3d_removed": self._segment_removed,
            "segment3d_similarities": self._segment_similarities_as_color
        }

        self.available_output_types = {
            "rgb": "rgb",
            "depth": "depth",
            "pca2d": "semantic_features",
            "pca3d": "pca3d",
            "scale_gated_pca2d": "semantic_features_scale_gated",
            "scale_gated_pca3d": "pca3d_scale_gated",
            "cluster3d": "cluster3d",
            "segment3d": "segment3d",
            "segment3d_out": "segment3d_out",
            "segment3d_removed": "segment3d_removed",
            "segment3d_similarities": "segment3d_similarities"
        }

        self.output_post_processor = {
            "pca2d": self._get_pca_projected_color,
            "scale_gated_pca2d": self._get_scale_gated_pca_projected_color,
        }

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        pc = self.gaussian_model

        if hasattr(self, "duplicated_indices") and self.duplicated_indices is not None:
            offset = self.viewer_options.offset_vector.to('cuda')
            print(f"Applying offset {offset.tolist()} to {len(self.duplicated_indices)} duplicated Gaussians")
            xyz = pc.get_xyz.clone()
            xyz[self.duplicated_indices] += offset
        else:
            xyz = pc.get_xyz
        
        project_results = GSPlatRenderer.project(
            means3D=xyz,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
        )

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            comp = project_results[4]
            opacities = opacities * comp[:, None]

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        outputs = {}
        for i in render_types:
            colors, rasterize_bg_color, new_opacities = self.color_producers[i](project_results, pc, viewpoint_camera, bg_color, opacities)
            outputs[self.available_output_types[i]] = self.rasterize(project_results, img_height=img_height, img_width=img_width, colors=colors, bg_color=rasterize_bg_color, opacities=new_opacities)
            output_processor = self.output_post_processor.get(i)
            if output_processor is not None:
                outputs[self.available_output_types[i]] = output_processor(outputs[self.available_output_types[i]])

        
        return outputs

    def rasterize(self, project_results, img_height, img_width, colors, bg_color, opacities):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_results

        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            colors,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=DEFAULT_BLOCK_SIZE,
            background=bg_color,
            return_alpha=False,
        ).permute(2, 0, 1)  # type: ignore

    def _shs_to_rgb(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        means3D = pc.get_xyz
        viewdirs = means3D.detach() - viewpoint_camera.camera_center  # (N, 3)
        # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        return rgbs, bg_color, opacities

    def _depth_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return project_results[1].unsqueeze(-1), torch.zeros((1,), dtype=torch.float, device=bg_color.device), opacities

    def _semantic_features_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.semantic_features, torch.zeros((self.semantic_features.shape[-1],), dtype=torch.float, device=bg_color.device), opacities

    def _get_pca_projected_color(self, feature_map):
        return SegAnyGSUtils.get_pca_projected_colors(
            semantic_features=torch.nn.functional.normalize(feature_map.permute(1, 2, 0).view(-1, feature_map.shape[0]), dim=-1),
            pca_projection_matrix=self.pca_projection_matrix,
        ).view(*feature_map.shape[1:], 3).permute(2, 0, 1)

    def _pca_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.pca_colors, bg_color, opacities

    def _scale_gated_semantic_features_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.scale_conditioned_semantic_features.to(bg_color.device), torch.zeros((self.scale_conditioned_semantic_features.shape[-1],), dtype=torch.float, device=bg_color.device), opacities

    def _get_scale_gated_pca_projected_color(self, feature_map):
        return SegAnyGSUtils.get_pca_projected_colors(
            semantic_features=torch.nn.functional.normalize(feature_map.permute(1, 2, 0).view(-1, feature_map.shape[0]), dim=-1),
            pca_projection_matrix=self.scale_conditioned_pca_projection_matrix,
        ).view(*feature_map.shape[1:], 3).permute(2, 0, 1)

    def _scale_gated_pca_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.scale_gated_pca_colors, bg_color, opacities

    def _cluster_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        if self.cluster_color is None:
            # TODO: fix cluster twice sometimes
            try:
                self.viewer_options.print_cluster_start_message()
            except:
                pass
            self.cluster_in_3d()
            try:
                self.viewer_options.print_cluster_finished_message()
            except:
                pass

        return self.cluster_color, bg_color, opacities

    def _segment_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        # if self.segment_mask is not None:
        if self.segment_mask is not None and self.segment_mask.shape[0] == colors.shape[0]:
            colors[self.segment_mask] = torch.tensor([0., 1., 1.], dtype=torch.float, device=bg_color.device)
        return colors, bg_color, opacities

    def _segment_out(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            opacities = opacities * self.segment_mask.unsqueeze(-1)
        return colors, bg_color, opacities

    def _segment_removed(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            opacities = opacities * (~self.segment_mask).unsqueeze(-1)
        return colors, bg_color, opacities
    
    def lock_current_offset(self):
        if self.duplicated_indices is None:
            print("No duplicated indices to lock.")
            return

        device = self.gaussian_model.get_property("means").device
        offset = self.viewer_options.offset_vector.to(device)

        # Apply the offset permanently
        means = self.gaussian_model.get_property("means")
        means[self.duplicated_indices] += offset
        self.gaussian_model.set_property("means", means)

        # Clear duplicated_indices so future offset doesn't apply to these
        self.duplicated_indices = None

        # Reset the live offset (optional but nice UX)
        self.viewer_options.offset_vector = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.viewer_options.viewer.rerender_for_all_client()

        print("Locked offset for duplicated Gaussians.")

    def _segment_similarities_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        if self.similarities is not None:
            return self.similarities.unsqueeze(-1), torch.zeros((1, 1), dtype=torch.float, device=opacities.device), opacities
        return torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float, device=opacities.device), bg_color, opacities

    def cluster_in_3d(self):
        self.cluster_result = SegAnyGSUtils.cluster_3d_as_dict(self.scale_conditioned_semantic_features)
        self.cluster_color = torch.tensor(self.cluster_result["point_colors"], dtype=torch.float, device="cuda")

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        # with tabs.add_tab("Semantic"):
        #     self.viewer_options = ViewerOptions(self, viewer, server, initial_scale=self.initial_scale)
        #  with server.gui.add_folder("Selection"):
        self.viewer_options = ViewerOptions(self, viewer, server, initial_scale=self.initial_scale)


    def get_available_outputs(self) -> Dict:
        available_outputs = {}
        for i in self.available_output_types:
            available_outputs[i] = RendererOutputInfo(self.available_output_types[i], type=RendererOutputTypes.GRAY if self.is_type_depth_map(i) else RendererOutputTypes.RGB)

        return available_outputs

    def is_type_depth_map(self, t: str) -> bool:
        return t == "depth" or t == "segment3d_similarities"


    def scale_selected_gaussians(self, factor: float):
        if self.duplicated_indices is None:
            print("No duplicated gaussians to scale.")
            return
        if not hasattr(self, "original_duplicated_means") or not hasattr(self, "original_duplicated_scales"):
            print("Original values not saved. Cannot apply relative scaling.")
            return

        indices = self.duplicated_indices
        device = self.gaussian_model.get_property("means").device

        # Fetch original data
        original_means = self.original_duplicated_means.to(device)
        original_scales = self.original_duplicated_scales.to(device)
        centroid = original_means.mean(dim=0, keepdim=True)

        # Compute scaled positions and scales
        # new_means = (original_means - centroid) * factor + centroid
        # if not hasattr(self, "last_rotation_matrix"):
        #     self.last_rotation_matrix = torch.eye(3, device=device)  # Fallback to identity

        # means_local = (original_means - centroid)
        # means_local_scaled = means_local * factor
        # new_means = means_local_scaled @ self.last_rotation_matrix.T + centroid
        # new_scales = original_scales * factor
        new_means = (original_means - centroid) * factor + centroid
        new_scales = original_scales * factor

        # Apply to model
        means = self.gaussian_model.get_property("means")
        scales = self.gaussian_model.get_property("scales")
        means[indices] = new_means
        scales[indices] = new_scales
        self.gaussian_model.set_property("means", means)
        self.gaussian_model.set_property("scales", scales)

        print(f"Set object scale to {factor}x of original.")

    def get_rotation_matrix(self, x_rad, y_rad, z_rad, device):
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(x_rad), -torch.sin(x_rad)],
            [0, torch.sin(x_rad),  torch.cos(x_rad)]
        ], device=device)

        Ry = torch.tensor([
            [ torch.cos(y_rad), 0, torch.sin(y_rad)],
            [0, 1, 0],
            [-torch.sin(y_rad), 0, torch.cos(y_rad)]
        ], device=device)

        Rz = torch.tensor([
            [torch.cos(z_rad), -torch.sin(z_rad), 0],
            [torch.sin(z_rad),  torch.cos(z_rad), 0],
            [0, 0, 1]
        ], device=device)

        return Rz @ Ry @ Rx  # Apply in ZYX order
    
    def rotate_selected_gaussians(self, x_rad=0.0, y_rad=0.0, z_rad=0.0):
        if self.duplicated_indices is None:
            print("No duplicated gaussians to rotate.")
            return

        device = self.gaussian_model.get_property("means").device
        indices = self.duplicated_indices

        # Create rotation matrix from Euler angles
        # Rx = vtf.so3_x(torch.tensor(x_rad, device=device))
        # Ry = vtf.so3_y(torch.tensor(y_rad, device=device))
        # Rz = vtf.so3_z(torch.tensor(z_rad, device=device))
        # R = Rz @ Ry @ Rx  # Combined rotation matrix (ZYX)
        x_rad, y_rad, z_rad = torch.tensor(x_rad, device=device), torch.tensor(y_rad, device=device), torch.tensor(z_rad, device=device)
        R = self.get_rotation_matrix(x_rad, y_rad, z_rad, device=device)

        
        # --- 1. Rotate positions around the object's centroid ---
        means = self.gaussian_model.get_property("means")
        selected_means = means[indices]
        centroid = selected_means.mean(dim=0, keepdim=True)
        rotated_means = ((selected_means - centroid) @ R.T) + centroid
        means[indices] = rotated_means
        self.gaussian_model.set_property("means", means)

        # --- 2. Rotate the orientations (quaternions) ---
        rotations = self.gaussian_model.get_property("rotations")[indices]  # (N, 4)
        
        # Convert rotation matrix to quaternion
        from internal.utils.rotation import rotmat2qvec  # your earlier util
        R_q = rotmat2qvec(R).expand_as(rotations)  # (N, 4), repeat for each Gaussian

        # Quaternion multiplication: q_new = R_q * q_old
        w1, x1, y1, z1 = [t.to(device=device) for t in torch.split(R_q, 1, dim=1)]
        w2, x2, y2, z2 = [t.to(device=device) for t in torch.split(rotations, 1, dim=1)]

        q_new = torch.cat([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=1)

        # Normalize and assign
        q_new = q_new / q_new.norm(dim=1, keepdim=True)
        all_rotations = self.gaussian_model.get_property("rotations")
        all_rotations[indices] = q_new
        self.gaussian_model.set_property("rotations", all_rotations)
        self.last_rotation_matrix = R
        self.original_duplicated_means = ((self.original_duplicated_means - centroid) @ R.T) + centroid
        print("Applied rotation to selected Gaussians (positions + orientations).")

    def add_selected_gaussians(self, offset=None):
        if not isinstance(self.segment_mask, torch.Tensor) or self.segment_mask.sum() == 0:
            print("No gaussians selected.")
            return None

        device = self.gaussian_model.get_property("means").device
        mask = self.segment_mask.to(device)
        indices = torch.where(mask)[0]

        means = self.gaussian_model.get_property("means")[indices]
        opacities = self.gaussian_model.get_property("opacities")[indices]
        scales = self.gaussian_model.get_property("scales")[indices]
        rotations = self.gaussian_model.get_property("rotations")[indices]
        shs = self.gaussian_model.get_property("shs")[indices]

        # No offset here
        new_start = self.gaussian_model.get_property("means").shape[0]
        new_end = new_start + means.shape[0]
        new_indices = torch.arange(new_start, new_end, device=device)

        self.gaussian_model.set_property("means", torch.cat([self.gaussian_model.get_property("means"), means], dim=0))
        self.gaussian_model.set_property("opacities", torch.cat([self.gaussian_model.get_property("opacities"), opacities], dim=0))
        self.gaussian_model.set_property("scales", torch.cat([self.gaussian_model.get_property("scales"), scales], dim=0))
        self.gaussian_model.set_property("rotations", torch.cat([self.gaussian_model.get_property("rotations"), rotations], dim=0))
        self.gaussian_model.set_property("shs", torch.cat([self.gaussian_model.get_property("shs"), shs], dim=0))

        self.semantic_features = torch.cat([
            self.semantic_features,
            self.semantic_features[indices.to(self.semantic_features.device)]
        ], dim=0)
        self.scale_conditioned_semantic_features = torch.cat([
            self.scale_conditioned_semantic_features,
            self.scale_conditioned_semantic_features[indices.to(self.scale_conditioned_semantic_features.device)]
        ], dim=0)

        print(f"Duplicated {len(indices)} gaussians. Total: {self.gaussian_model.n_gaussians}")
        self.duplicated_indices = new_indices
        self.original_duplicated_means = means.clone()
        self.original_duplicated_scales = scales.clone()
        return new_indices

class OptionCallbacks:

    def __init__(
            self,
            options,
    ):
        self.options = options

    @property
    def renderer(self) -> SegAnyGSRenderer:
        return self.options.renderer

    @property
    def viewer(self):
        return self.options.viewer

    @property
    def scale_gate(self):
        return self.renderer.scale_gate

    def get_update_scale_conditioned_features_callback(self, on_features_updated_callbacks):
        def update_scale_conditioned_features(scale):
            semantic_features = self.renderer.semantic_features.cuda()

            scale_conditioned_semantic_features = torch.nn.functional.normalize(
                semantic_features * self.scale_gate(scale).to(semantic_features.device),
                dim=-1,
            )
            scale_conditioned_pca_projection_matrix = torch.nn.functional.normalize(self.renderer.pca_projection_matrix * self.scale_gate(scale).to(semantic_features.device).unsqueeze(-1), dim=-1)

            self.renderer.scale_conditioned_semantic_features = scale_conditioned_semantic_features
            self.renderer.scale_conditioned_pca_projection_matrix = scale_conditioned_pca_projection_matrix
            for i in on_features_updated_callbacks:
                i(scale_conditioned_semantic_features)

            # move to cpu after all callback invoked (slow scale update a lot)
            # self.renderer.scale_conditioned_semantic_features = scale_conditioned_semantic_features.cpu()

        return update_scale_conditioned_features

    def update_scale_conditioned_pca_colors(self, scale_conditioned_semantic_features):
        self.renderer.scale_gated_pca_colors = SegAnyGSUtils.get_pca_projected_colors(
            scale_conditioned_semantic_features,
            self.renderer.scale_conditioned_pca_projection_matrix,
        )

    def get_update_selected_point_number_by_mask_callback(self, point_number):
        def update_point_number(mask):
            if mask is None:
                point_number.value = 0
            else:
                point_number.value = mask.sum().item()

        return update_point_number

    def update_segment_mask_on_scale_conditioned_feature_updated(self, *args, **kwargs):
        self.options._segment()


class ViewerOptions:
    def __init__(
            self,
            renderer: SegAnyGSRenderer,
            viewer, server,
            initial_scale: float,
    ):
        self.renderer = renderer
        self.viewer = viewer
        self.server = server

        self.renderer.viewer_options = self  # Connect ViewerOptions to Renderer
        
        self.offset_vector = torch.tensor([0.0, 0.0, 0.0])

        # callback lists
        self.callbacks = OptionCallbacks(self)
        self._on_scale_updated_callbacks = []
        self._on_segment_mask_updated_callbacks = []
        self._on_scale_conditioned_features_updated_callbacks = [
            self.callbacks.update_scale_conditioned_pca_colors,
            self.callbacks.update_segment_mask_on_scale_conditioned_feature_updated,
        ]
        self._on_render_output_type_switched_callbacks = []

        self._on_scale_updated_callbacks.append(
            self.callbacks.get_update_scale_conditioned_features_callback(self._on_scale_conditioned_features_updated_callbacks),
        )

        # properties
        self.scale = initial_scale
        self.similarity_score = 0.9
        self.similarity_score_gamma = 1.

        self._feature_map = None
        self.feature_list = []

        self.segment_result_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "segments",
        )
        self.cluster_result_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "clusters",
        )

        # setup ui
        self._setup_output_type_dropdown()
        self._setup_segment()

    @property
    def scale_gate(self) -> ScaleGateUtils:
        return self.renderer.scale_gate

    @property
    def semantic_features(self) -> torch.Tensor:
        return self.renderer.semantic_features

    @property
    def scale_conditioned_semantic_features(self) -> torch.Tensor:
        return self.renderer.scale_conditioned_semantic_features

    @property
    def segment_mask(self):
        return self.renderer.segment_mask

    @segment_mask.setter
    def segment_mask(self, value):
        self.renderer.segment_mask = value
        for i in self._on_segment_mask_updated_callbacks:
            i(value)

    @property
    def similarities(self):
        return self.renderer.similarities

    @similarities.setter
    def similarities(self, value):
        self.renderer.similarities = value

    @property
    def cluster_result(self):
        return self.renderer.cluster_result

    @cluster_result.setter
    def cluster_result(self, value):
        self.renderer.cluster_result = value
        if value is None:
            self.renderer.cluster_color = None
        else:
            self.renderer.cluster_color = torch.tensor(value["point_colors"], dtype=torch.float, device="cuda")

    def _setup_output_type_dropdown(self):
        render_type_dropdown = self.server.gui.add_dropdown(
            label="Render Type",
            options=list(self.renderer.available_output_types.keys()),
        )

        @render_type_dropdown.on_update
        def _(event):
            if event.client is None:
                return
            self._switch_renderer_output_type(render_type_dropdown.value)

        def update_dropdown(value):
            render_type_dropdown.value = value

        self._on_render_output_type_switched_callbacks.append(update_dropdown)

    def _setup_scale_number(self):
        scale_slider = self.server.gui.add_slider(
            "Scale",
            min=0.,
            max=1.,
            step=0.001,
            initial_value=self.scale,
        )
        self._scale_slider = scale_slider

        @scale_slider.on_update
        def _(event):
            if event.client is None:
                return
            with self.server.atomic():
                self.scale = scale_slider.value
                for i in self._on_scale_updated_callbacks:
                    i(scale_slider.value)
                self.viewer.rerender_for_all_client()

    """
    Segment
    """

    def _segment(self):
        if len(self.feature_list) == 0:
            self.segment_mask = None
            self.similarities = None
            return

        scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cuda()
        
        mask, similarities = SegAnyGSUtils.get_segment_mask_by_raw_feature_list(
            scale_conditioned_semantic_features,
            self.feature_list,
            self.scale_gate,
            self.scale,
            self.similarity_score,
            self.similarity_score_gamma,
            return_similarity_matrix=True,
        )
        similarities = torch.max(similarities, dim=-1).values
        self.segment_mask = mask
        self.similarities = similarities

    def _add_segment_by_query_feature(self, query_feature):
        current_mask = self.segment_mask
        current_similarities = self.similarities
        if current_mask is None:
            current_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
            current_similarities = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.float, device="cuda")

        mask, similarities = SegAnyGSUtils.get_segment_mask_by_raw_feature_list(
            self.scale_conditioned_semantic_features.cuda(),
            [query_feature],
            self.scale_gate,
            self.scale,
            self.similarity_score,
            self.similarity_score_gamma,
            return_similarity_matrix=True,
        )
        similarities = torch.max(similarities, dim=-1).values

        self.segment_mask = torch.logical_or(current_mask, mask)
        self.similarities = torch.maximum(current_similarities, similarities)

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model = self.renderer.gaussian_model
        # Get raw parameters directly
        xyz = model.get_property("means").detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # unused
        
        # Get full SH coefficients and split them
        features = model.get_features  # This returns concatenated [n, N_SHs, 3]
        shs_dc = features[:, :1, :]   # Take first SH coefficient [n, 1, 3]
        shs_rest = features[:, 1:, :] # Take remaining coefficients [n, N_SHs-1, 3]
        
        # Reshape for PLY format
        f_dc = shs_dc.detach().cpu().numpy().transpose(0, 2, 1).reshape(xyz.shape[0], -1)
        f_rest = shs_rest.detach().cpu().numpy().transpose(0, 2, 1).reshape(xyz.shape[0], -1)
        
        opacities = model.get_property("opacities").detach().cpu().numpy()
        scales = model.get_property("scales").detach().cpu().numpy()
        rotations = model.get_property("rotations").detach().cpu().numpy()

        # === Attribute names for PLY header ===
        attribute_names = (
            ["x", "y", "z"] +
            ["nx", "ny", "nz"] +
            [f"f_dc_{i}" for i in range(f_dc.shape[1])] +
            [f"f_rest_{i}" for i in range(f_rest.shape[1])] +
            ["opacity"] +
            [f"scale_{i}" for i in range(scales.shape[1])] +
            [f"rot_{i}" for i in range(rotations.shape[1])]
        )
        dtype_full = [(name, 'f4') for name in attribute_names]

        # === Combine all attributes ===
        attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scales, rotations], axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))

        # === Write to .ply ===
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"[✓] Saved {xyz.shape[0]} Gaussians to PLY at: {path}")

    def _setup_segment(self):
        viewer, server = self.viewer, self.server

        from internal.viewer.client import ClientThread

        def switch_to_segment_output():
            if self.viewer.viewer_renderer.output_type_dropdown.value.startswith("segment3d_") is False:
                self._switch_renderer_output_type("segment3d")

        # setup feature map renderer
        feature_map_render = GSplatContrastiveFeatureRenderer()
        feature_map_render.anti_aliased = self.renderer.anti_aliased

        point_number = server.gui.add_number(
            label="Prompt",
            initial_value=0,
            disabled=True,
        )

        enable_click_mode_button = server.gui.add_button("Enter Click Mode")
        disable_click_mode_button = server.gui.add_button("Exit Click Mode", visible=False, color="red")

        @enable_click_mode_button.on_click
        def _(event):
            enable_click_mode_button.visible = False
            disable_click_mode_button.visible = True

            max_res = viewer.max_res_when_static.value
            camera = ClientThread.get_camera(
                event.client.camera,
                image_size=max_res,
            ).to_device(viewer.device)

            self._feature_map = feature_map_render(
                viewpoint_camera=camera,
                pc=viewer.viewer_renderer.gaussian_model,
                bg_color=torch.zeros((self.semantic_features.shape[-1],), dtype=torch.float, device=viewer.device),
                semantic_features=self.semantic_features.to(device=viewer.device),
            )["render"].permute(1, 2, 0)

            @server.on_scene_pointer(event_type="click")
            def on_scene_click(event):
                x, y = round(event.screen_pos[0][0] * (self._feature_map.shape[1] - 1)), round(event.screen_pos[0][1] * (self._feature_map.shape[0] - 1))
                print(f"x={x}, y={y}")

                feature = self._feature_map[y, x]
                self.feature_list.append(feature)
                self._add_segment_by_query_feature(feature)
                point_number.value += 1
                viewer.rerender_for_all_client()

                new_indices = self.renderer.add_selected_gaussians(offset=None)

                if new_indices is not None:
                    new_mask = torch.zeros((self.renderer.gaussian_model.n_gaussians,), dtype=torch.bool, device="cuda")
                    new_mask[new_indices] = True
                    self.segment_mask = new_mask

                switch_to_segment_output()

                        
        @disable_click_mode_button.on_click
        def _(event):
            server.remove_scene_pointer_callback()
            self._feature_map = None
            enable_click_mode_button.visible = True
            disable_click_mode_button.visible = False

        lock_button = server.gui.add_button("Unselect Object")

        @lock_button.on_click
        def _(_):
            self.renderer.lock_current_offset()
            self.segment_mask = None
            self.similarities = None
            point_number.value = 0
            viewer.rerender_for_all_client()

        server.gui.add_markdown("")
        server.gui.add_markdown("")

        with server.gui.add_folder("Object Controls"):
            # server.gui.add_markdown("##### Offset Controls")
            offset_x = server.gui.add_slider("offset x", min=-10.0, max=10.0, step=0.01, initial_value=self.offset_vector[0].item())
            offset_y = server.gui.add_slider("offset y", min=-10.0, max=10.0, step=0.01, initial_value=self.offset_vector[1].item())
            offset_z = server.gui.add_slider("offset z", min=-10.0, max=10.0, step=0.01, initial_value=self.offset_vector[2].item())

            @offset_x.on_update
            def _(event):
                self.offset_vector[0] = offset_x.value
                self.viewer.rerender_for_all_client()

            @offset_y.on_update
            def _(event):
                self.offset_vector[1] = offset_y.value
                self.viewer.rerender_for_all_client()

            @offset_z.on_update
            def _(event):
                self.offset_vector[2] = offset_z.value
                self.viewer.rerender_for_all_client()

            server.gui.add_markdown("============================")
            scale_slider = server.gui.add_slider("scale factor", min=0.01, max=5.0, step=0.01, initial_value=1.0)

            @scale_slider.on_update
            def _(event):
                if self.renderer.segment_mask is None or self.renderer.segment_mask.sum() == 0:
                    self._show_message(event.client, "No object selected")
                    return

                factor = scale_slider.value
                self.renderer.scale_selected_gaussians(factor)
                self.viewer.rerender_for_all_client()

            server.gui.add_markdown("============================")
            rotate_x = server.gui.add_slider("rotate x", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
            rotate_y = server.gui.add_slider("rotate y", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
            rotate_z = server.gui.add_slider("rotate z", min=-180.0, max=180.0, step=1.0, initial_value=0.0)

            def _rotate_gaussians_from_sliders():
                x_deg = rotate_x.value
                y_deg = rotate_y.value
                z_deg = rotate_z.value

                self.renderer.rotate_selected_gaussians(
                    x_rad=math.radians(x_deg),
                    y_rad=math.radians(y_deg),
                    z_rad=math.radians(z_deg)
                )

            @rotate_x.on_update
            def _(event):
                _rotate_gaussians_from_sliders()
                self.viewer.rerender_for_all_client()

            @rotate_y.on_update
            def _(event):
                _rotate_gaussians_from_sliders()
                self.viewer.rerender_for_all_client()

            @rotate_z.on_update
            def _(event):
                _rotate_gaussians_from_sliders()
                self.viewer.rerender_for_all_client()

        # save segment
        server.gui.add_markdown("")
        with server.gui.add_folder("Save PLY File"):
            save_name = server.gui.add_text("Name", initial_value="")
            save_button = server.gui.add_button("Save")

            @save_button.on_click
            def _(event):
                name = save_name.value.strip()
                if not name:
                    self._show_message(event.client, "Please enter a name for the PLY file.")
                    return

                if not self._filename_check(name):
                    self._show_message(event.client, "Invalid filename. Use only letters, numbers, dashes, underscores, or periods.")
                    return

                path = os.path.join("segmented_objects", f"{name}.ply")
                try:
                    self.save_ply(path)
                    self._show_message(event.client, f"PLY file saved to: {path}")
                except Exception as e:
                    print("Failed to save PLY file:", e)
                    traceback.print_exc()
                    self._show_message(event.client, f"Error while saving PLY file: {str(e)}")

                for i in range(len(self.renderer.all_train_cameras)-1):
                    camera = self.renderer.all_train_cameras.__getitem__(i)

                    # Render the images with just the segmented object in it
                    out = self.renderer.forward(
                        camera,
                        self.renderer.gaussian_model,
                        torch.tensor(self.renderer.bg_color).cuda(),
                        scaling_modifier=1.0,
                        render_types=["rgb"],
                    )

                    # Save rendered images that contain the segmented object
                    img_name = self.renderer.original_img_names[i].replace(".jpg", "")

                    os.makedirs("segmented_objects", exist_ok=True)
                    os.makedirs("segmented_objects/rendered_images", exist_ok=True)

                    torchvision.utils.save_image(out['rgb'], f"segmented_objects/rendered_images/{img_name}.png")

                print('Done rendering images')
    
    def _scan_cluster_files(self):
        return self._scan_pt_files(self.cluster_result_save_dir)

    def _scan_pt_files(self, path):
        file_list = []
        try:
            for i in os.listdir(path):
                if i.endswith(".pt"):
                    file_list.append(i)
        except:
            pass
        return file_list

    def save_cluster_results(self, name):
        if self.cluster_result is None:
            raise RuntimeError("Please click 'Re-Cluster in 3D' first")

        match = re.search(r"^[a-zA-Z0-9_\-.]+$", name)
        if match:
            output_path = os.path.join(self.cluster_result_save_dir, f"{name}.pt")
            if os.path.exists(output_path):
                raise RuntimeError("File already exists")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(self.cluster_result, output_path)

            return output_path
        else:
            raise RuntimeError("Invalid name")

    def _switch_renderer_output_type(self, type):
        output_type_info = self.renderer.get_available_outputs().get(type, None)
        if output_type_info is None:
            return

        viewer = self.viewer
        viewer.viewer_renderer.output_type_dropdown.value = type
        viewer.viewer_renderer._set_output_type(type, output_type_info)

        for i in self._on_render_output_type_switched_callbacks:
            i(type)

        viewer.rerender_for_all_client()

    def print_cluster_start_message(self, client=None):
        message = "Cluster takes some time. The viewer will not response any requests during this process (may including the 'Close' button below), please be patient...<br/>You will be noticed when it is completed."
        print(message)
        self._show_message(client, message)

    def print_cluster_finished_message(self, client=None):
        message = f"Cluster completed: {len(self.cluster_result['cluster_labels'])} labels"
        print(message)
        self._show_message(client, message)

    def _show_message(self, client, message: str):
        target = client
        if target is None:
            target = self.server

        with target.gui.add_modal("Message") as modal:
            target.gui.add_markdown(message)
            close_button = target.gui.add_button("Close")

            @close_button.on_click
            def _(_) -> None:
                try:
                    modal.close()
                except:
                    pass

    def _filename_check(self, name) -> bool:
        return re.search(r"^[a-zA-Z0-9_\-.]+$", name) is not None
