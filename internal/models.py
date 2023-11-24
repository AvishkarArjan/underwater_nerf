import torch
from torch.utils._pytree import tree_map
import torch.nn as nn
import numpy as np

from internal import geopoly
from internal import ref_utils
from internal import coord
from internal import render
from internal import stepfun

def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)

class Model:
    """The Sea-thru-NeRF Model containining all MLPs.
            Args:
              rand: random number generator (or None for deterministic output).
              batch/rays: util.Rays, a pytree of ray origins, directions, and view-dirs.
              train_frac: float in [0, 1], what fraction of training is complete.
              compute_extras: bool, if True, compute extra quantities besides color.
              zero_glo: bool, if True, when using GLO pass in vector of zeros.

            Returns:
              ret: list, [*(rgb, distance, acc)]
            """
    """Checkout multi-nerf pytorch to get random val genrator - no need for rng"""
    
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = 'contract'  # The curve used for ray dists.
    ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
    disable_integration: bool = False  # If True, use PE instead of IPE.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.

    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        # self.nerf_mlp = NerfMLP(num_glo_features=self.num_glo_features,                              num_glo_embeddings=self.num_glo_embeddings)
        self.nerf_mlp = UWMLP() if self.config.use_uw_mlp else NerfMLP()
        self.prop_mlp = self.nerf_mlp if self.single_mlp else PropMLP()
        # self.prop_mlp = torch.compile(self.prop_mlp)
        if self.num_glo_features > 0 and not config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            max_num_exposures = self.num_glo_embeddings
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)


    def forward(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=True,
    ):
       
        device = batch['origins'].device
        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None


         # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'])

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.near_anneal_rate is None:
            init_s_near = 0.
        else:
            init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
                                  self.near_anneal_init)
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(batch['near'])
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                    init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]


            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                rand,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far))

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            gaussians = render.cast_rays(
                tdist,
                batch['origins'],
                batch['directions'],
                batch['radii'],
                self.ray_shape,
                diag=False)

            if self.disable_integration:
                # Setting the covariance of our Gaussian samples to 0 disables the
                # "integrated" part of integrated positional encoding.
                gaussians = (gaussians[0], torch.zeros_like(gaussians[1]))

            # Push our Gaussians through one of our two MLPs.
            mlp = self.prop_mlp if is_prop else self.nerf_mlp
            ray_results = mlp(
                rand,
                gaussians,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
                imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values'),
            )

            
            if self.config.use_uw_mlp and not is_prop and not self.config.gen_eq:
                # Get the weights used by volumetric rendering (and our other losses).
                weights, alpha, trans, bs_weights, trans_atten, alpha_bs, trans_bs = render.compute_alpha_weights_uw(
                    density_obj=ray_results['density'],
                    sigma_bs=ray_results['sigma_bs'],
                    sigma_atten=ray_results['sigma_atten'],
                    tdist=tdist, dirs=batch.directions,c_med = ray_results['c_med'], xyz_atten=self.config.uw_atten_xyz, extra_samples=self.config.extra_samples)


            elif self.config.gen_eq and not is_prop:
                weights, alpha, trans, alpha_bs, trans_bs, alpha_atten, trans_atten = render.compute_alpha_weights_uw_gen(
                    density_obj=ray_results['density'],
                    sigma_bs=ray_results['sigma_bs'],
                    sigma_atten=ray_results['sigma_atten'],
                    tdist=tdist, dirs=batch.directions,
                )

            else:
                # Get the weights used by volumetric rendering (and our other losses).
                weights, alpha, trans = render.compute_alpha_weights(
                    density=ray_results['density'],
                    tdist=tdist,
                    dirs=batch.directions,
                    opaque_background=self.opaque_background,
                )
            
            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval


            if self.config.use_uw_mlp and not is_prop and not self.config.gen_eq:

                rendering = render.volumetric_rendering_uw(density=ray_results['density'],
                                                           rgbs=ray_results['rgb'],
                                                           c_med=ray_results['c_med'],
                                                           bs_weights=bs_weights, trans_atten=trans_atten,
                                                           trans=trans,
                                                           weights=weights,
                                                           tdist=tdist,
                                                           t_far=batch.far,
                                                           compute_extras=compute_extras, extra_samples=self.config.extra_samples,
                                                           extras={
                                                               k: v
                                                               for k, v in ray_results.items()
                                                               if k.startswith('distance') or k in ['roughness']
                                                           })
                rendering['sigma_bs'] = ray_results['sigma_bs']
                rendering['sigma_atten'] = ray_results['sigma_atten']
                rendering['weights'] = weights
                rendering['trans'] = trans

            elif self.config.gen_eq and not is_prop:
                rendering = render.volumetric_rendering_uw_gen(density=ray_results['density'],
                                                               rgbs=ray_results['rgb'],
                                                               c_med=ray_results['c_med'],
                                                               alpha_bs=alpha_bs, alpha_atten=alpha_atten,
                                                               trans_bs=trans_bs, trans_atten=trans_atten,
                                                               trans=trans,
                                                               weights=weights,
                                                               tdist=tdist,
                                                               sigma_bs=ray_results['sigma_bs'],
                                                               sigma_atten=ray_results['sigma_atten'],
                                                               t_far=batch.far,
                                                               compute_extras=compute_extras,
                                                               extras={
                                                                   k: v
                                                                   for k, v in ray_results.items()
                                                                   if k.startswith('distance') or k in ['roughness']
                                                               })
                
                rendering['sigma_bs'] = ray_results['sigma_bs']
                rendering['sigma_atten'] = ray_results['sigma_atten']
                rendering['weights'] = weights
                rendering['trans'] = trans
                rendering['tdist'] = tdist

            else:
                # Render each ray.
                rendering = render.volumetric_rendering(
                    rgbs=ray_results['rgb'],
                    weights=weights,
                    tdist=tdist,
                    bg_rgbs=bg_rgbs,
                    t_far=batch.far,
                    compute_extras=compute_extras,
                    extras={
                        k: v
                        for k, v in ray_results.items()
                        if k.startswith('normals') or k in ['roughness']
                    })
                rendering['density'] = ray_results['density']
                rendering['weights'] = weights

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                if not is_prop and self.config.use_uw_mlp:
                    # rendering['ray_weights'] = (
                    #     weights[:,:,-1].reshape([-1, weights.shape[-1]])[:n, :])
                    rendering['ray_weights'] = (
                        weights.reshape([-1, weights.shape[-1]])[:n, :])
                else:
                    rendering['ray_weights'] = (
                        weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]
                rendering['ray_tdist'] = tdist[:n, :]

                if not is_prop and self.config.use_uw_mlp:
                    # rendering['ray_trans'] = trans_bs[:n, :,-1]
                    # rendering['ray_alpha'] = alpha_bs[:n, :,-1]
                    # rendering['ray_density'] = jnp.repeat(jnp.expand_dims(ray_results['sigma_bs'][:n,-1],-1),32,1)
                    rendering['ray_trans'] = trans[:n, :]
                    rendering['ray_alpha'] = alpha[:n, :]
                    rendering['ray_density'] = ray_results['density'][:n, :]
                else:
                    rendering['ray_trans'] = trans[:n, :]
                    rendering['ray_alpha'] = alpha[:n, :]
                    rendering['ray_density'] = ray_results['density'][:n, :]
                rendering['ray_xy'] = batch.imageplane[:n, :]
                rendering['ray_x'] = batch.x_coord[:n]
                rendering['ray_y'] = batch.y_coord[:n]
            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_results['trans'] = trans.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], axis=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


class MLP(nn.Module):
    """A PosEnc MLP."""
    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 256  # The width of the first part of MLP.
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 1  # The depth of the second part of ML.
    net_width_viewdirs: int = 128  # The width of the second part of MLP.
    min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
    max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
    weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
    skip_layer: int = 4  # Add a skip connection to the output of every N layers.
    skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
    basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Make sure that normals are computed if reflection direction is used.
        # if self.use_reflections and not (self.enable_pred_normals or
        #                                  not self.disable_density_normals):
        #     raise ValueError('Normals must be computed for reflection directions.')

        # Precompute and store (the transpose of) the basis being used.
        pos_basis_t = torch.from_numpy(
            geopoly.generate_basis(self.basis_shape, self.basis_subdivisions).copy().T).float()
        self.register_buffer("pos_basis_t", pos_basis_t)

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]

        input_dim = pos_basis_t.shape[-1] * self.max_deg_point * 2
        last_dim = input_dim
        for i in range(self.net_depth):
            lin = nn.Linear(last_dim, self.net_width)
            torch.nn.init.kaiming_uniform_(lin.weight)
            self.register_module(f"lin_first_stage_{i}", lin)
            last_dim = self.net_width
            if i % self.skip_layer == 0 and i > 0:
                last_dim += input_dim
        self.density_layer = nn.Linear(last_dim, 1)  # Hardcoded to a single channel.

        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)

            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                self.bottleneck_layer = nn.Linear(last_dim, self.bottleneck_width)
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0

            last_dim_rgb += dim_dir_enc

            if self.use_n_dot_v:
                last_dim_rgb += 1

            if self.__class__.__name__ == "NerfMLP" and self.num_glo_features > 0:
                last_dim_rgb += self.num_glo_embeddings
            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i % self.skip_layer_dir == 0 and i > 0:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, covs, rand):
        """Helper function to output density."""
        # Encode input positions

        if self.warp_fn is not None:
            means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = (
            coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
        x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                     self.min_deg_point, self.max_deg_point)

        inputs = x
        # Evaluate network to produce the output density.
        for i in range(self.net_depth):
            x = self.get_submodule(f"lin_first_stage_{i}")(x)
            x = F.relu(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x

    def forward(self,
                rand,
                gaussians,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None):
        """Evaluate the MLP.

    Args:
      rand: Random number generator.
      gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        means, covs = gaussians
        if self.disable_density_normals:
            raw_density, x = self.predict_density(means, covs, rand)
            raw_grad_density = None
            normals = None
        else:
            means.requires_grad_(True)
            # Flatten the input so value_and_grad can be vmap'ed.
            means_flat = means.reshape((-1, means.shape[-1]))
            covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1:])

            raw_density_flat, x_flat = self.predict_density(means_flat, covs_flat, rand)
            d_output = torch.ones_like(raw_density_flat, requires_grad=False, device=raw_density_flat.device)
            with torch.enable_grad():
                raw_grad_density_flat = torch.autograd.grad(
                    outputs=raw_density_flat,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            # Unflatten the output.
            raw_density = raw_density_flat.reshape(means.shape[:-1])
            x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
            raw_grad_density = raw_grad_density_flat.reshape(means.shape)

            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = self.bottleneck_layer(x)

                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)
                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)

                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)

                # Append GLO vector if used.
                if glo_vec is not None:
                    glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                 bottleneck.shape[:-1] + glo_vec.shape[-1:])
                    x.append(glo_vec)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i % self.skip_layer_dir == 0 and i > 0:
                        x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )



class UWMLP(nn.Module):
    """A Under-water MLP."""
    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 256  # The width of the first part of MLP.
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 1  # The depth of the second part of ML.
    net_width_viewdirs: int = 128  # The width of the second part of MLP.
    min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
    max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
    weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
    skip_layer: int = 4  # Add a skip connection to the output of every N layers.
    skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    water_bias: float = -1 # Shift added to raw densities pre-activation
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    uw_old_model:bool = False # If True same sigmas
    uw_fog_model: bool = False # If True same sigmas same channel
    uw_rgb_dir: bool = False # If False, no view dir for rgb obj
    uw_atten_xyz: bool = False  # If True use rgb xyz coordinates also as input for sigma_atten prediction
    gen_eq: bool = False  # If True use general eq. (11)-(14)
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
    basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')
        
        self.net_depth_water = 1

        # Precompute and store (the transpose of) the basis being used.
        pos_basis_t = torch.from_numpy(
            geopoly.generate_basis(self.basis_shape, self.basis_subdivisions).copy().T).float()
        self.register_buffer("pos_basis_t", pos_basis_t)

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]

        input_dim = pos_basis_t.shape[-1] * self.max_deg_point * 2
        last_dim = input_dim
        for i in range(self.net_depth):
            lin = nn.Linear(last_dim, self.net_width)
            torch.nn.init.kaiming_uniform_(lin.weight)
            self.register_module(f"lin_first_stage_{i}", lin)
            last_dim = self.net_width
            if i % self.skip_layer == 0 and i > 0:
                last_dim += input_dim
        self.density_layer = nn.Linear(last_dim, 1)  # Hardcoded to a single channel.

        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)

            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                self.bottleneck_layer = nn.Linear(last_dim, self.bottleneck_width)
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0

            last_dim_rgb += dim_dir_enc

            if self.use_n_dot_v:
                last_dim_rgb += 1

            if self.__class__.__name__ == "NerfMLP" and self.num_glo_features > 0:
                last_dim_rgb += self.num_glo_embeddings
            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i % self.skip_layer_dir == 0 and i > 0:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, covs, rand):
        """Helper function to output density."""
        # Encode input positions

        if self.warp_fn is not None:
            means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = (
            coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
        x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                     self.min_deg_point, self.max_deg_point)

        inputs = x
        # Evaluate network to produce the output density.
        for i in range(self.net_depth):
            x = self.get_submodule(f"lin_first_stage_{i}")(x)
            x = F.relu(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x

    def forward(self,
                rand,
                gaussians,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None):
        """Evaluate the MLP.

    Args:
      rand: Random number generator.
      gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        means, covs = gaussians
        if self.disable_density_normals:
            raw_density, x = self.predict_density(means, covs, rand)
            raw_grad_density = None
            normals = None
        else:
            means.requires_grad_(True)
            # Flatten the input so value_and_grad can be vmap'ed.
            means_flat = means.reshape((-1, means.shape[-1]))
            covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1:])

            raw_density_flat, x_flat = self.predict_density(means_flat, covs_flat, rand)
            d_output = torch.ones_like(raw_density_flat, requires_grad=False, device=raw_density_flat.device)
            with torch.enable_grad():
                raw_grad_density_flat = torch.autograd.grad(
                    outputs=raw_density_flat,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            # Unflatten the output.
            raw_density = raw_density_flat.reshape(means.shape[:-1])
            x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
            raw_grad_density = raw_grad_density_flat.reshape(means.shape)

            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = self.bottleneck_layer(x)

                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)
                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)

                    dir_enc_for_water = dir_enc

                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)

                # Append GLO vector if used.
                # if glo_vec is not None:
                #     glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                #                                  bottleneck.shape[:-1] + glo_vec.shape[-1:])
                #     x.append(glo_vec)
                
                #water MLP
                if glo_vec is None:
                    dir_enc_for_water_1=dir_enc_for_water
                    for i in range(self.net_depth_water):
                        dir_enc_for_water_1=x.get_submodule(f"lin_first_stage_{i}")(x)
                        dir_enc_for_water_1=F.relu(dir_enc_for_water_1)
                        if i% self.skip_layer ==0 and i>0:
                            dir_enc_for_water_1= torch.cat([dir_enc_for_water, dir_enc_for_water_1], dim=-1)
                    
                    c_med = torch.sigmoid(nn.Linear(self.num_rgb_channels, dir_enc_for_water_1))

                    if self.uw_for_model:
                        sigma_bs=F.Softplus(nn.Linear(1, dir_enc_for_water_1))
                    else:
                        sigma_bs=F.Softplus(nn.Linear(self.num_rgb_channels, dir_enc_for_water_1) + self.water_bias)

                    if self.uw_old_model or self.uw_fog_model:
                        sigma_atten=sigma_bs
                    elif self.uw_atten_xyz:
                        dir_enc_for_water_1 = torch.broadcast_to(
                            dir_enc_for_water_1[..., None, :],
                            bottleneck.shape[:-1] + (dir_enc_for_water_1.shape[-1],)
                        )
                        dir_enc_for_water_1=torch.cat([dir_enc_for_water_1, x[0]], axis=-1)
                        sigma_atten = F.Softplus(nn.Linear(self.num_rgb_channels, dir_enc_for_water_1))
                    else:
                        sigma_atten = F.Softplus(nn.Linear(self.num_rgb_channels, dir_enc_for_water_1) + self.water_bias)
                    
                    if self.uw_rgb_dir:
                        x.append(dir_enc)
                else:
                    sigma_bs == F.relu(glo_vec[..., 0:3])
                    sigma_atten=F.relu(glo_vec[..., 3:6])
                    c_med=F.relu(glo_vec[..., 6:])
                        

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i % self.skip_layer_dir == 0 and i > 0:
                        x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
            c_med=c_med,
            sigma_bs=sigma_bs,
            sigma_atten=sigma_atten,
            glo_vec=glo_vec
        )

class NerfMLP(MLP):
    pass

class PropMLP(MLP):
    pass


def construct_model(rays, config):
    # construct a Sea-Thru Nerf Model
    """
    Args:
        rng: removed - rand generated in Model class
        rays: example of input Rays
        confif: A Config class

    Returns:
        model: nn.Module, NeRF model with parameters
        init_variables: nerf model parameters

    """
    # grab 10 rays
    ray = tree_map(lambda x: np.reshape(x, [-1, x.shape[-1]])[:10], rays)
    model = Model(config=config)

    init_variables = model.init(
        rays=ray,
        train_frac=1,
        compute_extras=False,
        zero_glo=model.num_glo_features==0
    )

    return model, init_variables

def render_image():
    pass
