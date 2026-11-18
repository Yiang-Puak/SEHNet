"""Parameter-efficient functional hypothesis ensemble for frozen ESM features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleLinear(nn.Module):
    """A shared linear map with rank-one adapters for each hypothesis."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int,
        *,
        diversify_input: bool = False,
        residual_rank: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.ensemble_size = int(ensemble_size)
        self.residual_rank = int(residual_rank)
        if self.residual_rank < 0:
            raise ValueError("residual_rank must be non-negative")
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.input_scale = nn.Parameter(
            torch.ones(ensemble_size, input_dim)
        )
        self.output_scale = nn.Parameter(
            torch.ones(ensemble_size, output_dim)
        )
        self.member_bias = nn.Parameter(
            torch.zeros(ensemble_size, output_dim)
        )
        if self.residual_rank > 0:
            self.residual_left = nn.Parameter(
                torch.zeros(
                    ensemble_size,
                    output_dim,
                    self.residual_rank,
                )
            )
            self.residual_right = nn.Parameter(
                torch.empty(
                    ensemble_size,
                    self.residual_rank,
                    input_dim,
                )
            )
            nn.init.normal_(
                self.residual_right,
                mean=0.0,
                std=input_dim**-0.5,
            )
        else:
            self.register_parameter("residual_left", None)
            self.register_parameter("residual_right", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = input_dim**-0.5
        nn.init.uniform_(self.bias, -bound, bound)
        if diversify_input:
            with torch.no_grad():
                # Do not advance the global CPU RNG here: changing K would
                # otherwise also change every shared layer initialized later.
                sign_generator = torch.Generator(device="cpu")
                sign_generator.manual_seed(
                    (torch.initial_seed() + 0x5EED5EED) % (2**63 - 1)
                )
                signs = torch.empty(
                    self.input_scale.shape,
                    dtype=self.input_scale.dtype,
                    device="cpu",
                ).bernoulli_(0.5, generator=sign_generator)
                signs = signs.to(device=self.input_scale.device)
                self.input_scale.copy_(signs.mul_(2.0).sub_(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._forward(inputs, detach_parameters=False)

    def forward_detached(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the same classifier while updating only its inputs."""

        return self._forward(inputs, detach_parameters=True)

    def _forward(
        self,
        inputs: torch.Tensor,
        *,
        detach_parameters: bool,
    ) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.ensemble_size:
            raise ValueError(
                "BatchEnsembleLinear expects shape "
                f"(batch,{self.ensemble_size},{self.input_dim}), "
                f"found {tuple(inputs.shape)}"
            )
        parameter = (
            (lambda value: value.detach())
            if detach_parameters
            else (lambda value: value)
        )
        outputs = F.linear(
            inputs * parameter(self.input_scale).unsqueeze(0),
            parameter(self.weight),
            parameter(self.bias),
        )
        outputs = (
            outputs * parameter(self.output_scale).unsqueeze(0)
            + parameter(self.member_bias).unsqueeze(0)
        )
        if self.residual_left is not None:
            low_rank_features = torch.einsum(
                "bki,kri->bkr",
                inputs,
                parameter(self.residual_right),
            )
            outputs = outputs + torch.einsum(
                "bkr,kor->bko",
                low_rank_features,
                parameter(self.residual_left),
            )
        return outputs


class EnsembleBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int,
        dropout: float,
        *,
        diversify_input: bool = False,
    ) -> None:
        super().__init__()
        self.linear = BatchEnsembleLinear(
            input_dim,
            output_dim,
            ensemble_size,
            diversify_input=diversify_input,
        )
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        batch_size, ensemble_size, width = outputs.shape
        outputs = self.norm(outputs.reshape(batch_size * ensemble_size, width))
        outputs = outputs.reshape(batch_size, ensemble_size, width)
        return self.dropout(F.relu(outputs))


class IndependentHypothesisClassifier(nn.Module):
    """A complete label classifier for each functional hypothesis."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.ensemble_size = int(ensemble_size)
        self.member_weight = nn.Parameter(
            torch.empty(ensemble_size, output_dim, input_dim)
        )
        self.member_bias = nn.Parameter(
            torch.empty(ensemble_size, output_dim)
        )
        for index in range(ensemble_size):
            nn.init.kaiming_uniform_(
                self.member_weight[index],
                a=5**0.5,
            )
        bound = input_dim**-0.5
        nn.init.uniform_(self.member_bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._forward(inputs, detach_parameters=False)

    def forward_detached(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the same classifier while updating only its inputs."""

        return self._forward(inputs, detach_parameters=True)

    def _forward(
        self,
        inputs: torch.Tensor,
        *,
        detach_parameters: bool,
    ) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.ensemble_size:
            raise ValueError(
                "IndependentHypothesisClassifier expects shape "
                f"(batch,{self.ensemble_size},{self.input_dim}), "
                f"found {tuple(inputs.shape)}"
            )
        weight = (
            self.member_weight.detach()
            if detach_parameters
            else self.member_weight
        )
        bias = (
            self.member_bias.detach()
            if detach_parameters
            else self.member_bias
        )
        return (
            torch.einsum("bki,koi->bko", inputs, weight)
            + bias.unsqueeze(0)
        )


class ReZeroSwiGLURefinement(nn.Module):
    """Function-preserving gated growth for converged hypothesis features."""

    def __init__(
        self,
        hidden_dim: int,
        ensemble_size: int,
        expansion: float,
        dropout: float,
    ) -> None:
        super().__init__()
        inner_dim = max(64, int(round(hidden_dim * float(expansion))))
        self.norm = nn.LayerNorm(hidden_dim)
        self.value = nn.Linear(hidden_dim, inner_dim)
        self.gate = nn.Linear(hidden_dim, inner_dim)
        self.output = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.member_scale = nn.Parameter(torch.zeros(ensemble_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        refined = self.output(
            F.silu(self.gate(normalized)) * self.value(normalized)
        )
        return features + (
            self.member_scale[None, :, None]
            * self.dropout(refined)
        )


class ParentContrastiveLocalAttention(nn.Module):
    """GO semantic queries over centered local protein evidence."""

    def __init__(
        self,
        *,
        local_dim: int,
        attention_dim: int,
        go_semantics: torch.Tensor,
        parent_contrast: torch.Tensor,
        depth_groups: torch.Tensor,
        label_chunk_size: int,
        task_query_residual: bool = False,
        query_source: str = "semantic",
        num_local_layers: int = 1,
        classifier_parent_contrast: bool = True,
        classifier_dim: int | None = None,
        parent_child_indices: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
        parent_counts: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if label_chunk_size <= 0:
            raise ValueError("label_chunk_size must be positive")
        if (
            go_semantics.ndim != 2
            or parent_contrast.shape != go_semantics.shape
            or depth_groups.shape != (go_semantics.shape[0],)
        ):
            raise ValueError("Invalid GO semantic local-attention buffers")
        self.local_dim = int(local_dim)
        self.attention_dim = int(attention_dim)
        self.label_chunk_size = int(label_chunk_size)
        self.num_local_layers = int(num_local_layers)
        if self.num_local_layers <= 0:
            raise ValueError("num_local_layers must be positive")
        self.classifier_parent_contrast = bool(
            classifier_parent_contrast
        )
        self.query_source = str(query_source)
        if self.query_source not in {"semantic", "classifier"}:
            raise ValueError(
                f"Unsupported local query source: {self.query_source}"
            )
        if task_query_residual and self.query_source != "semantic":
            raise ValueError(
                "Task query residual is only defined for semantic queries"
            )
        if self.num_local_layers == 1:
            self.local_norm = nn.LayerNorm(self.local_dim)
            self.key_proj = nn.Linear(
                self.local_dim,
                self.attention_dim,
                bias=False,
            )
            self.value_proj = nn.Linear(
                self.local_dim,
                self.attention_dim,
                bias=False,
            )
            self.layer_norms = None
            self.layer_key_projs = None
            self.layer_value_projs = None
            self.register_parameter("depth_layer_logits", None)
        else:
            self.local_norm = None
            self.key_proj = None
            self.value_proj = None
            self.layer_norms = nn.ModuleList(
                nn.LayerNorm(self.local_dim)
                for _ in range(self.num_local_layers)
            )
            self.layer_key_projs = nn.ModuleList(
                nn.Linear(
                    self.local_dim,
                    self.attention_dim,
                    bias=False,
                )
                for _ in range(self.num_local_layers)
            )
            self.layer_value_projs = nn.ModuleList(
                nn.Linear(
                    self.local_dim,
                    self.attention_dim,
                    bias=False,
                )
                for _ in range(self.num_local_layers)
            )
            self.depth_layer_logits = nn.Parameter(
                torch.zeros(4, self.num_local_layers)
            )
        if self.query_source == "semantic":
            semantic_dim = int(go_semantics.shape[1])
            self.locator_proj = nn.Linear(
                semantic_dim,
                self.attention_dim,
                bias=False,
            )
            self.discriminator_proj = nn.Linear(
                semantic_dim,
                self.attention_dim,
                bias=False,
            )
            self.register_module("prototype_proj", None)
            if task_query_residual:
                self.query_residual = nn.Parameter(
                    torch.zeros(
                        go_semantics.shape[0],
                        self.attention_dim,
                    )
                )
            else:
                self.register_parameter("query_residual", None)
        else:
            if classifier_dim is None or classifier_dim <= 0:
                raise ValueError(
                    "Classifier queries require a positive classifier_dim"
                )
            if self.classifier_parent_contrast and (
                parent_child_indices is None
                or parent_indices is None
                or parent_counts is None
            ):
                raise ValueError(
                    "Classifier queries require direct GO parent edges"
                )
            if (
                self.classifier_parent_contrast
                and parent_counts is not None
                and parent_counts.shape != (go_semantics.shape[0],)
            ):
                raise ValueError("Invalid classifier-query parent counts")
            self.register_module("locator_proj", None)
            self.register_module("discriminator_proj", None)
            self.register_parameter("query_residual", None)
            self.prototype_proj = nn.Linear(
                int(classifier_dim),
                self.attention_dim,
                bias=False,
            )
        self.depth_gates = nn.Parameter(torch.zeros(4))
        self.register_buffer(
            "go_semantics",
            go_semantics.float(),
            persistent=False,
        )
        self.register_buffer(
            "parent_contrast",
            parent_contrast.float(),
            persistent=False,
        )
        self.register_buffer(
            "has_parent_contrast",
            parent_contrast.norm(dim=-1).gt(0),
            persistent=False,
        )
        self.register_buffer(
            "depth_groups",
            depth_groups.long(),
            persistent=False,
        )
        self.register_buffer(
            "parent_child_indices",
            (
                parent_child_indices.long()
                if parent_child_indices is not None
                else torch.empty(0, dtype=torch.long)
            ),
            persistent=False,
        )
        self.register_buffer(
            "parent_indices",
            (
                parent_indices.long()
                if parent_indices is not None
                else torch.empty(0, dtype=torch.long)
            ),
            persistent=False,
        )
        self.register_buffer(
            "parent_counts",
            (
                parent_counts.long()
                if parent_counts is not None
                else torch.zeros(
                    go_semantics.shape[0],
                    dtype=torch.long,
                )
            ),
            persistent=False,
        )

    def forward(
        self,
        local_segments: torch.Tensor,
        local_mask: torch.Tensor,
        label_prototypes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_ndim = 3 if self.num_local_layers == 1 else 4
        if local_segments.ndim != expected_ndim or (
            local_segments.shape[-1] != self.local_dim
        ):
            raise ValueError(
                "Expected centered local segments with shape "
                + (
                    f"(batch,segments,{self.local_dim})"
                    if self.num_local_layers == 1
                    else (
                        f"(batch,{self.num_local_layers},segments,"
                        f"{self.local_dim})"
                    )
                )
                + ", found "
                f"{tuple(local_segments.shape)}"
            )
        expected_mask_shape = (
            local_segments.shape[:2]
            if self.num_local_layers == 1
            else (local_segments.shape[0], local_segments.shape[2])
        )
        if tuple(local_mask.shape) != tuple(expected_mask_shape):
            raise ValueError(
                f"Local mask shape mismatch: {tuple(local_mask.shape)}"
            )
        mask = local_mask.bool()
        present = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~present, 0] = True

        if self.query_source == "semantic":
            locators = self.locator_proj(self.go_semantics)
            discriminator_logits = self.discriminator_proj(
                self.parent_contrast
            )
            if self.query_residual is not None:
                locators = locators + self.query_residual
                discriminator_logits = (
                    discriminator_logits
                    + self.query_residual
                    * self.has_parent_contrast[:, None].to(
                        self.query_residual.dtype
                    )
                )
        else:
            if (
                label_prototypes is None
                or label_prototypes.ndim != 2
                or label_prototypes.shape[0] != len(self.depth_groups)
            ):
                raise ValueError(
                    "Classifier local queries require label prototypes "
                    "in model term order"
                )
            locators = self.prototype_proj(label_prototypes.detach())
            if self.classifier_parent_contrast:
                parent_sums = torch.zeros_like(locators)
                parent_sums.index_add_(
                    0,
                    self.parent_child_indices,
                    locators[self.parent_indices],
                )
                counts = self.parent_counts.clamp_min(1).to(
                    locators.dtype
                )
                discriminator_logits = locators - (
                    parent_sums / counts[:, None]
                )
                discriminator_logits = (
                    discriminator_logits
                    * self.parent_counts.gt(0)[:, None].to(locators.dtype)
                )
            else:
                discriminator_logits = locators
        discriminators = F.normalize(discriminator_logits, dim=-1)
        scale = self.attention_dim**-0.5

        def attend_one_layer(
            keys: torch.Tensor,
            values: torch.Tensor,
        ) -> torch.Tensor:
            chunks = []
            for start in range(0, len(locators), self.label_chunk_size):
                end = min(start + self.label_chunk_size, len(locators))
                attention_logits = torch.einsum(
                    "bsd,cd->bcs",
                    keys,
                    locators[start:end],
                ) * scale
                attention_logits = attention_logits.masked_fill(
                    ~safe_mask[:, None, :],
                    torch.finfo(attention_logits.dtype).min,
                )
                weights = torch.softmax(attention_logits, dim=-1)
                context = torch.einsum(
                    "bcs,bsd->bcd",
                    weights,
                    values,
                )
                context = F.normalize(context, dim=-1)
                chunks.append(
                    torch.einsum(
                        "bcd,cd->bc",
                        context,
                        discriminators[start:end],
                    )
                )
            layer_evidence = torch.cat(chunks, dim=1)
            return (
                layer_evidence
                * present[:, None].to(layer_evidence.dtype)
            )

        if self.num_local_layers == 1:
            assert self.local_norm is not None
            assert self.key_proj is not None
            assert self.value_proj is not None
            normalized_segments = self.local_norm(local_segments)
            evidence = attend_one_layer(
                self.key_proj(normalized_segments),
                self.value_proj(normalized_segments),
            )
        else:
            assert self.layer_norms is not None
            assert self.layer_key_projs is not None
            assert self.layer_value_projs is not None
            assert self.depth_layer_logits is not None
            layer_evidence = []
            for layer_index in range(self.num_local_layers):
                normalized_segments = self.layer_norms[layer_index](
                    local_segments[:, layer_index]
                )
                layer_evidence.append(
                    attend_one_layer(
                        self.layer_key_projs[layer_index](
                            normalized_segments
                        ),
                        self.layer_value_projs[layer_index](
                            normalized_segments
                        ),
                    )
                )
            evidence_by_layer = torch.stack(layer_evidence, dim=-1)
            depth_layer_weights = torch.softmax(
                self.depth_layer_logits,
                dim=-1,
            )
            label_layer_weights = depth_layer_weights[self.depth_groups]
            evidence = (
                evidence_by_layer
                * label_layer_weights[None].to(evidence_by_layer.dtype)
            ).sum(dim=-1)
        label_gates = self.depth_gates[self.depth_groups]
        return evidence * label_gates[None, :].to(evidence.dtype)


class LatentDomainLocalFusion(nn.Module):
    """Pool local ESM evidence through a small shared domain bottleneck.

    The fixed uniform layer mix is deliberately label independent.  A small
    bank of learned domain queries first pools sequence segments, then each
    functional hypothesis selects among the resulting protein-specific
    domain tokens.  The global hypothesis features are detached only on the
    local-query path so the auxiliary stream cannot reshape the main encoder
    through its attention query.
    """

    def __init__(
        self,
        *,
        local_dim: int,
        hidden_dim: int,
        ensemble_size: int,
        attention_dim: int = 64,
        num_domains: int = 8,
        num_local_layers: int = 3,
        layer_contrast: bool = False,
        competitive_segment_assignment: bool = False,
        label_readout: bool = False,
        normalize_label_queries: bool = False,
        adaptive_label_query_norm: bool = False,
        confidence_direction_routing: bool = False,
        ic_semantic_residual: bool = False,
        label_semantics: torch.Tensor | None = None,
        label_semantic_weights: torch.Tensor | None = None,
        semantic_dim: int = 768,
        label_chunk_size: int = 512,
        label_depth_groups: torch.Tensor | None = None,
        parent_child_indices: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
        parent_counts: torch.Tensor | None = None,
        label_selection_mode: str = "term_specific",
        discriminator_mode: str = "child_minus_parent",
    ) -> None:
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if num_domains <= 0:
            raise ValueError("num_domains must be positive")
        if num_local_layers <= 0:
            raise ValueError("num_local_layers must be positive")
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(ensemble_size)
        self.attention_dim = int(attention_dim)
        self.num_domains = int(num_domains)
        self.num_local_layers = int(num_local_layers)
        self.layer_contrast = bool(layer_contrast)
        if self.layer_contrast and self.num_local_layers < 2:
            raise ValueError(
                "Latent-domain layer contrast requires at least two "
                "local ESM layers"
            )
        self.competitive_segment_assignment = bool(
            competitive_segment_assignment
        )
        self.label_readout = bool(label_readout)
        if label_selection_mode not in {"term_specific", "uniform_domains"}:
            raise ValueError("Unknown latent-domain label selection mode")
        if discriminator_mode not in {"child_minus_parent", "prototype"}:
            raise ValueError("Unknown latent-domain discriminator mode")
        self.label_selection_mode = label_selection_mode
        self.discriminator_mode = discriminator_mode
        self.normalize_label_queries = bool(normalize_label_queries)
        self.adaptive_label_query_norm = bool(adaptive_label_query_norm)
        self.confidence_direction_routing = bool(
            confidence_direction_routing
        )
        self.ic_semantic_residual = bool(ic_semantic_residual)
        if self.normalize_label_queries and not self.label_readout:
            raise ValueError(
                "Normalized latent-domain label queries require label readout"
            )
        if self.adaptive_label_query_norm and not self.label_readout:
            raise ValueError(
                "Adaptive latent-domain query normalization requires "
                "label readout"
            )
        if self.confidence_direction_routing and not self.label_readout:
            raise ValueError(
                "Confidence-direction routing requires label readout"
            )
        if self.ic_semantic_residual and not self.label_readout:
            raise ValueError(
                "IC-weighted semantic residual requires label readout"
            )
        if self.adaptive_label_query_norm and self.normalize_label_queries:
            raise ValueError(
                "Fixed and adaptive label-query normalization are mutually "
                "exclusive"
            )
        if self.confidence_direction_routing and (
            self.normalize_label_queries
            or self.adaptive_label_query_norm
        ):
            raise ValueError(
                "Confidence-direction routing and existing label-query "
                "normalizations are mutually exclusive"
            )
        if self.ic_semantic_residual and (
            self.normalize_label_queries
            or self.adaptive_label_query_norm
            or self.confidence_direction_routing
        ):
            raise ValueError(
                "IC-weighted semantic residual is a controlled SEHNet "
                "variable and cannot be combined with query normalization"
            )
        self.label_chunk_size = int(label_chunk_size)
        if self.label_chunk_size <= 0:
            raise ValueError("label_chunk_size must be positive")
        if self.label_readout and (
            label_depth_groups is None
            or parent_child_indices is None
            or parent_indices is None
            or parent_counts is None
        ):
            raise ValueError(
                "Label-aware latent-domain readout requires GO depth "
                "groups and direct parent edges"
            )
        if (
            self.label_readout
            and label_depth_groups is not None
            and parent_counts is not None
            and (
                label_depth_groups.ndim != 1
                or parent_counts.shape != label_depth_groups.shape
            )
        ):
            raise ValueError("Invalid GO label-readout buffers")
        if self.ic_semantic_residual and (
            label_semantics is None
            or label_semantic_weights is None
            or label_semantics.shape
            != (len(label_depth_groups), int(semantic_dim))
            or label_semantic_weights.shape != label_depth_groups.shape
        ):
            raise ValueError(
                "IC-weighted semantic residual requires term-aligned "
                "semantic vectors and weights"
            )

        # Normalize each ESM layer separately before uniform fusion and
        # projection into a shared latent space.
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.local_dim)
            for _ in range(self.num_local_layers)
        )
        self.local_projection = nn.Linear(
            self.local_dim,
            self.attention_dim,
            bias=False,
        )
        if self.layer_contrast:
            # The fixed mean stream remains the exact SEHNet input.  A
            # zero-initialized late-minus-early projection can learn the
            # signed representation transition that a convex layer mixture
            # cannot express, while preserving exact SEHNet initialization.
            # Do not advance the RNG used by SEHNet's downstream domain queries
            # and prototype projection.  This keeps every shared parameter
            # bit-identical under the same seed.
            with torch.random.fork_rng(devices=[]):
                self.layer_contrast_projection = nn.Linear(
                    self.local_dim,
                    self.attention_dim,
                    bias=False,
                )
            nn.init.zeros_(self.layer_contrast_projection.weight)
        else:
            self.layer_contrast_projection = None
        self.domain_queries = nn.Parameter(
            torch.empty(self.num_domains, self.attention_dim)
        )
        nn.init.normal_(
            self.domain_queries,
            mean=0.0,
            std=self.attention_dim**-0.5,
        )
        self.domain_query_norm = nn.LayerNorm(self.attention_dim)
        self.domain_norm = nn.LayerNorm(self.attention_dim)
        if self.label_readout:
            self.global_norm = None
            self.global_query = None
            self.evidence_output = None
            self.register_parameter("member_scale", None)
            self.prototype_proj = nn.Linear(
                self.hidden_dim,
                self.attention_dim,
                bias=False,
            )
            self.depth_gates = nn.Parameter(torch.zeros(4))
            if self.ic_semantic_residual:
                # The zero scalar makes initialization exactly SEHNet.  A
                # private RNG context prevents the new projection from
                # changing any downstream shared initialization.
                with torch.random.fork_rng(devices=[]):
                    self.semantic_projection = nn.Linear(
                        int(semantic_dim),
                        self.attention_dim,
                        bias=False,
                    )
                self.semantic_residual_gate = nn.Parameter(torch.zeros(()))
            else:
                self.semantic_projection = None
                self.register_parameter("semantic_residual_gate", None)
            if self.adaptive_label_query_norm:
                # lambda=tanh(raw) controls how much of the task-prototype
                # magnitude is retained for each GO depth group.  lambda=0
                # is exactly SEHNet; lambda=1 approaches RMS-normalized routing
                # without discarding the direction learned by the classifier.
                self.query_norm_exponents = nn.Parameter(torch.zeros(4))
            else:
                self.register_parameter("query_norm_exponents", None)
        else:
            self.global_norm = nn.LayerNorm(self.hidden_dim)
            self.global_query = nn.Linear(
                self.hidden_dim,
                self.attention_dim,
                bias=False,
            )
            self.evidence_output = nn.Linear(
                self.attention_dim,
                self.hidden_dim,
                bias=False,
            )
            self.member_scale = nn.Parameter(torch.zeros(self.ensemble_size))
            self.prototype_proj = None
            self.register_parameter("depth_gates", None)
            self.register_parameter("query_norm_exponents", None)
            self.semantic_projection = None
            self.register_parameter("semantic_residual_gate", None)
        self.register_buffer(
            "label_semantics",
            (
                label_semantics.float()
                if label_semantics is not None
                else torch.empty(0, int(semantic_dim))
            ),
            persistent=False,
        )
        self.register_buffer(
            "label_semantic_weights",
            (
                label_semantic_weights.float()
                if label_semantic_weights is not None
                else torch.empty(0)
            ),
            persistent=self.ic_semantic_residual,
        )
        self.register_buffer(
            "label_depth_groups",
            (
                label_depth_groups.long()
                if label_depth_groups is not None
                else torch.empty(0, dtype=torch.long)
            ),
            persistent=self.label_readout,
        )
        self.register_buffer(
            "parent_child_indices",
            (
                parent_child_indices.long()
                if parent_child_indices is not None
                else torch.empty(0, dtype=torch.long)
            ),
            persistent=self.label_readout,
        )
        self.register_buffer(
            "parent_indices",
            (
                parent_indices.long()
                if parent_indices is not None
                else torch.empty(0, dtype=torch.long)
            ),
            persistent=self.label_readout,
        )
        self.register_buffer(
            "parent_counts",
            (
                parent_counts.long()
                if parent_counts is not None
                else torch.empty(0, dtype=torch.long)
            ),
            persistent=self.label_readout,
        )

    def forward(
        self,
        features: torch.Tensor,
        local_segments: torch.Tensor,
        local_mask: torch.Tensor,
        label_prototypes: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        expected_shape = (
            features.shape[0],
            self.num_local_layers,
            local_mask.shape[-1],
            self.local_dim,
        )
        if tuple(local_segments.shape) != expected_shape:
            raise ValueError(
                "Expected multi-layer local segments with shape "
                f"{expected_shape}, found {tuple(local_segments.shape)}"
            )
        if (
            features.ndim != 3
            or features.shape[1] != self.ensemble_size
            or features.shape[2] != self.hidden_dim
        ):
            raise ValueError(
                "Expected hypothesis features with shape "
                f"(batch,{self.ensemble_size},{self.hidden_dim}), "
                f"found {tuple(features.shape)}"
            )
        if tuple(local_mask.shape) != (
            features.shape[0],
            local_segments.shape[2],
        ):
            raise ValueError(
                f"Local mask shape mismatch: {tuple(local_mask.shape)}"
            )

        mask = local_mask.bool()
        valid = mask[:, None, :]
        # Missing sequence features remain exact zeros through LayerNorm and
        # the bias-free projection, so the fusion is an exact identity.
        normalized_layers = torch.stack(
            [
                self.layer_norms[layer_index](
                    local_segments[:, layer_index]
                )
                for layer_index in range(self.num_local_layers)
            ],
            dim=1,
        )
        normalized_segments = normalized_layers.mean(dim=1)
        projected_segments = self.local_projection(normalized_segments)
        if self.layer_contrast_projection is not None:
            layer_transition = (
                normalized_layers[:, -1] - normalized_layers[:, 0]
            ) * (2.0**-0.5)
            projected_segments = projected_segments + (
                self.layer_contrast_projection(layer_transition)
            )

        domain_queries = self.domain_query_norm(self.domain_queries)
        segment_logits = torch.einsum(
            "md,bsd->bms",
            domain_queries.to(projected_segments.dtype),
            projected_segments,
        ) * (self.attention_dim**-0.5)
        # Competitive allocation distributes each segment across queries,
        # then normalizes valid segment weights within each query for pooling.
        segment_weights = torch.softmax(
            segment_logits.float().masked_fill(~valid, -1e4),
            dim=(
                -2
                if self.competitive_segment_assignment
                else -1
            ),
        )
        segment_weights = segment_weights * valid.float()
        segment_weights = segment_weights / segment_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        domain_tokens = torch.einsum(
            "bms,bsd->bmd",
            segment_weights.to(projected_segments.dtype),
            projected_segments,
        )

        domain_keys = self.domain_norm(domain_tokens)
        present = mask.any(dim=-1)
        local_residual = None
        label_evidence = None
        if self.label_readout:
            if (
                label_prototypes is None
                or label_prototypes.ndim != 2
                or label_prototypes.shape
                != (len(self.label_depth_groups), self.hidden_dim)
            ):
                raise ValueError(
                    "GO-label latent-domain readout requires classifier "
                    "prototypes in model term order"
                )
            assert self.prototype_proj is not None
            assert self.depth_gates is not None
            # Keep the small label-prototype path in FP32.  Once a depth gate
            # opens, FP16 normalization gradients can otherwise overflow and
            # contaminate the shared 1024x64 projection.
            with torch.autocast(
                device_type=label_prototypes.device.type,
                enabled=False,
            ):
                locators = self.prototype_proj(
                    label_prototypes.detach().float()
                )
                if self.semantic_projection is not None:
                    assert self.semantic_residual_gate is not None
                    semantic_locators = self.semantic_projection(
                        self.label_semantics.float()
                    )
                    locators = locators + (
                        torch.tanh(self.semantic_residual_gate)
                        * self.label_semantic_weights[:, None]
                        * semantic_locators
                    )
            # Scaled dot-product attention assumes query and key components
            # have comparable variance.  Classifier-derived locators can have
            # very small norms, making the label-to-domain softmax effectively
            # uniform after the 1/sqrt(d) scale.  Parameter-free LayerNorm
            # restores the intended query geometry without changing the
            # child-minus-parent discriminator used to score local evidence.
            label_confidences = None
            if self.confidence_direction_routing:
                # A task-supervised label prototype contains two distinct
                # signals.  Its direction locates compatible protein domains,
                # while its norm carries label reliability.  Standard dot-
                # product attention entangles both: small reliable directions
                # become uniform, whereas full normalization treats every
                # label as equally trustworthy.  CDOR factorizes the two with
                # no trainable parameters.  sqrt(d) reproduces the effective
                # cosine scale of LayerNorm query/key attention.
                attention_locators = F.normalize(
                    locators,
                    dim=-1,
                    eps=1e-6,
                ) * (self.attention_dim**0.5)
                detached_norms = locators.detach().norm(
                    dim=-1
                ).clamp_min(1e-6)
                depth_square_sums = detached_norms.new_zeros(4)
                depth_square_sums.index_add_(
                    0,
                    self.label_depth_groups,
                    detached_norms.square(),
                )
                depth_counts = torch.bincount(
                    self.label_depth_groups,
                    minlength=4,
                ).to(detached_norms.dtype)
                depth_rms = (
                    depth_square_sums
                    / depth_counts.clamp_min(1.0)
                ).sqrt().clamp_min(1e-6)
                label_confidences = (
                    detached_norms
                    / depth_rms[self.label_depth_groups]
                )
            elif self.normalize_label_queries:
                attention_locators = F.layer_norm(
                    locators,
                    (self.attention_dim,),
                )
            elif self.adaptive_label_query_norm:
                assert self.query_norm_exponents is not None
                rms = locators.square().mean(dim=-1).add(1e-8).sqrt()
                exponents = torch.tanh(self.query_norm_exponents)[
                    self.label_depth_groups
                ]
                attention_locators = locators * rms.pow(-exponents)[:, None]
            else:
                attention_locators = locators
            if self.discriminator_mode == "child_minus_parent":
                parent_sums = torch.zeros_like(locators)
                parent_sums.index_add_(
                    0,
                    self.parent_child_indices,
                    locators[self.parent_indices],
                )
                counts = self.parent_counts.clamp_min(1).to(locators.dtype)
                discriminator_logits = locators - parent_sums / counts[:, None]
            else:
                discriminator_logits = locators
            # Normalize before masking.  Masking first creates exact zero
            # vectors for labels without a retained direct parent; their
            # normalization backward is singular in FP16 even though their
            # forward evidence is zero.
            discriminators = F.normalize(
                discriminator_logits,
                dim=-1,
                eps=1e-6,
            )
            discriminators = discriminators * self.parent_counts.gt(0)[
                :, None
            ].to(discriminators.dtype)
            evidence_chunks = []
            entropy_sum = domain_keys.new_zeros((), dtype=torch.float32)
            for start in range(
                0,
                len(self.label_depth_groups),
                self.label_chunk_size,
            ):
                end = min(
                    start + self.label_chunk_size,
                    len(self.label_depth_groups),
                )
                if self.label_selection_mode == "term_specific":
                    label_logits = torch.einsum(
                        "bmd,cd->bcm",
                        domain_keys,
                        attention_locators[start:end].to(domain_keys.dtype),
                    ) * (self.attention_dim**-0.5)
                    label_weights = torch.softmax(label_logits.float(), dim=-1)
                else:
                    label_weights = domain_keys.new_full(
                        (features.shape[0], end - start, self.num_domains),
                        1.0 / self.num_domains,
                    )
                label_weights = (
                    label_weights
                    * present[:, None, None].to(label_weights.dtype)
                )
                selected = torch.einsum(
                    "bcm,bmd->bcd",
                    label_weights.to(domain_tokens.dtype),
                    domain_tokens,
                )
                selected = F.normalize(
                    selected.float(),
                    dim=-1,
                    eps=1e-6,
                )
                chunk_evidence = torch.einsum(
                    "bcd,cd->bc",
                    selected,
                    discriminators[start:end],
                )
                if label_confidences is not None:
                    chunk_evidence = (
                        chunk_evidence
                        * label_confidences[start:end][None, :]
                    )
                evidence_chunks.append(
                    chunk_evidence
                    * present[:, None].to(chunk_evidence.dtype)
                )
                entropy_sum = entropy_sum + (
                    -label_weights
                    * label_weights.clamp_min(1e-8).log()
                ).sum()
            label_evidence = torch.cat(evidence_chunks, dim=-1)
            label_gates = torch.tanh(self.depth_gates)[
                self.label_depth_groups
            ]
            label_evidence = (
                label_evidence
                * label_gates[None, :].to(label_evidence.dtype)
            )
            domain_entropy = entropy_sum / (
                present.sum().clamp_min(1).to(entropy_sum.dtype)
                * float(len(self.label_depth_groups))
            )
            fused = features
        else:
            assert self.global_norm is not None
            assert self.global_query is not None
            assert self.evidence_output is not None
            assert self.member_scale is not None
            # Stop-gradient on this query path protects the 3B global
            # anchor; the identity path below still trains normally.
            hypothesis_queries = self.global_query(
                self.global_norm(features.detach())
            )
            domain_logits = torch.einsum(
                "bkd,bmd->bkm",
                hypothesis_queries,
                domain_keys,
            ) * (self.attention_dim**-0.5)
            domain_weights = torch.softmax(domain_logits.float(), dim=-1)
            domain_weights = (
                domain_weights
                * present[:, None, None].to(domain_weights.dtype)
            )
            selected_domains = torch.einsum(
                "bkm,bmd->bkd",
                domain_weights.to(domain_tokens.dtype),
                domain_tokens,
            )
            local_residual = self.evidence_output(selected_domains)
            gate = torch.tanh(self.member_scale)
            fused = features + gate[None, :, None] * local_residual
            domain_entropy = -(
                domain_weights
                * domain_weights.clamp_min(1e-8).log()
            ).sum(dim=-1)
            domain_entropy = (
                domain_entropy.sum()
                / present.sum().clamp_min(1).to(domain_entropy.dtype)
                / float(self.ensemble_size)
            )

        segment_entropy = -(
            segment_weights
            * segment_weights.clamp_min(1e-8).log()
        ).sum(dim=-1)
        segment_entropy = (
            segment_entropy.sum()
            / present.sum().clamp_min(1).to(segment_entropy.dtype)
            / float(self.num_domains)
        )
        return (
            fused,
            local_residual,
            label_evidence,
            segment_entropy,
            domain_entropy,
        )

    @torch.no_grad()
    def mean_query_cosine(self) -> torch.Tensor:
        normalized = F.normalize(self.domain_queries.float(), dim=-1)
        similarities = normalized @ normalized.T
        if self.num_domains == 1:
            return similarities.new_zeros(())
        off_diagonal = ~torch.eye(
            self.num_domains,
            dtype=torch.bool,
            device=similarities.device,
        )
        return similarities[off_diagonal].mean()

    @torch.no_grad()
    def zero_evidence_gate(self) -> None:
        if self.label_readout:
            assert self.depth_gates is not None
            self.depth_gates.zero_()
        else:
            assert self.member_scale is not None
            self.member_scale.zero_()

    @torch.no_grad()
    def gate_abs_mean(self) -> torch.Tensor:
        gate = (
            self.depth_gates
            if self.label_readout
            else self.member_scale
        )
        assert gate is not None
        return torch.tanh(gate).abs().mean()

    @torch.no_grad()
    def semantic_residual_scale(self) -> torch.Tensor:
        if self.semantic_residual_gate is None:
            return self.domain_queries.new_zeros(())
        return torch.tanh(self.semantic_residual_gate)


@dataclass
class HypothesisEnsembleOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    member_logits: torch.Tensor
    member_probs: torch.Tensor
    features: torch.Tensor
    breadth_score: torch.Tensor | None = None
    breadth_offset: torch.Tensor | None = None
    routing_weights: torch.Tensor | None = None
    routing_entropy: torch.Tensor | None = None
    local_evidence: torch.Tensor | None = None
    latent_domain_evidence: torch.Tensor | None = None
    latent_domain_label_evidence: torch.Tensor | None = None
    latent_domain_segment_entropy: torch.Tensor | None = None
    latent_domain_selection_entropy: torch.Tensor | None = None
    anchor_member_logits: torch.Tensor | None = None


class FunctionalHypothesisEnsemble(nn.Module):
    """Residual MLP whose hypotheses share weights but keep rank-one adapters."""

    def __init__(
        self,
        class_nums: int,
        esm_dim: int = 2560,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        ensemble_size: int = 8,
        classifier_rank: int = 0,
        independent_classifier: bool = False,
        functional_breadth: bool = False,
        breadth_target_mean: float = 0.0,
        breadth_target_std: float = 1.0,
        semantic_router: bool = False,
        go_semantics: torch.Tensor | None = None,
        router_temperature: float = 1.0,
        function_alignment: bool = False,
        function_alignment_dim: int = 128,
        function_semantic_dim: int = 768,
        gated_refinement: bool = False,
        gated_refinement_expansion: float = 1.0,
        local_semantic_attention: bool = False,
        local_segment_dim: int = 640,
        local_attention_dim: int = 64,
        local_label_chunk_size: int = 512,
        local_task_query_residual: bool = False,
        local_query_source: str = "semantic",
        local_num_layers: int = 1,
        local_classifier_parent_contrast: bool = True,
        local_go_semantics: torch.Tensor | None = None,
        local_parent_contrast: torch.Tensor | None = None,
        local_depth_groups: torch.Tensor | None = None,
        local_parent_child_indices: torch.Tensor | None = None,
        local_parent_indices: torch.Tensor | None = None,
        local_parent_counts: torch.Tensor | None = None,
        latent_domain_fusion: bool = False,
        latent_domain_count: int = 8,
        latent_domain_dim: int = 64,
        latent_domain_num_layers: int = 3,
        latent_domain_layer_contrast: bool = False,
        latent_domain_competition: bool = False,
        latent_domain_label_readout: bool = False,
        latent_domain_normalize_label_queries: bool = False,
        latent_domain_adaptive_label_query_norm: bool = False,
        latent_domain_confidence_direction_routing: bool = False,
        latent_domain_ic_semantic_residual: bool = False,
        latent_label_semantics: torch.Tensor | None = None,
        latent_label_semantic_weights: torch.Tensor | None = None,
        latent_semantic_dim: int = 768,
        latent_label_chunk_size: int = 512,
        latent_label_depth_groups: torch.Tensor | None = None,
        latent_parent_child_indices: torch.Tensor | None = None,
        latent_parent_indices: torch.Tensor | None = None,
        latent_parent_counts: torch.Tensor | None = None,
        latent_domain_readout_mode: str = "term_specific",
        latent_domain_discriminator_mode: str = "child_minus_parent",
        anchor_preserving_local_gradients: bool = False,
    ) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2")
        self.ensemble_size = int(ensemble_size)
        self.classifier_rank = int(classifier_rank)
        self.independent_classifier = bool(independent_classifier)
        if self.independent_classifier and self.classifier_rank > 0:
            raise ValueError(
                "independent_classifier and classifier_rank are mutually exclusive"
            )
        self.functional_breadth = bool(functional_breadth)
        self.semantic_router_enabled = bool(semantic_router)
        self.function_alignment_enabled = bool(function_alignment)
        self.gated_refinement_enabled = bool(gated_refinement)
        self.local_semantic_attention_enabled = bool(
            local_semantic_attention
        )
        self.latent_domain_fusion_enabled = bool(latent_domain_fusion)
        self.latent_domain_layer_contrast = bool(
            latent_domain_layer_contrast
        )
        if (
            self.latent_domain_layer_contrast
            and not self.latent_domain_fusion_enabled
        ):
            raise ValueError(
                "Latent-domain layer contrast requires latent-domain fusion"
            )
        self.latent_domain_label_readout = bool(
            latent_domain_label_readout
        )
        self.latent_domain_normalize_label_queries = bool(
            latent_domain_normalize_label_queries
        )
        self.latent_domain_adaptive_label_query_norm = bool(
            latent_domain_adaptive_label_query_norm
        )
        self.latent_domain_confidence_direction_routing = bool(
            latent_domain_confidence_direction_routing
        )
        self.latent_domain_ic_semantic_residual = bool(
            latent_domain_ic_semantic_residual
        )
        if (
            self.latent_domain_normalize_label_queries
            and not self.latent_domain_label_readout
        ):
            raise ValueError(
                "Normalized latent-domain label queries require GO-label "
                "readout"
            )
        if (
            self.latent_domain_adaptive_label_query_norm
            and not self.latent_domain_label_readout
        ):
            raise ValueError(
                "Adaptive latent-domain label-query normalization requires "
                "GO-label readout"
            )
        if (
            self.latent_domain_confidence_direction_routing
            and not self.latent_domain_label_readout
        ):
            raise ValueError(
                "Confidence-direction latent-domain routing requires "
                "GO-label readout"
            )
        if (
            self.latent_domain_ic_semantic_residual
            and not self.latent_domain_label_readout
        ):
            raise ValueError(
                "IC-weighted semantic residual requires GO-label readout"
            )
        if (
            self.latent_domain_adaptive_label_query_norm
            and self.latent_domain_normalize_label_queries
        ):
            raise ValueError(
                "Fixed and adaptive latent-domain query normalization are "
                "mutually exclusive"
            )
        if self.latent_domain_confidence_direction_routing and (
            self.latent_domain_normalize_label_queries
            or self.latent_domain_adaptive_label_query_norm
        ):
            raise ValueError(
                "Confidence-direction routing and label-query "
                "normalization are mutually exclusive"
            )
        if self.latent_domain_ic_semantic_residual and (
            self.latent_domain_normalize_label_queries
            or self.latent_domain_adaptive_label_query_norm
            or self.latent_domain_confidence_direction_routing
        ):
            raise ValueError(
                "IC-weighted semantic residual cannot be combined with "
                "label-query normalization"
            )
        self.anchor_preserving_local_gradients = bool(
            anchor_preserving_local_gradients
        )
        if (
            self.latent_domain_label_readout
            and not self.latent_domain_fusion_enabled
        ):
            raise ValueError(
                "Latent-domain GO-label readout requires latent-domain "
                "fusion"
            )
        if (
            self.local_semantic_attention_enabled
            and self.latent_domain_fusion_enabled
        ):
            raise ValueError(
                "Local semantic attention and latent-domain fusion are "
                "mutually exclusive"
            )
        if (
            self.anchor_preserving_local_gradients
            and not self.latent_domain_fusion_enabled
        ):
            raise ValueError(
                "Anchor-preserving local gradients require "
                "latent-domain fusion"
            )
        self.router_temperature = float(router_temperature)
        if self.router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        self.input_block = EnsembleBlock(
            esm_dim,
            hidden_dim,
            ensemble_size,
            dropout,
            diversify_input=True,
        )
        self.residual_block = EnsembleBlock(
            hidden_dim,
            hidden_dim,
            ensemble_size,
            dropout,
        )
        # Instantiate the shared classifier before local-evidence
        # modules.  With a fixed seed this keeps every shared parameter
        # bit-identical between a global-only and full model, so a
        # paired from-scratch comparison does not confound the new module
        # with a different classifier initialization.
        if self.independent_classifier:
            self.classifier = IndependentHypothesisClassifier(
                hidden_dim,
                class_nums,
                ensemble_size,
            )
        else:
            self.classifier = BatchEnsembleLinear(
                hidden_dim,
                class_nums,
                ensemble_size,
                residual_rank=self.classifier_rank,
            )
        if self.local_semantic_attention_enabled:
            if (
                local_go_semantics is None
                or local_parent_contrast is None
                or local_depth_groups is None
            ):
                raise ValueError(
                    "Local semantic attention requires GO semantics, "
                    "parent contrasts and depth groups"
                )
            self.local_semantic_attention = (
                ParentContrastiveLocalAttention(
                    local_dim=local_segment_dim,
                    attention_dim=local_attention_dim,
                    go_semantics=local_go_semantics,
                    parent_contrast=local_parent_contrast,
                    depth_groups=local_depth_groups,
                    label_chunk_size=local_label_chunk_size,
                    task_query_residual=local_task_query_residual,
                    query_source=local_query_source,
                    num_local_layers=local_num_layers,
                    classifier_parent_contrast=(
                        local_classifier_parent_contrast
                    ),
                    classifier_dim=hidden_dim,
                    parent_child_indices=local_parent_child_indices,
                    parent_indices=local_parent_indices,
                    parent_counts=local_parent_counts,
                )
            )
        else:
            self.local_semantic_attention = None
        if self.latent_domain_fusion_enabled:
            # Optional-module initialization must not advance the global CPU
            # RNG, otherwise an otherwise identical baseline receives a
            # different shuffled training order.
            with torch.random.fork_rng(devices=[]):
                self.latent_domain_fusion = LatentDomainLocalFusion(
                    local_dim=local_segment_dim,
                    hidden_dim=hidden_dim,
                    ensemble_size=ensemble_size,
                    attention_dim=latent_domain_dim,
                    num_domains=latent_domain_count,
                    num_local_layers=latent_domain_num_layers,
                    layer_contrast=self.latent_domain_layer_contrast,
                    competitive_segment_assignment=(
                        latent_domain_competition
                    ),
                    label_readout=self.latent_domain_label_readout,
                    normalize_label_queries=(
                        self.latent_domain_normalize_label_queries
                    ),
                    adaptive_label_query_norm=(
                        self.latent_domain_adaptive_label_query_norm
                    ),
                    confidence_direction_routing=(
                        self.latent_domain_confidence_direction_routing
                    ),
                    ic_semantic_residual=(
                        self.latent_domain_ic_semantic_residual
                    ),
                    label_semantics=latent_label_semantics,
                    label_semantic_weights=(
                        latent_label_semantic_weights
                    ),
                    semantic_dim=latent_semantic_dim,
                    label_chunk_size=latent_label_chunk_size,
                    label_depth_groups=latent_label_depth_groups,
                    parent_child_indices=latent_parent_child_indices,
                    parent_indices=latent_parent_indices,
                    parent_counts=latent_parent_counts,
                    label_selection_mode=latent_domain_readout_mode,
                    discriminator_mode=latent_domain_discriminator_mode,
                )
        else:
            self.latent_domain_fusion = None
        if self.functional_breadth:
            breadth_hidden = max(64, hidden_dim // 4)
            self.breadth_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, breadth_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(breadth_hidden, 1),
            )
            self.breadth_gain = nn.Parameter(torch.zeros(()))
        else:
            self.breadth_head = None
            self.register_parameter("breadth_gain", None)
        if self.semantic_router_enabled:
            if go_semantics is None:
                raise ValueError(
                    "go_semantics is required when semantic_router is enabled"
                )
            if go_semantics.ndim != 2 or go_semantics.shape[0] != class_nums:
                raise ValueError(
                    "Expected GO semantics with shape "
                    f"({class_nums}, semantic_dim), found "
                    f"{tuple(go_semantics.shape)}"
                )
            normalized_semantics = go_semantics.float()
            normalized_semantics = (
                normalized_semantics
                - normalized_semantics.mean(dim=0, keepdim=True)
            )
            normalized_semantics = F.normalize(
                normalized_semantics,
                dim=-1,
            )
            self.register_buffer(
                "go_semantics",
                normalized_semantics,
                persistent=False,
            )
            self.protein_router = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )
            self.go_router = nn.Linear(
                normalized_semantics.shape[1],
                ensemble_size,
            )
            nn.init.zeros_(self.protein_router[-1].weight)
            nn.init.zeros_(self.protein_router[-1].bias)
            nn.init.zeros_(self.go_router.weight)
            nn.init.zeros_(self.go_router.bias)
        else:
            self.register_buffer("go_semantics", None, persistent=False)
            self.protein_router = None
            self.go_router = None
        if self.function_alignment_enabled:
            if function_alignment_dim <= 0:
                raise ValueError("function_alignment_dim must be positive")
            self.protein_alignment_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, function_alignment_dim),
            )
            self.function_alignment_head = nn.Sequential(
                nn.LayerNorm(function_semantic_dim),
                nn.Linear(function_semantic_dim, function_alignment_dim),
            )
        else:
            self.protein_alignment_head = None
            self.function_alignment_head = None
        if self.gated_refinement_enabled:
            if gated_refinement_expansion <= 0.0:
                raise ValueError(
                    "gated_refinement_expansion must be positive"
                )
            self.gated_refinement = ReZeroSwiGLURefinement(
                hidden_dim=hidden_dim,
                ensemble_size=ensemble_size,
                expansion=gated_refinement_expansion,
                dropout=dropout,
            )
        else:
            self.gated_refinement = None
        self.register_buffer(
            "breadth_target_mean",
            torch.tensor(float(breadth_target_mean)),
            persistent=False,
        )
        self.register_buffer(
            "breadth_target_std",
            torch.tensor(max(float(breadth_target_std), 1e-6)),
            persistent=False,
        )

    def forward(
        self,
        esm: torch.Tensor,
        local_segments: torch.Tensor | None = None,
        local_mask: torch.Tensor | None = None,
    ) -> HypothesisEnsembleOutput:
        views = esm.unsqueeze(1).expand(-1, self.ensemble_size, -1)
        features = self.input_block(views)
        features = features + self.residual_block(features)
        anchor_features = features
        if (
            self.local_semantic_attention is not None
            or self.latent_domain_fusion is not None
        ):
            if local_segments is None or local_mask is None:
                raise ValueError(
                    "local_segments and local_mask are required by this model"
                )
        latent_domain_evidence = None
        latent_domain_label_evidence = None
        latent_domain_segment_entropy = None
        latent_domain_selection_entropy = None
        if self.latent_domain_fusion is not None:
            (
                features,
                latent_domain_evidence,
                latent_domain_label_evidence,
                latent_domain_segment_entropy,
                latent_domain_selection_entropy,
            ) = self.latent_domain_fusion(
                features,
                local_segments,
                local_mask,
                (
                    self.classifier.weight
                    if self.latent_domain_label_readout
                    else None
                ),
            )
        if self.gated_refinement is not None:
            features = self.gated_refinement(features)
        anchor_member_logits = None
        if self.anchor_preserving_local_gradients and self.training:
            # Numerically retain the ordinary fused feature tensor while
            # removing its identity-path gradient into the global anchor.
            # The detached classifier copy then sends this loss only into
            # the latent-domain residual and its zero-initialized gate.
            local_training_features = (
                anchor_features.detach()
                + (features - anchor_features)
            )
            member_logits = self.classifier.forward_detached(
                local_training_features
            )
            # Global-only prediction path.
            # Its supervised loss is the sole update source for the shared
            # global encoder and classifier during anchor-preserving training.
            anchor_member_logits = self.classifier(anchor_features)
        else:
            member_logits = self.classifier(features)
        if latent_domain_label_evidence is not None:
            member_logits = (
                member_logits + latent_domain_label_evidence[:, None, :]
            )
        local_evidence = None
        if self.local_semantic_attention is not None:
            local_evidence = self.local_semantic_attention(
                local_segments,
                local_mask,
                (
                    self.classifier.weight
                    if (
                        self.local_semantic_attention.query_source
                        == "classifier"
                    )
                    else None
                ),
            )
            member_logits = member_logits + local_evidence[:, None, :]
        breadth_score = None
        breadth_offset = None
        if self.breadth_head is not None:
            breadth_score = self.breadth_head(features.mean(dim=1)).squeeze(-1)
            breadth_offset = self.breadth_gain * breadth_score
            member_logits = member_logits + breadth_offset[:, None, None]
        member_probs = torch.sigmoid(member_logits)
        routing_weights = None
        routing_entropy = None
        if self.semantic_router_enabled:
            # Keep the sparse extreme-label mixture in FP32 under AMP so
            # low-probability labels still provide a gradient to the router.
            protein_route_logits = self.protein_router(
                features.float()
            ).squeeze(-1)
            go_route_logits = self.go_router(self.go_semantics.float())
            routing_logits = (
                protein_route_logits[:, None, :]
                + go_route_logits[None, :, :]
            )
            routing_weights = torch.softmax(
                routing_logits / self.router_temperature,
                dim=-1,
            )
            probs = torch.einsum(
                "bck,bkc->bc",
                routing_weights,
                member_probs.float(),
            )
            routing_entropy = (
                -routing_weights
                * routing_weights.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        else:
            probs = member_probs.mean(dim=1)
        eps = 1e-7 if probs.dtype == torch.float32 else torch.finfo(probs.dtype).eps
        logits = torch.logit(probs.clamp(eps, 1.0 - eps))
        return HypothesisEnsembleOutput(
            logits=logits,
            probs=probs,
            member_logits=member_logits,
            member_probs=member_probs,
            features=features,
            breadth_score=breadth_score,
            breadth_offset=breadth_offset,
            routing_weights=routing_weights,
            routing_entropy=routing_entropy,
            local_evidence=local_evidence,
            latent_domain_evidence=latent_domain_evidence,
            latent_domain_label_evidence=latent_domain_label_evidence,
            latent_domain_segment_entropy=(
                latent_domain_segment_entropy
            ),
            latent_domain_selection_entropy=(
                latent_domain_selection_entropy
            ),
            anchor_member_logits=anchor_member_logits,
        )

    def alignment_embeddings(
        self,
        protein_features: torch.Tensor,
        function_semantics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project protein evidence and its annotated function set."""

        if (
            self.protein_alignment_head is None
            or self.function_alignment_head is None
        ):
            raise RuntimeError("Function alignment is not enabled")
        protein_embedding = F.normalize(
            self.protein_alignment_head(protein_features.float()),
            dim=-1,
        )
        function_embedding = F.normalize(
            self.function_alignment_head(function_semantics.float()),
            dim=-1,
        )
        return protein_embedding, function_embedding


SEHNet = FunctionalHypothesisEnsemble
