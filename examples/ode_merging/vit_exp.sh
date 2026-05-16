#!/bin/bash

# CLIP-ViT-Base-patch32
fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_base_patch32_TA8 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_base_patch32_TALL14 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14

fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_base_patch32_TALL20 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20


# CLIP-ViT-Base-patch16
fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_base_patch16_TA8 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_base_patch16_TALL14 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14

fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_base_patch16_TALL20 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20


# CLIP-ViT-Large-patch14
fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_large_patch14_TA8 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_large_patch14_TALL14 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14

fusion_bench \
    fabric.loggers.root_dir=outputs/ode \
    fabric.loggers.name=vit_large_patch14_TALL20 \
    method=ode_merging/clip_ode_merging \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20