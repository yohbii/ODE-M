#!/bin/bash

# number of samples on 3 architectures
for num_samples in 8 16 32 64 128 256 512 1024 2048; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/ode \
        fabric.loggers.name=vit_base_patch32_TA8_numsamples \
        fabric.loggers.version=${num_samples} \
        method=ode_merging/clip_ode_merging \
        method.num_samples=${num_samples} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done

for num_samples in 8 16 32 64 128 256 512 1024 2048; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/ode \
        fabric.loggers.name=vit_base_patch16_TA8 \
        fabric.loggers.version=${num_samples} \
        method=ode_merging/clip_ode_merging \
        method.num_samples=${num_samples} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done

for num_samples in 8 16 32 64 128 256 512 1024 2048; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/ode \
        fabric.loggers.name=vit_large_patch14_TA8 \
        fabric.loggers.version=${num_samples} \
        method=ode_merging/clip_ode_merging \
        method.num_samples=${num_samples} \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done


# Step Size
for step_size in 1e-3 2.5e-3 5e-3 1e-2 2.5e-2 5e-2 1e-1 2e-1; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/ode \
        fabric.loggers.name=vit_base_patch32_TA8_stepsize \
        fabric.loggers.version=${step_size} \
        method=ode_merging/clip_ode_merging \
        method.integration.step_size=${step_size} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done

for step_size in 1e-3 2.5e-3 5e-3 1e-2 2.5e-2 5e-2 1e-1 2e-1; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/ode \
        fabric.loggers.name=vit_base_patch16_TA8 \
        fabric.loggers.version=${step_size} \
        method=ode_merging/clip_ode_merging \
        method.integration.step_size=${step_size} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done

for step_size in 1e-3 2.5e-3 5e-3 1e-2 2.5e-2 5e-2 1e-1 2e-1; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/ode \
        fabric.loggers.name=vit_large_patch14_TA8 \
        fabric.loggers.version=${step_size} \
        method=ode_merging/clip_ode_merging \
        method.integration.step_size=${step_size} \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done