package dev.taxmachine.computium.mixin;

import org.embeddedt.embeddium.impl.render.chunk.data.SectionRenderDataStorage;
import org.embeddedt.embeddium.impl.render.chunk.lists.ChunkRenderList;
import org.embeddedt.embeddium.impl.render.chunk.multidraw.MultiDrawEmitter;
import org.embeddedt.embeddium.impl.render.chunk.region.RenderRegion;
import org.embeddedt.embeddium.impl.render.chunk.terrain.TerrainRenderPass;
import org.embeddedt.embeddium.impl.render.viewport.CameraTransform;
import org.embeddedt.embeddium.impl.gl.device.CommandList;
import org.embeddedt.embeddium.impl.gl.tessellation.GlPrimitiveType;
import org.embeddedt.embeddium.impl.gl.tessellation.GlTessellation;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Phase-2 shadow hook: HEAD records the emitter's command count, TAIL
 * compares the commands Beddium added during the fill against an
 * independent CPU mirror of the mesh. Pure CPU — ZERO GL calls inside
 * Beddium's fill path (an earlier GPU-roundtrip shadow validated the
 * compute shader bit-exact 378/378, then proved too invasive to run per
 * fill: its synchronous upload/dispatch/readback perturbed the shared
 * GL command stream Beddium uses for mesh uploads). Unsorted passes
 * only; sorted/translucent keeps the CPU path.
 *
 * <p>Cutover (later, behind {@code enableGpuCommandGeneration}) will
 * cancel here and draw from the GPU buffer in an executeBatch hook.
 */
@Mixin(
    targets = "org.embeddedt.embeddium.impl.render.chunk.DefaultChunkRenderer",
    remap = false
)
public abstract class MixinDefaultChunkRendererFill {

    @Inject(
        method = "fillCommandBuffer(Lorg/embeddedt/embeddium/impl/render/chunk/multidraw/MultiDrawEmitter;Lorg/embeddedt/embeddium/impl/render/chunk/region/RenderRegion;Lorg/embeddedt/embeddium/impl/render/chunk/data/SectionRenderDataStorage;Lorg/embeddedt/embeddium/impl/render/chunk/lists/ChunkRenderList;Lorg/embeddedt/embeddium/impl/render/viewport/CameraTransform;Lorg/embeddedt/embeddium/impl/render/chunk/terrain/TerrainRenderPass;Z)V",
        at = @At("HEAD"),
        remap = false
    )
    private static void computium$onFillCommandBufferHead(MultiDrawEmitter emitter, RenderRegion region,
            SectionRenderDataStorage storage, ChunkRenderList renderList, CameraTransform camera,
            TerrainRenderPass pass, boolean useBlockFaceCulling, CallbackInfo ci) {
        // Snapshot first: packs descriptors + churn words from one read
        // pass (single word source for dispatch, mirror, heartbeat).
        // Runs when validation OR generation needs it.
        if (dev.taxmachine.computium.ComputiumConfig.enableGpuCommandValidation
                || dev.taxmachine.computium.ComputiumConfig.enableGpuCommandGeneration) {
            dev.taxmachine.computium.ComputiumCompute.snapshotFillState(
                    emitter, region, storage, renderList, pass);
        }
        if (dev.taxmachine.computium.ComputiumConfig.enableGpuCommandGeneration) {
            // Cutover: generate on GPU (async, no readback). Beddium's own
            // fill still runs below (emitter stays warm for fallback).
            dev.taxmachine.computium.ComputiumCompute.runGpuGen(
                    emitter, region, storage, renderList, camera, pass, useBlockFaceCulling);
        }
    }

    @Inject(
        method = "fillCommandBuffer(Lorg/embeddedt/embeddium/impl/render/chunk/multidraw/MultiDrawEmitter;Lorg/embeddedt/embeddium/impl/render/chunk/region/RenderRegion;Lorg/embeddedt/embeddium/impl/render/chunk/data/SectionRenderDataStorage;Lorg/embeddedt/embeddium/impl/render/chunk/lists/ChunkRenderList;Lorg/embeddedt/embeddium/impl/render/viewport/CameraTransform;Lorg/embeddedt/embeddium/impl/render/chunk/terrain/TerrainRenderPass;Z)V",
        at = @At("RETURN"),
        remap = false
    )
    private static void computium$onFillCommandBufferTail(MultiDrawEmitter emitter, RenderRegion region,
            SectionRenderDataStorage storage, ChunkRenderList renderList, CameraTransform camera,
            TerrainRenderPass pass, boolean useBlockFaceCulling, CallbackInfo ci) {
        if (dev.taxmachine.computium.ComputiumConfig.enableGpuCommandValidation) {
            // Shadow only: validate, never cancel (cutover flips this).
            // The face flag is Beddium's own per-region decision — mirror
            // it or the comparison is meaningless.
            dev.taxmachine.computium.ComputiumCompute.runCpuValidation(
                    emitter, region, storage, renderList, camera, pass, useBlockFaceCulling);
        }
    }

    @Redirect(
        method = "render(Lorg/embeddedt/embeddium/impl/render/chunk/ChunkRenderMatrices;Lorg/embeddedt/embeddium/impl/gl/device/CommandList;Lorg/embeddedt/embeddium/impl/render/chunk/lists/ChunkRenderListIterable;Lorg/embeddedt/embeddium/impl/render/chunk/terrain/TerrainRenderPass;Lorg/embeddedt/embeddium/impl/render/viewport/CameraTransform;Lorg/embeddedt/embeddium/impl/render/viewport/CameraTransform;)V",
        at = @At(
            value = "INVOKE",
            target = "Lorg/embeddedt/embeddium/impl/render/chunk/multidraw/MultiDrawEmitter;executeBatch(Lorg/embeddedt/embeddium/impl/gl/device/CommandList;Lorg/embeddedt/embeddium/impl/gl/tessellation/GlTessellation;Lorg/embeddedt/embeddium/impl/gl/tessellation/GlPrimitiveType;)V"
        ),
        remap = false
    )
    private void computium$redirectExecuteBatch(MultiDrawEmitter emitter, CommandList commandList,
            GlTessellation tessellation, GlPrimitiveType primitiveType) {
        // Cutover: a GPU-generated region draws indirect from our command
        // buffer; everything else (sorted passes, fallback, ungenerated)
        // takes Beddium's stock path untouched.
        if (dev.taxmachine.computium.ComputiumCompute.consumeDrewGpu()) {
            dev.taxmachine.computium.ComputiumCompute.drawIndirect(commandList, tessellation, primitiveType);
        } else {
            emitter.executeBatch(commandList, tessellation, primitiveType);
        }
    }
}
