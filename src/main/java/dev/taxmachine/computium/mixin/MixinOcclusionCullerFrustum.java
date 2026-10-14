package dev.taxmachine.computium.mixin;

import org.embeddedt.embeddium.impl.render.chunk.occlusion.OcclusionNode;
import org.embeddedt.embeddium.impl.render.viewport.Viewport;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Phase-1 hook: Beddium's per-section visibility gate,
 * {@code OcclusionCuller.isSectionVisible} ({@code = isWithinRenderDistance
 * && isWithinFrustum}, private static, called for every candidate section
 * in the BFS). When the GPU bitset proves frustum-invisible, cancels with
 * {@code false}; everything else proceeds untouched, so traversal pruning
 * stays exactly Beddium's. See {@code ../ARCHITECTURE.md}.
 *
 * <p>Injector params use EXACT Beddium types (Mixin 0.8.7 validates
 * injector descriptors by string equality). Config
 * {@code computium.mixin.json} is {@code required: false}: Beddium-absent
 * boots must never break.
 */
@Mixin(
    targets = "org.embeddedt.embeddium.impl.render.chunk.occlusion.OcclusionCuller",
    remap = false
)
public abstract class MixinOcclusionCullerFrustum {

    private static boolean computium$logged = false;

    @Inject(
        method = "isSectionVisible(Lorg/embeddedt/embeddium/impl/render/chunk/occlusion/OcclusionNode;Lorg/embeddedt/embeddium/impl/render/viewport/Viewport;F)Z",
        at = @At("HEAD"),
        remap = false,
        cancellable = true
    )
    private static void computium$onIsSectionVisible(OcclusionNode node, Viewport viewport,
            float maxDistance, CallbackInfoReturnable<Boolean> cir) {
        if (!computium$logged) {
            computium$logged = true;
            // Note: this fires on Beddium's graph thread (no GL context
            // here), so no capability probe — just hook presence.
            System.out.println("[Computium] OcclusionCuller.isSectionVisible hooked"
                    + " (compute-gated=" + dev.taxmachine.computium.ComputiumConfig.enableGpuFrustumCulling + ")");
        }
        if (dev.taxmachine.computium.ComputiumCompute.shouldCullNode(node)) {
            cir.setReturnValue(Boolean.FALSE);
        }
    }
}
