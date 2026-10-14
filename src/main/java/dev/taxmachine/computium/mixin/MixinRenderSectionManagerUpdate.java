package dev.taxmachine.computium.mixin;

import org.embeddedt.embeddium.impl.render.chunk.RenderSectionManager;
import org.embeddedt.embeddium.impl.render.viewport.Viewport;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Phase-1 per-frame hook: runs the GPU frustum pass once per frame, before
 * Beddium walks the section graph. Non-cancellable by design — Beddium
 * always proceeds normally; the compute result is purely additive (the
 * cull hook consults it, everything else is untouched). Any failure
 * inside latches the engine off and the game keeps the stock path.
 *
 * <p>Injector params mirror the target EXACTLY (Mixin 0.8.7 validates by
 * string equality — no {@code this} param, no {@code Object} placeholders;
 * the manager instance is {@code this}, cast to the target type).
 * Beddium is a compileOnly reference (never shipped); on Beddium-absent
 * paths the config ({@code required: false}) skips this mixin silently.
 */
@Mixin(
    targets = "org.embeddedt.embeddium.impl.render.chunk.RenderSectionManager",
    remap = false
)
public abstract class MixinRenderSectionManagerUpdate {

    @Inject(
        method = "update(Lorg/embeddedt/embeddium/impl/render/viewport/Viewport;IZ)V",
        at = @At("HEAD"),
        remap = false
    )
    private void computium$onUpdate(Viewport viewport, int frame, boolean spectator,
            CallbackInfo ci) {
        dev.taxmachine.computium.ComputiumCompute.frameSample();
        if (dev.taxmachine.computium.ComputiumConfig.enableGpuFrustumCulling) {
            RenderSectionManager manager = (RenderSectionManager) (Object) this;
            dev.taxmachine.computium.ComputiumCompute.runFrame(manager, viewport);
        }
    }
}
