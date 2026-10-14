package dev.taxmachine.computium;

import java.io.File;
import net.minecraftforge.common.config.Configuration;

/**
 * Computium configuration. Everything defaults OFF until its phase proves
 * green on the matrix — an enabled flag without a proven path is how
 * you brick someone's game from a config file.
 */
public final class ComputiumConfig {
    private ComputiumConfig() {
    }

    /** Phase 1: GPU frustum culling (compute shader + readback). */
    public static boolean enableGpuFrustumCulling = false;

    /**
     * Phase 2 shadow validation: generate indirect commands on GPU and
     * compare against Beddium's CPU emitter (logs only — Beddium still
     * draws). Cutover comes later behind enableGpuCommandGeneration.
     */
    public static boolean enableGpuCommandValidation = false;

    /** Phase 2: GPU indirect-command generation (Nvidium-class). */
    public static boolean enableGpuCommandGeneration = false;

    public static void init(File file) {
        Configuration cfg = new Configuration(file);
        try {
            cfg.load();
            enableGpuFrustumCulling = cfg.getBoolean("enableGpuFrustumCulling", "general", false,
                    "Phase 1: cull chunk sections in a GL 4.3 compute shader. Requires Beddium + GL43+; falls back silently.");
            enableGpuCommandValidation = cfg.getBoolean("enableGpuCommandValidation", "general", false,
                    "Phase 2 shadow: validate GPU indirect commands vs CPU emitter (logs only).");
            enableGpuCommandGeneration = cfg.getBoolean("enableGpuCommandGeneration", "general", false,
                    "Phase 2: emit indirect draw commands on GPU. Requires phase 1 green first.");
        } catch (Exception e) {
            System.out.println("[Computium] config load failed, defaults OFF: " + e);
        } finally {
            if (cfg.hasChanged()) {
                cfg.save();
            }
        }
    }
}
