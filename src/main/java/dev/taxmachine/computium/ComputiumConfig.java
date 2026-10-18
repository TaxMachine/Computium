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

    /**
     * Cadence (ms) des readbacks GPU synchrones (compteur de commandes de la
     * phase 2 + heartbeat de cutover). Un readback {@code glGetBufferSubData}
     * attend que le GPU ait fini tout le travail en file : à chaque passe il
     * provoquait des stalls de plusieurs secondes et des milliers de lignes de
     * log. Les 3 premières passes restent toujours échantillonnées (preuve de
     * démarrage). 0 = chaque passe (mode dev).
     */
    public static int readbackIntervalMs = 2000;

    /**
     * Cadence (ms) de la validation CPU shadow (en plus du 1/16 existant).
     * 0 = pas de limite temporelle (comportement dev).
     */
    public static int validationIntervalMs = 500;

    /**
     * Cadence (ms) des lignes de stats ([Computium] cpu/gen shadow check,
     * cutover heartbeat, resume de generation). Les anomalies (mismatch) et
     * changements de streak restent toujours loggés, quelle que soit la
     * cadence. 0 = tout logger.
     */
    public static int logIntervalMs = 10000;

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
            readbackIntervalMs = cfg.getInt("readbackIntervalMs", "general", 2000, 0, 60000,
                    "Cadence (ms) des readbacks GPU synchrones (compteur gen + heartbeat). Les 3 premieres passes restent echantillonnees. 0 = chaque passe (stalls).");
            validationIntervalMs = cfg.getInt("validationIntervalMs", "general", 500, 0, 60000,
                    "Cadence (ms) de la validation CPU shadow (en plus du 1/16). 0 = pas de limite.");
            logIntervalMs = cfg.getInt("logIntervalMs", "general", 10000, 0, 600000,
                    "Cadence (ms) des lignes de stats. Les mismatches sont toujours logges. 0 = tout logger.");
        } catch (Exception e) {
            System.out.println("[Computium] config load failed, defaults OFF: " + e);
        } finally {
            if (cfg.hasChanged()) {
                cfg.save();
            }
        }
    }
}
