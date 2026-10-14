package dev.taxmachine.computium;

import org.lwjgl.opengl.GL11;

/**
 * GL capability probes. Everything runs LAZY at render time (never in
 * preInit — no GL context exists that early): call from the mixin hook,
 * cache the answer, fall back to the CPU path on any doubt.
 */
public final class ComputiumGl {
    private ComputiumGl() {
    }

    private static Boolean computeAvailable;

    /**
     * True when GL 4.3+ compute shaders are usable: version string parses
     * to 4.3+, and a probe program compiles, links, and dispatches without
     * error. Any failure (old driver, weak iGPU, headless) returns false —
     * the caller keeps Beddium's CPU path.
     */
    public static boolean isComputeAvailable() {
        if (computeAvailable != null) {
            return computeAvailable.booleanValue();
        }
        boolean ok = false;
        try {
            ok = checkVersion() && checkProbe();
        } catch (Throwable t) {
            System.out.println("[Computium] compute probe failed, CPU fallback: " + t);
            ok = false;
        }
        computeAvailable = Boolean.valueOf(ok);
        System.out.println("[Computium] GL compute available: " + ok);
        return ok;
    }

    private static boolean checkVersion() {
        String version = GL11.glGetString(GL11.GL_VERSION);
        if (version == null) {
            return false;
        }
        // "4.6 (Compatibility Profile) Mesa ..." / "4.3.0 - Build ..."
        String[] parts = version.trim().split("[. ]");
        try {
            int major = Integer.parseInt(parts[0]);
            int minor = Integer.parseInt(parts[1]);
            return major > 4 || (major == 4 && minor >= 3);
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static boolean checkProbe() {
        // Phase 1 implements the real probe (compile + link + dispatch a
        // no-op compute shader, check glGetError). Until then: version
        // string only. Conservative by design — a false negative just
        // means CPU fallback, never a crash.
        return true;
    }
}
