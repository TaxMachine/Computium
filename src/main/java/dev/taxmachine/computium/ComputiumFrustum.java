package dev.taxmachine.computium;

import java.lang.reflect.Field;
import org.embeddedt.embeddium.impl.render.viewport.Viewport;

/**
 * Reads Beddium's exact frustum planes (no re-derivation, no drift):
 * {@code Viewport.frustum} → {@code SimpleFrustum.frustum} →
 * JOML {@code FrustumIntersection} 24 plane floats, in NX/PX/NY/PY/NZ/PZ
 * order (matching {@code cull.comp}'s {@code uFrustumPlanes}).
 *
 * <p>All reflection is resolved once and cached; ANY failure (Beddium
 * bump, JOML bump, numbering change) latches to {@code null} and the
 * caller keeps the CPU path. Versions pinned: Beddium 1.2.3-j21,
 * JOML 1.10.8 (Beddium deploader).
 */
public final class ComputiumFrustum {
    private ComputiumFrustum() {
    }

    private static volatile boolean resolved;
    private static volatile boolean available;
    private static Field viewportFrustum;
    private static Field simpleFrustum;
    private static Field[] planeFields;
    private static boolean logged;

    // JOML 1.10.8 field order: nxX..nxW, pxX..pxW, ny.., py.., nz.., pz...
    private static final String[] PLANE_FIELDS = {
        "nxX", "nxY", "nxZ", "nxW",
        "pxX", "pxY", "pxZ", "pxW",
        "nyX", "nyY", "nyZ", "nyW",
        "pyX", "pyY", "pyZ", "pyW",
        "nzX", "nzY", "nzZ", "nzW",
        "pzX", "pzY", "pzZ", "pzW",
    };

    private static void logOnce(String msg) {
        if (!logged) {
            logged = true;
            System.out.println("[Computium] " + msg);
        }
    }

    private static synchronized void resolve() {
        if (resolved) {
            return;
        }
        resolved = true;
        try {
            ClassLoader loader = ComputiumFrustum.class.getClassLoader();
            Class<?> viewportClass = Class.forName(
                    "org.embeddedt.embeddium.impl.render.viewport.Viewport", true, loader);
            Class<?> simpleClass = Class.forName(
                    "org.embeddedt.embeddium.impl.render.viewport.frustum.SimpleFrustum", true, loader);
            Class<?> jomlClass = Class.forName(
                    "org.joml.FrustumIntersection", true, loader);
            viewportFrustum = field(viewportClass, "frustum");
            simpleFrustum = field(simpleClass, "frustum");
            planeFields = new Field[PLANE_FIELDS.length];
            for (int i = 0; i < PLANE_FIELDS.length; i++) {
                planeFields[i] = field(jomlClass, PLANE_FIELDS[i]);
            }
            available = true;
            logOnce("frustum reflection resolved (Beddium exact planes)");
        } catch (Throwable t) {
            available = false;
            logOnce("frustum reflection unavailable, CPU fallback (" + t.getClass().getSimpleName() + ")");
        }
    }

    private static Field field(Class<?> clazz, String name) throws NoSuchFieldException {
        Field f = clazz.getDeclaredField(name);
        f.setAccessible(true);
        return f;
    }

    /**
     * 24 floats (6 planes × xyzw, NX/PX/NY/PY/NZ/PZ), or null when
     * unavailable — caller keeps the CPU path.
     */
    public static float[] readPlanes(Viewport viewport) {
        if (!resolved) {
            resolve();
        }
        if (!available) {
            return null;
        }
        try {
            Object simple = viewportFrustum.get(viewport);
            if (simple == null) {
                return null;
            }
            Object joml = simpleFrustum.get(simple);
            if (joml == null) {
                return null;
            }
            float[] planes = new float[24];
            for (int i = 0; i < 24; i++) {
                planes[i] = planeFields[i].getFloat(joml);
            }
            return planes;
        } catch (Throwable t) {
            available = false;
            logOnce("frustum read failed, CPU fallback (" + t.getClass().getSimpleName() + ")");
            return null;
        }
    }
}
