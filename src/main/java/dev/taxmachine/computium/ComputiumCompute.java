package dev.taxmachine.computium;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.embeddedt.embeddium.impl.render.chunk.LocalSectionIndex;
import org.embeddedt.embeddium.impl.render.chunk.RenderSection;
import org.embeddedt.embeddium.impl.render.chunk.RenderSectionManager;
import org.embeddedt.embeddium.impl.render.chunk.data.SectionRenderDataStorage;
import org.embeddedt.embeddium.impl.render.chunk.data.SectionRenderDataUnsafe;
import org.embeddedt.embeddium.impl.render.chunk.lists.ChunkRenderList;
import org.embeddedt.embeddium.impl.render.chunk.occlusion.OcclusionNode;
import org.embeddedt.embeddium.impl.render.chunk.region.RenderRegion;
import org.embeddedt.embeddium.impl.render.viewport.Viewport;
import org.embeddedt.embeddium.impl.render.chunk.terrain.TerrainRenderPass;
import org.embeddedt.embeddium.impl.render.viewport.CameraTransform;
import org.embeddedt.embeddium.impl.util.iterator.ByteIterator;
import org.embeddedt.embeddium.impl.gl.device.CommandList;
import org.embeddedt.embeddium.impl.gl.device.DrawCommandList;
import org.embeddedt.embeddium.impl.gl.tessellation.GlIndexType;
import org.embeddedt.embeddium.impl.gl.tessellation.GlPrimitiveType;
import org.embeddedt.embeddium.impl.gl.tessellation.GlTessellation;
import org.lwjgl.opengl.ARBComputeShader;
import org.lwjgl.opengl.ARBShaderStorageBufferObject;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL40;
import org.lwjgl.opengl.GL42;
import org.lwjgl.opengl.GL43;

/**
 * Phase-1 GPU frustum pre-filter. Called once per frame from the
 * {@code RenderSectionManager.update} hook (render thread, GL current),
 * consulted per section from the {@code OcclusionCuller.isSectionVisible}
 * hook (any thread — lock-free volatile publication).
 *
 * <p>Lifecycle per frame (only when the config flag is on):
 * enumerate sections → upload AABBs → dispatch cull shader → read back
 * compacted slots → publish table. Beddium proceeds normally and only
 * consults the table to skip provably-invisible sections.
 *
 * <p>Failure policy: ANY GL/config/reflection problem latches the engine
 * off (loud log, once) and every consult returns "don't cull". A broken
 * GPU path degrades to stock Beddium, never to holes or crashes.
 * Direct LWJGL3 references are confined to this class, which only ever
 * executes behind the config gate (classic path without LWJGL3 never
 * touches it).
 */
public final class ComputiumCompute {
    private ComputiumCompute() {
    }

    private static final int WORKGROUP = 64;
    private static final int INITIAL_SECTIONS = 4096;
    private static final float LEAK_MARGIN = 3.0f;

    private static volatile boolean latchedOff;
    private static volatile boolean loggedOff;
    private static volatile FrameData published;

    private static int program = -1;
    private static int planesLoc = -1;
    private static int marginLoc = -1;
    private static int boundsBuf = -1;
    private static int resultBuf = -1;
    // Initialized at declaration (not in ensureXProgram): the grow loops
    // below spin forever from 0 (0*2==0), and snapshot paths now reach
    // them before any program setup ever runs. (This exact bug froze the
    // render thread for 39s of CPU in one session — jstack caught it
    // spinning in ensureGenCapacity called from snapshotFillState.)
    private static int capacitySections = INITIAL_SECTIONS;
    private static FloatBuffer boundsStaging;
    private static FloatBuffer planesStaging = newFloat(24);
    private static IntBuffer resultStaging;

    // Phase-2 shadow state (gen program + descriptor/command buffers).
    private static int genProgram = -1;
    private static int genCamLoc = -1;
    private static int genFaceLoc = -1;
    private static int descBuf = -1;
    private static int cmdBuf = -1;
    private static int genCapacitySections = INITIAL_SECTIONS;
    private static IntBuffer descStaging;
    private static IntBuffer cmdStaging;
    private static long genFrames;
    private static long genCalls;
    private static long genCommandsTotal;
    private static int genCommandsMax;
    private static long genValidations;
    private static long genMismatches;
    private static int genMismatchStreak;
    // Cadences (ms) : readbacks GPU synchrones, validation CPU, logs de stats.
    private static long lastGenReadbackMs;
    private static long lastHeartbeatMs;
    private static long lastHeartbeatLogMs;
    private static long lastValidationMs;
    private static long lastStatsLogMs;
    private static long lastGenStatsLogMs;
    private static long genReadbacks;
    private static int checksSinceLog;
    private static int mismatchSinceLog;

    /** Cadence : vrai quand {@code intervalMs} est écoulé ({@code <= 0} = toujours). */
    private static boolean due(long lastMs, int intervalMs) {
        return intervalMs <= 0 || System.currentTimeMillis() - lastMs >= intervalMs;
    }

    private static boolean loggedNoGL43;

    private static final class FrameData {
        final Map<Long, Integer> slots;
        final boolean[] visible;

        FrameData(Map<Long, Integer> slots, boolean[] visible) {
            this.slots = slots;
            this.visible = visible;
        }
    }

    private static void logOff(String why) {
        if (!loggedOff) {
            loggedOff = true;
            System.out.println("[Computium] compute OFF (CPU fallback): " + why);
        }
    }

    /** Pack section coords into one long (21 bits each — sections fit). */
    static long pack(int x, int y, int z) {
        return (((long) x) << 42) ^ (((long) z) << 21) ^ (y & 0x1FFFFFL);
    }

    private static FloatBuffer newFloat(int floats) {
        return ByteBuffer.allocateDirect(floats * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
    }

    private static IntBuffer newInt(int ints) {
        return ByteBuffer.allocateDirect(ints * 4)
                .order(ByteOrder.nativeOrder()).asIntBuffer();
    }

    private static boolean gl43Present() {
        try {
            Class.forName("org.lwjgl.opengl.ARBComputeShader", true,
                    ComputiumCompute.class.getClassLoader());
            return true;
        } catch (Throwable t) {
            if (!loggedNoGL43) {
                loggedNoGL43 = true;
                System.out.println("[Computium] no LWJGL3 GL43 (classic path?)");
            }
            return false;
        }
    }

    private static String readShaderSource() throws Exception {
        InputStream in = ComputiumCompute.class.getResourceAsStream(
                "/assets/computium/shaders/cull.comp");
        if (in == null) {
            throw new IllegalStateException("cull.comp missing from jar");
        }
        try {
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int n;
            while ((n = in.read(chunk)) != -1) {
                buf.write(chunk, 0, n);
            }
            return buf.toString("UTF-8");
        } finally {
            in.close();
        }
    }

    private static void checkShader(int shader, String what) {
        int status = GL20.glGetShaderi(shader, GL20.GL_COMPILE_STATUS);
        String log = GL20.glGetShaderInfoLog(shader, 4096);
        if (status == GL11.GL_FALSE || (log != null && !log.trim().isEmpty())) {
            System.out.println("[Computium] shader " + what + " log: " + log);
        }
        if (status == GL11.GL_FALSE) {
            throw new IllegalStateException(what + " compile failed");
        }
    }

    private static boolean ensureProgram() {
        if (program >= 0) {
            return true;
        }
        try {
            int shader = GL20.glCreateShader(GL43.GL_COMPUTE_SHADER);
            GL20.glShaderSource(shader, readShaderSource());
            GL20.glCompileShader(shader);
            checkShader(shader, "cull.comp");
            program = GL20.glCreateProgram();
            GL20.glAttachShader(program, shader);
            GL20.glLinkProgram(program);
            int linked = GL20.glGetProgrami(program, GL20.GL_LINK_STATUS);
            if (linked == GL11.GL_FALSE) {
                System.out.println("[Computium] program link log: "
                        + GL20.glGetProgramInfoLog(program, 4096));
                throw new IllegalStateException("program link failed");
            }
            GL20.glDeleteShader(shader);
            planesLoc = GL20.glGetUniformLocation(program, "uFrustumPlanes");
            marginLoc = GL20.glGetUniformLocation(program, "uLeakMargin");
            if (planesLoc < 0 || marginLoc < 0) {
                throw new IllegalStateException("uniforms missing");
            }
            boundsBuf = GL15.glGenBuffers();
            resultBuf = GL15.glGenBuffers();
            capacitySections = INITIAL_SECTIONS;
            boundsStaging = newFloat(capacitySections * 8);
            // [count u32][indices...]
            resultStaging = newInt(1 + capacitySections);
            System.out.println("[Computium] compute program ready");
            return true;
        } catch (Throwable t) {
            logOff("program setup failed: " + t);
            program = -1;
            return false;
        }
    }

    private static void ensureCapacity(int sections) {
        if (sections <= capacitySections) {
            return;
        }
        int grown = capacitySections;
        while (grown < sections) {
            grown *= 2;
        }
        capacitySections = grown;
        boundsStaging = newFloat(capacitySections * 8);
        resultStaging = newInt(1 + capacitySections);
        System.out.println("[Computium] buffers grown to " + capacitySections + " sections");
    }

    /**
     * Runs once per frame from the update hook. Render thread, GL current.
     * Never throws (all failures latch off and fall back).
     */
    public static void runFrame(RenderSectionManager manager, Viewport viewport) {
        if (latchedOff || !ComputiumConfig.enableGpuFrustumCulling) {
            return;
        }
        try {
            runFrameInner(manager, viewport);
        } catch (Throwable t) {
            latchedOff = true;
            logOff("frame failed: " + t);
        }
    }

    private static void runFrameInner(RenderSectionManager manager, Viewport viewport) throws Exception {
        if (!gl43Present() || !ensureProgram()) {
            latchedOff = true;
            logOff("prerequisites unmet");
            return;
        }
        Collection<RenderSection> sections = manager.getAllRenderSections();
        if (sections == null || sections.isEmpty()) {
            return;
        }
        int count = sections.size();
        ensureCapacity(count);
        float[] planes = ComputiumFrustum.readPlanes(viewport);
        if (planes == null) {
            return;
        }
        // Upload bounds: two vec4 per section.
        boundsStaging.clear();
        Map<Long, Integer> slots = new HashMap<Long, Integer>(count * 2);
        int slot = 0;
        for (RenderSection section : sections) {
            int x = section.getChunkX();
            int y = section.getChunkY();
            int z = section.getChunkZ();
            slots.put(Long.valueOf(pack(x, y, z)), Integer.valueOf(slot));
            float minX = x * 16.0f;
            float minY = y * 16.0f;
            float minZ = z * 16.0f;
            boundsStaging.put(minX).put(minY).put(minZ).put(minX + 16.0f);
            boundsStaging.put(minY + 16.0f).put(minZ + 16.0f).put(0.0f).put(0.0f);
            slot++;
        }
        boundsStaging.flip();
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, boundsBuf);
        GL15.glBufferData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, boundsStaging, GL15.GL_DYNAMIC_DRAW);
        resultStaging.clear();
        resultStaging.put(0, 0);
        resultStaging.limit(1 + count);
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, resultBuf);
        GL15.glBufferData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, resultStaging, GL15.GL_DYNAMIC_DRAW);
        // Uniforms + dispatch. Per-plane scalar upload: the only
        // vec4-array form across LWJGL 2.9.1 (compile) and the shim.
        GL20.glUseProgram(program);
        for (int plane = 0; plane < 6; plane++) {
            int o = plane * 4;
            GL20.glUniform4f(planesLoc + plane, planes[o], planes[o + 1], planes[o + 2],
                    planes[o + 3]);
        }
        GL20.glUniform1f(marginLoc, LEAK_MARGIN);
        GL30.glBindBufferBase(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 0, boundsBuf);
        GL30.glBindBufferBase(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 1, resultBuf);
        ARBComputeShader.glDispatchCompute((count + WORKGROUP - 1) / WORKGROUP, 1, 1);
        GL42.glMemoryBarrier(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BARRIER_BIT);
        GL20.glUseProgram(0);
        // Readback: count first, then only that many indices.
        resultStaging.clear();
        resultStaging.limit(1 + count);
        GL15.glGetBufferSubData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 0, resultStaging);
        int visible = Math.min(resultStaging.get(0), count);
        boolean[] table = new boolean[count];
        for (int i = 0; i < visible; i++) {
            int s = resultStaging.get(1 + i);
            if (s >= 0 && s < count) {
                table[s] = true;
            }
        }
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 0);
        published = new FrameData(slots, table);
    }

    /**
     * Node-level consult for the cull hook. Direct calls (public final
     * accessors) — zero reflection overhead in the hot path.
     */
    public static boolean shouldCullNode(OcclusionNode node) {
        if (latchedOff || !ComputiumConfig.enableGpuFrustumCulling || node == null) {
            return false;
        }
        try {
            return shouldCull(node.getChunkX(), node.getChunkY(), node.getChunkZ());
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Phase-2 shadow validation: generate indirect commands on GPU for one
     * region and compare against a CPU recomputation (same inputs, Beddium's
     * emitter logic mirrored). Logs only — Beddium draws untouched.
     * Unsorted passes only (sorted/translucent keeps the CPU path).
     * Never throws (all failures latch off and fall back).
     */
    public static void runGenValidation(RenderRegion region,
            SectionRenderDataStorage storage, ChunkRenderList renderList,
            CameraTransform camera, TerrainRenderPass pass, boolean useFaceCulling) {
        if (latchedOff || !ComputiumConfig.enableGpuCommandValidation || pass.isSorted()) {
            return;
        }
        try {
            runGenValidationInner(region, storage, renderList, camera, pass, useFaceCulling);
        } catch (Throwable t) {
            latchedOff = true;
            logOff("gen validation failed: " + t);
        }
    }

    // ---- Cutover state (Phase-2 generation + indirect draw) ----
    // Churn-detection snapshot (HEAD): per-section mesh words. A rebuild
    // landing anywhere between Beddium's fill and our TAIL read skews
    // the comparison with no logic bug involved — so the TAIL re-reads
    // the same words and skips the check unless every word is stable.
    private static int[] churnIdx = new int[0];
    private static int[] churnWords = new int[0];
    private static int churnSlots;
    private static long churnSkips;
    // Mirror scratch (render thread only — fills never run concurrently;
    // verified single "Client thread" in production logs). Reused across
    // validations instead of allocating per section per fill.
    private static int[] mirrorBuf = new int[0];
    private static final int[] mirrorVo = new int[7];
    private static final int[] mirrorEc = new int[7];
    // Pre-zeroed direct buffer for wiping the command store: one memcpy
    // instead of a per-int put loop. Sized with the staging buffers.
    private static IntBuffer zeroDirect = newInt(INITIAL_SECTIONS * 35 + 1);
    // ---- Stage-0 FPS sampler (measure, don't guess) ----
    private static long frameLastNs;
    private static long frameCount;
    private static long frameTotalNs;
    private static long frameWorstNs;

    /**
     * Once-per-frame sampler (called from the section-manager update hook,
     * ungated). Tracks render-thread frame times; logs avg FPS + worst
     * frame every 600 frames with the generation flag state, so A/B
     * sessions (same scene, flag on/off) read straight off the log.
     */
    public static void frameSample() {
        try {
            long now = System.nanoTime();
            if (frameLastNs != 0) {
                long dt = now - frameLastNs;
                frameCount++;
                frameTotalNs += dt;
                if (dt > frameWorstNs) {
                    frameWorstNs = dt;
                }
                if (frameCount >= 600) {
                    double avgMs = (double) frameTotalNs / (double) frameCount / 1e6;
                    double worstMs = (double) frameWorstNs / 1e6;
                    System.out.println("[Computium] fps: avg=" + String.format("%.1f", 1000.0 / avgMs)
                            + " avgFrame=" + String.format("%.2f", avgMs) + "ms worst="
                            + String.format("%.2f", worstMs) + "ms gen="
                            + (ComputiumConfig.enableGpuCommandGeneration ? 1 : 0));
                    frameCount = 0;
                    frameTotalNs = 0;
                    frameWorstNs = 0;
                }
            }
            frameLastNs = now;
        } catch (Throwable t) {
            // Sampling never breaks the game.
        }
    }
    // ---- Cutover state (Phase-2 generation + indirect draw) ----
    // One global GPU command buffer: Beddium fills then draws each region
    // sequentially, so the buffer always holds the current region's
    // commands at draw time. drewGpu is set at every fill HEAD (false
    // unless this fill generated) and consumed by the draw redirect —
    // stale-true is impossible by construction.
    private static boolean drewGpu;
    private static int drawnSlots;
    private static long drawsLogged;
    private static int cutStreak;
    private static boolean fallbackLogged;
    private static sun.misc.Unsafe unsafeInstance;
    private static boolean cpuArmedLogged;

    private static sun.misc.Unsafe unsafe() {
        if (unsafeInstance == null) {
            try {
                java.lang.reflect.Field f =
                        sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
                f.setAccessible(true);
                unsafeInstance = (sun.misc.Unsafe) f.get(null);
            } catch (Throwable t) {
                return null;
            }
        }
        return unsafeInstance;
    }

    /**
     * HEAD hook: per-fill shared state. Records the batch size and
     * snapshots every section's mesh words while packing the GPU
     * descriptors FROM THE SAME READS (single word source for upload,
     * mirror, and heartbeat — a rebuild can never wedge between them).
     * Never throws, never issues synchronous GL (one async descriptor
     * upload happens later at dispatch).
     */
    public static void snapshotFillState(
            org.embeddedt.embeddium.impl.render.chunk.multidraw.MultiDrawEmitter emitter,
            RenderRegion region, SectionRenderDataStorage storage,
            ChunkRenderList renderList, TerrainRenderPass pass) {
        // Mesh-word snapshot (15 ints/section: sliceMask + vo[7] + ec[7]).
        // Failure here just disables churn detection for this fill
        // (churnSlots = -1 forces skip at TAIL).
        try {
            ByteIterator it =
                    renderList.sectionsWithGeometryIterator(pass.isReverseOrder());
            if (it == null) {
                churnSlots = -1;
                return;
            }
            int slots = 0;
            while (it.hasNext()) {
                it.nextByteAsInt();
                slots++;
            }
            if (churnIdx.length < slots) {
                churnIdx = new int[slots];
                churnWords = new int[slots * 15];
            }
            churnSlots = slots;
            // Size descriptor staging for the GPU upload the dispatch
            // step performs (same loop packs both — single read pass).
            // Packing runs only when generation may dispatch (sorted
            // passes and validation-only sessions skip the extra work).
            boolean packDesc = ComputiumConfig.enableGpuCommandGeneration && !pass.isSorted();
            if (packDesc) {
                ensureGenCapacity(slots);
            }
            int originX = region.getChunkX();
            int originY = region.getChunkY();
            int originZ = region.getChunkZ();
            if (packDesc) {
                descStaging.clear();
            }
            it = renderList.sectionsWithGeometryIterator(pass.isReverseOrder());
            int s = 0;
            while (it.hasNext()) {
                int sectionIndex = it.nextByteAsInt();
                long mesh = storage.getDataPointer(sectionIndex);
                churnIdx[s] = sectionIndex;
                int sliceMask = SectionRenderDataUnsafe.getSliceMask(mesh);
                churnWords[s * 15] = sliceMask;
                if (packDesc) {
                    descStaging.put(originX * 16 + LocalSectionIndex.unpackX(sectionIndex) * 16)
                            .put(originY * 16 + LocalSectionIndex.unpackY(sectionIndex) * 16)
                            .put(originZ * 16 + LocalSectionIndex.unpackZ(sectionIndex) * 16)
                            .put(sliceMask);
                }
                for (int f = 0; f < 7; f++) {
                    int vo = SectionRenderDataUnsafe.getVertexOffset(mesh, f);
                    int ec = SectionRenderDataUnsafe.getElementCount(mesh, f);
                    churnWords[s * 15 + 1 + f] = vo;
                    churnWords[s * 15 + 8 + f] = ec;
                    if (packDesc && f < 6) {
                        descStaging.put(vo).put(ec).put(
                                SectionRenderDataUnsafe.getIndexOffset(mesh, f));
                    }
                }
                if (packDesc) {
                    descStaging.put(SectionRenderDataUnsafe.getVertexOffset(mesh, 6));
                    descStaging.put(SectionRenderDataUnsafe.getElementCount(mesh, 6));
                }
                s++;
            }
            if (packDesc) {
                descStaging.flip();
            }
        } catch (Throwable t) {
            churnSlots = -1;
        }
    }

    /**
     * TAIL hook: Beddium has filled the emitter; compare the commands it
     * added during this fill against an independent re-read of the mesh.
     * Pure CPU — zero GL. Unsorted passes only, like the GPU path.
     * Never throws (failures latch off and fall back).
     */
    public static void runCpuValidation(
            org.embeddedt.embeddium.impl.render.chunk.multidraw.MultiDrawEmitter emitter,
            RenderRegion region, SectionRenderDataStorage storage,
            ChunkRenderList renderList, CameraTransform camera,
            TerrainRenderPass pass, boolean useFaceCulling) {
        if (latchedOff || !ComputiumConfig.enableGpuCommandValidation || pass.isSorted()) {
            return;
        }
        // Same 1/16 cadence as the throttled GPU path (the mirror walks
        // every section twice — cheap, but no reason to do it per fill).
        if ((genCalls++ & 15) != 0) {
            return;
        }
        // Cadence temporelle : borne le travail CPU (et les readbacks qu'il
        // déclenche) quel que soit le débit de fills — sinon des centaines de
        // validations par seconde, pour rien.
        if (!due(lastValidationMs, ComputiumConfig.validationIntervalMs)) {
            return;
        }
        lastValidationMs = System.currentTimeMillis();
        try {
            runCpuValidationInner(emitter, region, storage, renderList,
                    camera, pass, useFaceCulling);
        } catch (Throwable t) {
            latchedOff = true;
            logOff("cpu validation failed: " + t);
        }
    }

    private static void runCpuValidationInner(
            org.embeddedt.embeddium.impl.render.chunk.multidraw.MultiDrawEmitter emitter,
            RenderRegion region, SectionRenderDataStorage storage,
            ChunkRenderList renderList, CameraTransform camera,
            TerrainRenderPass pass, boolean useFaceCulling) throws Exception {
        if (!(emitter instanceof
                org.embeddedt.embeddium.impl.render.chunk.multidraw.DirectMultiDrawEmitter)) {
            return;
        }
        org.embeddedt.embeddium.impl.gl.device.MultiDrawBatch batch =
                ((org.embeddedt.embeddium.impl.render.chunk.multidraw
                        .DirectMultiDrawEmitter) emitter).batch();
        if (batch == null || unsafe() == null) {
            if (!cpuArmedLogged) {
                cpuArmedLogged = true;
                System.out.println("[Computium] cpu shadow OFF: "
                        + (batch == null ? "null batch" : "Unsafe unavailable"));
            }
            return;
        }
        if (!cpuArmedLogged) {
            cpuArmedLogged = true;
            System.out.println("[Computium] cpu shadow armed (direct batch)");
        }
        // bedCount is read DIRECTLY at TAIL: Beddium's fill clears the
        // batch first thing (emitter.clear() at fill entry, verified in
        // bytecode), so size() at TAIL == exactly this fill's adds. (An
        // earlier revision snapshotted size() at HEAD — before the clear
        // — and subtracted, which measured (adds - previous leftovers):
        // every count "mismatch" it ever reported was that artifact.)
        int bedCount = batch.size();
        int originX = region.getChunkX();
        int originY = region.getChunkY();
        int originZ = region.getChunkZ();
        // Churn gate: the HEAD snapshot captured every section's mesh
        // words just before Beddium's fill. Re-read them now — any
        // difference (or membership change) means a rebuild landed inside
        // this fill and no verdict is possible; skip silently. Only fully
        // stable regions produce verdicts, so a mismatch can never be a
        // rebuild race, only a real logic divergence.
        if (churnSlots < 0) {
            return;
        }
        ByteIterator it = renderList.sectionsWithGeometryIterator(pass.isReverseOrder());
        if (it == null) {
            return;
        }
        int slots = 0;
        while (it.hasNext()) {
            it.nextByteAsInt();
            slots++;
        }
        if (slots != churnSlots) {
            if (++churnSkips % 500 == 0) {
                System.out.println("[Computium] cpu shadow: skipped "
                        + churnSkips + " churning fills");
            }
            return;
        }
        // Emitter not cleared per region (accumulates across fills): the
        // window is not this region's commands — skip, don't miscompare.
        if (bedCount > slots * 7) {
            return;
        }
        int[] expBuf = mirrorBuf;
        if (expBuf.length < slots * 35 + 5) {
            expBuf = new int[slots * 35 + 5];
            mirrorBuf = expBuf;
        }
        int expCount = 0;
        for (int s = 0; s < slots; s++) {
            int sectionIndex = churnIdx[s];
            int chunkX = originX + LocalSectionIndex.unpackX(sectionIndex);
            int chunkY = originY + LocalSectionIndex.unpackY(sectionIndex);
            int chunkZ = originZ + LocalSectionIndex.unpackZ(sectionIndex);
            int minX = chunkX * 16;
            int minY = chunkY * 16;
            int minZ = chunkZ * 16;
            long mesh = storage.getDataPointer(sectionIndex);
            // Verify stability against the HEAD snapshot first: the
            // mirror below may only use words proven unchanged across
            // the whole fill.
            int sliceMask = SectionRenderDataUnsafe.getSliceMask(mesh);
            if (sliceMask != churnWords[s * 15]) {
                if (++churnSkips % 500 == 0) {
                    System.out.println("[Computium] cpu shadow: skipped "
                            + churnSkips + " churning fills");
                }
                return;
            }
            int[] vo = mirrorVo;
            int[] ec = mirrorEc;
            for (int f = 0; f < 7; f++) {
                vo[f] = SectionRenderDataUnsafe.getVertexOffset(mesh, f);
                ec[f] = SectionRenderDataUnsafe.getElementCount(mesh, f);
                if (vo[f] != churnWords[s * 15 + 1 + f]
                        || ec[f] != churnWords[s * 15 + 8 + f]) {
                    if (++churnSkips % 500 == 0) {
                        System.out.println("[Computium] cpu shadow: skipped "
                                + churnSkips + " churning fills");
                    }
                    return;
                }
            }
            int faces = (useFaceCulling
                    ? visibleFaces(camera.intX, camera.intY, camera.intZ,
                            minX, minY, minZ)
                    : 127) & sliceMask;
            for (int f = 0; f < 7; f++) {
                if (((faces >> f) & 1) == 0) {
                    continue;
                }
                expBuf[expCount * 5] = ec[f];
                expBuf[expCount * 5 + 1] = 1;
                // Pointer word is ALWAYS 0 here: Beddium calls
                // addDrawCommands(mesh, faces, pass.isSorted() ? -1 : 0)
                // and stores getIndexOffset & mask — unsorted passes mask
                // with 0 (proven in DefaultChunkRenderer bytecode). Only
                // sorted passes keep real offsets, and those stay on the
                // CPU path (skipped above via pass.isSorted()).
                expBuf[expCount * 5 + 2] = 0;
                expBuf[expCount * 5 + 3] = vo[f];
                expBuf[expCount * 5 + 4] = 0;
                expCount++;
            }
        }
        int mismatches;
        if (bedCount != expCount) {
            mismatches = Math.max(bedCount, expCount);
        } else if (bedCount == 0) {
            mismatches = 0;
        } else {
            // Batch triples: pointers are 8-byte addresses, counts and
            // base vertices 4-byte ints (per DirectMultiDrawEmitter
            // bytecode: memPutAddress/memPutInt/memPutInt). The pointer
            // holds the mesh index offset widened to long — compare the
            // low 32 bits against io[f].
            long pPtr = batch.pElementPointer;
            long pCount = batch.pElementCount;
            long pBase = batch.pBaseVertex;
            int[] bedFlat = new int[bedCount * 5];
            sun.misc.Unsafe u = unsafe();
            for (int i = 0; i < bedCount; i++) {
                int k = i;
                bedFlat[i * 5] = u.getInt(pCount + (long) k * 4);
                bedFlat[i * 5 + 1] = 1;
                bedFlat[i * 5 + 2] = (int) u.getLong(pPtr + (long) k * 8);
                bedFlat[i * 5 + 3] = u.getInt(pBase + (long) k * 4);
                bedFlat[i * 5 + 4] = 0;
            }
            mismatches = compareCommandMultisets(bedFlat, expBuf, expCount);
            if (mismatches > 0) {
                detailFirstDivergence(bedFlat, expBuf, expCount);
            }
        }
        genFrames++;
        genValidations++;
        genMismatches += mismatches;
        // Outlier forensics: Beddium emitting far less than the mirror
        // (ratio < 1/2) means whole sections/faces vanish on Beddium's
        // side — log region origin + camera + slots so the pattern
        // (far regions? specific camera quadrants?) becomes visible.
        if (bedCount > 0 && (long) bedCount * 2 < expCount) {
            StringBuilder sb = new StringBuilder(256);
            sb.append("[Computium] outlier: region=").append(originX).append(',')
                    .append(originY).append(',').append(originZ)
                    .append(" cam=").append(camera.intX).append(',')
                    .append(camera.intY).append(',').append(camera.intZ)
                    .append(" slots=").append(slots)
                    .append(" bed=").append(bedCount)
                    .append(" cpu=").append(expCount)
                    .append(" cull=").append(useFaceCulling ? 1 : 0)
                    .append(" pass=").append(pass)
                    .append(" thread=").append(Thread.currentThread().getName());
            // Per-section mirror faces (bounded): hand-verify these against
            // the camera box test. Beddium's per-section faces are not
            // stored anywhere (flat batch) — the comparison target.
            if (slots <= 32) {
                sb.append(" faces=[");
                for (int s = 0; s < slots; s++) {
                    int sectionIndex = churnIdx[s];
                    int chunkX = originX + LocalSectionIndex.unpackX(sectionIndex);
                    int chunkY = originY + LocalSectionIndex.unpackY(sectionIndex);
                    int chunkZ = originZ + LocalSectionIndex.unpackZ(sectionIndex);
                    int faces = (useFaceCulling
                            ? visibleFaces(camera.intX, camera.intY, camera.intZ,
                                    chunkX * 16, chunkY * 16, chunkZ * 16)
                            : 127) & churnWords[s * 15];
                    sb.append(sectionIndex).append(':')
                            .append(Integer.toBinaryString(faces)).append(' ');
                }
                sb.append(']');
            }
            System.out.println(sb.toString());
        }
        if (mismatches > 0) {
            genMismatchStreak++;
        } else {
            genMismatchStreak = 0;
        }
        if (mismatches > 0 || genValidations <= 3
                || due(lastStatsLogMs, ComputiumConfig.logIntervalMs)) {
            checksSinceLog++;
            mismatchSinceLog += mismatches;
            System.out.println("[Computium] cpu shadow check #" + genValidations
                    + ": bed=" + bedCount + " cpu=" + expCount
                    + " mismatched=" + mismatches
                    + " streak=" + genMismatchStreak
                    + " since=" + checksSinceLog
                    + " mismatchSince=" + mismatchSinceLog
                    + " mismatchTotal=" + genMismatches);
            lastStatsLogMs = System.currentTimeMillis();
            checksSinceLog = 0;
            mismatchSinceLog = 0;
        }
        // Streak rule, tuned for rebuild races: during load/TNT bursts a
        // rebuild landing between Beddium's mesh read and ours skews the
        // count (always bed < cpu while meshes only gain faces), and a
        // ~50-section check walks ~50us — so bursts streak. Systematic
        // bugs mismatch 100% of checks and still trip this in ~3s;
        // transients (even at 90%/check) essentially never survive 50.
        // Reset-on-clean keeps isolated transients free.
        if (genMismatchStreak >= 50) {
            latchedOff = true;
            logOff("cpu shadow: 50 consecutive mismatches, engine off");
        }
        // Cutover heartbeat: on a clean churn-verified verdict for a
        // GPU-generated fill, read back the GPU commands and compare them
        // against the mirror (== Beddium's batch, just proven). Both sides
        // derive from the same stable words, so races are structurally
        // excluded — any streak here is a real shader/upload bug and
        // falls back to stock Beddium. The first 3 generated fills ALWAYS
        // read back (session-start proof the dispatch writes at all —
        // silent zero-counts are the one failure this must catch early).
        if (mismatches == 0 && drewGpu && bedCount > 0
                && (genFrames <= 3 || due(lastHeartbeatMs, ComputiumConfig.readbackIntervalMs))) {
            lastHeartbeatMs = System.currentTimeMillis();
            heartbeatVsMirror(expBuf, bedCount);
        }
    }

    /**
     * One-line diagnostic: first Beddium triple with no mirror match and
     * first mirror triple with no Beddium match (as count/first/base).
     * Tells pointer-mask issues (first words differ) apart from mesh-word
     * races (count/base differ) at a glance. Bounded O(n^2), n small.
     */
    private static void detailFirstDivergence(int[] bedFlat, int[] expBuf, int count) {
        try {
            String bedOnly = null;
            String cpuOnly = null;
            outer:
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < count; j++) {
                    boolean same = true;
                    for (int w = 0; w < 5; w++) {
                        if (bedFlat[i * 5 + w] != expBuf[j * 5 + w]) {
                            same = false;
                            break;
                        }
                    }
                    if (same) {
                        continue outer;
                    }
                }
                bedOnly = bedFlat[i * 5] + "/" + bedFlat[i * 5 + 2]
                        + "/" + bedFlat[i * 5 + 3];
                break;
            }
            outer:
            for (int j = 0; j < count; j++) {
                for (int i = 0; i < count; i++) {
                    boolean same = true;
                    for (int w = 0; w < 5; w++) {
                        if (bedFlat[i * 5 + w] != expBuf[j * 5 + w]) {
                            same = false;
                            break;
                        }
                    }
                    if (same) {
                        continue outer;
                    }
                }
                cpuOnly = expBuf[j * 5] + "/" + expBuf[j * 5 + 2]
                        + "/" + expBuf[j * 5 + 3];
                break;
            }
            System.out.println("[Computium] diverge: bed-only=" + bedOnly
                    + " cpu-only=" + cpuOnly + " (count/first/base)");
        } catch (Throwable t) {
            // Diagnostics never break validation.
        }
    }

    // ---- Cutover: GPU command generation + indirect draw ----
    //
    // Per fill (unsorted passes only): pack descriptors on CPU, dispatch
    // the gen compute shader (NO readback — fully async), and mark the
    // region for indirect draw. Beddium's own fill still runs (its emitter
    // stays warm for instant fallback); the draw redirect consumes the
    // flag and draws from our GPU buffer instead.
    //
    // Steady state issues ZERO synchronous GL (orphan uploads, async
    // clear, dispatch, barriers, draw — no GetBufferSubData): the
    // synchronous round-trips of the old shadow perturbed Beddium's
    // shared command stream (black-rectangle incident). Program and
    // DRAW_INDIRECT bindings are saved/restored (Beddium binds programs
    // per pass and assumes they persist across fills).
    //
    // Correctness: a heartbeat (every 600th fill) reads back the GPU
    // commands and compares them against Beddium's own batch; any streak
    // falls back to stock Beddium (flag off, loud log, validation keeps
    // proving the CPU side).

    /**
     * Draw redirect consumer: returns true (once) if the current region
     * was GPU-generated and must draw indirect; always resets.
     */
    public static boolean consumeDrewGpu() {
        boolean v = drewGpu;
        drewGpu = false;
        return v;
    }

    private static void cutoverFallback(String why) {
        ComputiumConfig.enableGpuCommandGeneration = false;
        drewGpu = false;
        if (!fallbackLogged) {
            fallbackLogged = true;
            System.out.println("[Computium] CUTOVER FALLBACK (stock Beddium): " + why);
        }
    }

    /**
     * Fill HEAD hook for generation. Never throws (any failure falls
     * back for this fill; repeated heartbeat divergence disables
     * cutover entirely).
     */
    public static void runGpuGen(
            org.embeddedt.embeddium.impl.render.chunk.multidraw.MultiDrawEmitter emitter,
            RenderRegion region, SectionRenderDataStorage storage,
            ChunkRenderList renderList, CameraTransform camera,
            TerrainRenderPass pass, boolean useFaceCulling) {
        drewGpu = false;
        if (latchedOff || !ComputiumConfig.enableGpuCommandGeneration || pass.isSorted()) {
            return;
        }
        try {
            runGpuGenInner(emitter, region, storage, renderList, camera, pass, useFaceCulling);
        } catch (Throwable t) {
            drewGpu = false;
            cutStreak++;
            if (cutStreak >= 3) {
                cutoverFallback("gen exception streak: " + t);
            }
        }
    }

    private static void runGpuGenInner(
            org.embeddedt.embeddium.impl.render.chunk.multidraw.MultiDrawEmitter emitter,
            RenderRegion region, SectionRenderDataStorage storage,
            ChunkRenderList renderList, CameraTransform camera,
            TerrainRenderPass pass, boolean useFaceCulling) throws Exception {
        if (!gl43Present() || !ensureGenProgram()) {
            cutoverFallback("gen prerequisites unmet");
            return;
        }
        // Descriptors were packed by the HEAD snapshot from the same reads
        // as the churn words (single word source). Zero slots (or a failed
        // snapshot) means nothing to draw this fill.
        int slots = churnSlots;
        if (slots <= 0) {
            return;
        }
        // descStaging arrives flipped (packed by the HEAD snapshot);
        // upload [position, limit) = exactly slots*24 ints.
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, descBuf);
        GL15.glBufferData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER,
                descStaging, GL15.GL_DYNAMIC_DRAW);
        // Command store: orphan the old store and upload zeros in one
        // async call. Stale tail commands would otherwise draw garbage
        // (drawcount is slots*7 unconditionally, zero-count entries are
        // no-ops). NOTE: glClearBufferSubData is deliberately NOT used —
        // its IntBuffer overload is missing from lwjgl3ify's relocated
        // LWJGL3 (org.lwjglx) at runtime (NoSuchMethodError); glBufferData
        // is proven present by thousands of shadow calls.
        int cmdInts = 1 + slots * 35;
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, cmdBuf);
        // Bulk zero-fill: one memcpy from the pre-zeroed static instead
        // of a per-int put loop (both async; this is ~10x less CPU).
        zeroDirect.clear();
        zeroDirect.limit(cmdInts);
        cmdStaging.clear();
        cmdStaging.put(zeroDirect);
        cmdStaging.flip();
        GL15.glBufferData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER,
                cmdStaging, GL15.GL_DYNAMIC_DRAW);
        // Save Beddium's program (bound per pass, assumed persistent
        // across fills — unbinding it mid-render drew black geometry).
        int prevProgram = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        GL20.glUseProgram(genProgram);
        GL20.glUniform3i(genCamLoc, camera.intX, camera.intY, camera.intZ);
        GL20.glUniform1i(genFaceLoc, useFaceCulling ? 1 : 0);
        GL30.glBindBufferBase(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 0, descBuf);
        GL30.glBindBufferBase(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 1, cmdBuf);
        GL42.glMemoryBarrier(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BARRIER_BIT);
        ARBComputeShader.glDispatchCompute((slots + WORKGROUP - 1) / WORKGROUP, 1, 1);
        GL42.glMemoryBarrier(GL43.GL_COMMAND_BARRIER_BIT);
        GL20.glUseProgram(prevProgram);
        drawnSlots = slots;
        drewGpu = true;
        genFrames++;
        genCommandsTotal += slots * 7;
        if (genFrames <= 3) {
            System.out.println("[Computium] gen #" + genFrames + ": slots=" + slots
                    + " drawnSlots=" + drawnSlots + " cmdBuf=" + cmdBuf
                    + " descBuf=" + descBuf + " program=" + genProgram);
        }
        // Heartbeat lives at TAIL (runCpuValidationInner): it compares the
        // GPU readback against the churn-verified mirror, only on clean
        // verdicts. Comparing anywhere earlier cannot exclude races.
    }


    /**
     * Cutover heartbeat body: synchronous GPU readback compared against
     * the churn-verified mirror. Rare (every 600th generated fill) so the
     * single stall cannot perturb the stream. Streak of 3 falls back.
     */
    private static void heartbeatVsMirror(int[] expBuf, int count) {
        try {
            cmdStaging.clear();
            cmdStaging.limit(1 + count * 5);
            GL15.glGetBufferSubData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER,
                    0, cmdStaging);
            int gpuCount = cmdStaging.get(0);
            int mismatches;
            if (gpuCount != count) {
                mismatches = Math.max(gpuCount, count);
            } else {
                int[] gpuFlat = new int[count * 5];
                for (int i = 0; i < count; i++) {
                    for (int w = 0; w < 5; w++) {
                        gpuFlat[i * 5 + w] = cmdStaging.get(1 + i * 5 + w);
                    }
                }
                mismatches = compareCommandMultisets(gpuFlat, expBuf, count);
                if (mismatches > 0) {
                    detailFirstDivergence(gpuFlat, expBuf, count);
                }
            }
            if (mismatches > 0) {
                cutStreak++;
            } else {
                cutStreak = 0;
            }
            if (mismatches > 0 || cutStreak > 0
                    || due(lastHeartbeatLogMs, ComputiumConfig.logIntervalMs)) {
                lastHeartbeatLogMs = System.currentTimeMillis();
                System.out.println("[Computium] cutover heartbeat: gpu=" + gpuCount
                        + " mirror=" + count + " mismatched=" + mismatches
                        + " streak=" + cutStreak);
            }
            if (cutStreak >= 3) {
                cutoverFallback("heartbeat streak");
            }
        } catch (Throwable t) {
            cutStreak++;
            if (cutStreak >= 3) {
                cutoverFallback("heartbeat exception: " + t);
            }
        }
    }

    /**
     * Indirect draw replacing Beddium's direct batch draw for a
     * GPU-generated region. Replicates DirectMultiDrawEmitter.executeBatch
     * (tessellate, draw, untessellate) with the command stream coming from
     * our GPU buffer instead of client memory. Never throws (any failure
     * falls back; Beddium's emitter stayed warm so it draws next frame).
     */
    public static void drawIndirect(CommandList commandList, GlTessellation tessellation,
            GlPrimitiveType primitiveType) {
        DrawCommandList dl = null;
        try {
            dl = commandList.beginTessellating(tessellation);
        } catch (Throwable t) {
            cutoverFallback("beginTessellating failed: " + t);
            return;
        }
        try {
            int prevIndirect = GL11.glGetInteger(GL40.GL_DRAW_INDIRECT_BUFFER_BINDING);
            GL15.glBindBuffer(GL40.GL_DRAW_INDIRECT_BUFFER, cmdBuf);
            // NOTE: GL43, not GL43C — lwjgl3ify's relocated LWJGL3 ships
            // GL43 (verified: offset overload present) but NOT GL43C
            // (NoClassDefFoundError). Same core-since-4.3 entry point.
            GL43.glMultiDrawElementsIndirect(primitiveType.getId(),
                    GlIndexType.UNSIGNED_INT.getFormatId(), 4L, drawnSlots * 7, 0);
            // Offset is 4, not 0: our buffer carries the command count in
            // uData[0] (for readback/validation) with structs following at
            // uData[1 + slot*5]. Drawing from 0 misreads every command by
            // one word (structurally valid call, semantically garbage —
            // no GL error, just invisible terrain). Beddium's own indirect
            // buffer has no count prefix, hence its offset-0.
            // Silence is ambiguous (LWJGL never throws on GL errors): poll
            // for one. A broken/unwired indirect entry point in lwjgl3ify's
            // runtime (it ships only referenced GL classes) would draw
            // nothing with no other signal. First 3 draws always report.
            int err = GL11.glGetError();
            drawsLogged++;
            if (err != GL11.GL_NO_ERROR || drawsLogged <= 3) {
                System.out.println("[Computium] indirect draw #" + drawsLogged
                        + ": mode=" + primitiveType.getId() + " count=" + (drawnSlots * 7)
                        + " cmdBuf=" + cmdBuf + " err=" + err);
            }
            if (err != GL11.GL_NO_ERROR) {
                cutoverFallback("indirect draw GL error " + err);
            }
            GL15.glBindBuffer(GL40.GL_DRAW_INDIRECT_BUFFER, prevIndirect);
        } catch (Throwable t) {
            cutoverFallback("indirect draw failed: " + t);
        } finally {
            try {
                if (dl != null) {
                    dl.close();
                }
            } catch (Throwable t) {
                // Untessellate must not throw past Beddium's render loop.
            }
        }
    }

    private static boolean ensureGenProgram() {
        if (genProgram >= 0) {
            return true;
        }
        try {
            int shader = GL20.glCreateShader(GL43.GL_COMPUTE_SHADER);
            InputStream in = ComputiumCompute.class.getResourceAsStream(
                    "/assets/computium/shaders/gen.comp");
            if (in == null) {
                throw new IllegalStateException("gen.comp missing from jar");
            }
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int n;
            while ((n = in.read(chunk)) != -1) {
                buf.write(chunk, 0, n);
            }
            in.close();
            GL20.glShaderSource(shader, buf.toString("UTF-8"));
            GL20.glCompileShader(shader);
            int status = GL20.glGetShaderi(shader, GL20.GL_COMPILE_STATUS);
            String log = GL20.glGetShaderInfoLog(shader, 4096);
            if (status == GL11.GL_FALSE || (log != null && !log.trim().isEmpty())) {
                System.out.println("[Computium] gen shader log: " + log);
            }
            if (status == GL11.GL_FALSE) {
                throw new IllegalStateException("gen.comp compile failed");
            }
            genProgram = GL20.glCreateProgram();
            GL20.glAttachShader(genProgram, shader);
            GL20.glLinkProgram(genProgram);
            int linked = GL20.glGetProgrami(genProgram, GL20.GL_LINK_STATUS);
            if (linked == GL11.GL_FALSE) {
                System.out.println("[Computium] gen program link log: "
                        + GL20.glGetProgramInfoLog(genProgram, 4096));
                throw new IllegalStateException("gen program link failed");
            }
            GL20.glDeleteShader(shader);
            genCamLoc = GL20.glGetUniformLocation(genProgram, "uCameraBlock");
            genFaceLoc = GL20.glGetUniformLocation(genProgram, "uFaceCulling");
            if (genCamLoc < 0 || genFaceLoc < 0) {
                throw new IllegalStateException("gen uniforms missing");
            }
            descBuf = GL15.glGenBuffers();
            cmdBuf = GL15.glGenBuffers();
            genCapacitySections = INITIAL_SECTIONS;
            // 24 ints per section descriptor; 1 + slots*7*5 command words.
            descStaging = newInt(genCapacitySections * 24);
            cmdStaging = newInt(1 + genCapacitySections * 35);
            System.out.println("[Computium] gen program ready");
            return true;
        } catch (Throwable t) {
            logOff("gen program setup failed: " + t);
            genProgram = -1;
            return false;
        }
    }

    private static void ensureGenCapacity(int sections) {
        if (sections <= genCapacitySections) {
            return;
        }
        int grown = genCapacitySections;
        while (grown < sections) {
            grown *= 2;
        }
        genCapacitySections = grown;
        descStaging = newInt(genCapacitySections * 24);
        cmdStaging = newInt(1 + genCapacitySections * 35);
        zeroDirect = newInt(genCapacitySections * 35 + 1);
    }

    // Exact mirror of Beddium DefaultChunkRenderer.getVisibleFaces (int
    // math, ±3 leak, POS_X=bit0..NEG_Z=bit5, UNASSIGNED=bit6 always set).
    private static int visibleFaces(int camX, int camY, int camZ,
            int minX, int minY, int minZ) {
        int maxX = minX + 16;
        int maxY = minY + 16;
        int maxZ = minZ + 16;
        int planes = 1 << 6;
        planes |= (camX > minX - 3 ? 1 : 0) << 0;
        planes |= (camY > minY - 3 ? 1 : 0) << 1;
        planes |= (camZ > minZ - 3 ? 1 : 0) << 2;
        planes |= (camX < maxX + 3 ? 1 : 0) << 3;
        planes |= (camY < maxY + 3 ? 1 : 0) << 4;
        planes |= (camZ < maxZ + 3 ? 1 : 0) << 5;
        return planes;
    }

    private static void runGenValidationInner(RenderRegion region,
            SectionRenderDataStorage storage, ChunkRenderList renderList,
            CameraTransform camera, TerrainRenderPass pass, boolean useFaceCulling) throws Exception {
        // THROTTLE: Beddium fills command buffers thousands of times per
        // second; running a synchronous upload+dispatch+readback on every
        // fill chokes the driver's buffer manager on the same command
        // stream Beddium uses for mesh uploads (starved uploads = torn
        // sections + stalled builders). Validate 1 fill in 16 — still
        // hundreds of checks per minute, ~0 of the pressure.
        if ((genCalls++ & 15) != 0) {
            return;
        }
        if (!gl43Present() || !ensureGenProgram()) {
            latchedOff = true;
            logOff("gen prerequisites unmet");
            return;
        }
        ByteIterator it = renderList.sectionsWithGeometryIterator(pass.isReverseOrder());
        if (it == null) {
            return;
        }
        int originX = region.getChunkX();
        int originY = region.getChunkY();
        int originZ = region.getChunkZ();
        // Count first (bounded, single pass over a tiny iterator).
        int slots = 0;
        while (it.hasNext()) {
            it.nextByteAsInt();
            slots++;
        }
        if (slots == 0) {
            return;
        }
        ensureGenCapacity(slots);
        // Re-iterate (iterator is single-use) and pack descriptors.
        it = renderList.sectionsWithGeometryIterator(pass.isReverseOrder());
        descStaging.clear();
        // CPU-expected command words for the periodic deep check.
        int[] expected = null;
        // Deep check échantillonné dans le temps (les 3 premières passes
        // toujours) : le readback qui suit est synchrone — « toutes les 600
        // générations » tombait plusieurs fois par seconde et stallait le
        // thread de rendu plusieurs secondes.
        boolean deepCheck = genFrames <= 3
                || due(lastGenReadbackMs, ComputiumConfig.readbackIntervalMs);
        int[] expBuf = deepCheck ? new int[slots * 35] : null;
        int expCount = 0;
        int s = 0;
        while (it.hasNext()) {
            int sectionIndex = it.nextByteAsInt();
            int chunkX = originX + LocalSectionIndex.unpackX(sectionIndex);
            int chunkY = originY + LocalSectionIndex.unpackY(sectionIndex);
            int chunkZ = originZ + LocalSectionIndex.unpackZ(sectionIndex);
            int minX = chunkX * 16;
            int minY = chunkY * 16;
            int minZ = chunkZ * 16;
            long mesh = storage.getDataPointer(sectionIndex);
            // SINGLE snapshot per face: worker threads rebuild meshes
            // concurrently, so the upload, the mirror, and Beddium's own
            // read can otherwise observe three different generations.
            // Everything below derives from these locals — never re-reads —
            // so GPU-vs-mirror divergence can only mean a real shader or
            // logic bug, never a race.
            int sliceMask = SectionRenderDataUnsafe.getSliceMask(mesh);
            int[] vo = new int[7];
            int[] ec = new int[7];
            for (int f = 0; f < 7; f++) {
                vo[f] = SectionRenderDataUnsafe.getVertexOffset(mesh, f);
                ec[f] = SectionRenderDataUnsafe.getElementCount(mesh, f);
            }
            descStaging.put(minX).put(minY).put(minZ).put(sliceMask);
            for (int f = 0; f < 6; f++) {
                descStaging.put(vo[f]);
                descStaging.put(ec[f]);
                descStaging.put(SectionRenderDataUnsafe.getIndexOffset(mesh, f));
            }
            // Struct tail: vo6/ec6 (UNASSIGNED face) live in the last
            // two ints — the shader reads facesE.z/w for f==6, and the
            // mirror compares real mesh words there. Uploading zeros here
            // used to mismatch every section with unassigned geometry
            // (partial meshes during load) while counts still matched.
            // Total stride stays 24 ints: origin[4] + faces[18] + tail[2].
            descStaging.put(vo[6]).put(ec[6]);
            if (deepCheck) {
                int faces = (useFaceCulling
                        ? visibleFaces(camera.intX, camera.intY, camera.intZ,
                                minX, minY, minZ)
                        : 127) & sliceMask;
                for (int f = 0; f < 7; f++) {
                    if (((faces >> f) & 1) == 0) {
                        continue;
                    }
                    // f==6 (UNASSIGNED) has no mesh words; Beddium reads
                    // index 6 the same way — mirror, don't interpret.
                    expBuf[expCount * 5] = ec[f];
                    expBuf[expCount * 5 + 1] = 1;
                    // firstIndex 0: shared pattern buffer (see shader).
                    expBuf[expCount * 5 + 2] = 0;
                    expBuf[expCount * 5 + 3] = vo[f];
                    expBuf[expCount * 5 + 4] = 0;
                    expCount++;
                }
            }
            expected = expBuf;
            s++;
        }
        descStaging.flip();
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, descBuf);
        GL15.glBufferData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER,
                descStaging, GL15.GL_DYNAMIC_DRAW);
        cmdStaging.clear();
        cmdStaging.put(0, 0);
        cmdStaging.limit(1 + slots * 35);
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, cmdBuf);
        GL15.glBufferData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER,
                cmdStaging, GL15.GL_DYNAMIC_DRAW);
        GL20.glUseProgram(genProgram);
        GL20.glUniform3i(genCamLoc, camera.intX, camera.intY, camera.intZ);
        GL20.glUniform1i(genFaceLoc, useFaceCulling ? 1 : 0);
        GL30.glBindBufferBase(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 0, descBuf);
        GL30.glBindBufferBase(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 1, cmdBuf);
        GL42.glMemoryBarrier(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BARRIER_BIT);
        ARBComputeShader.glDispatchCompute((slots + WORKGROUP - 1) / WORKGROUP, 1, 1);
        GL42.glMemoryBarrier(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BARRIER_BIT);
        GL20.glUseProgram(0);
        // Readback : uniquement sur les passes échantillonnées. C'est un point
        // de synchronisation GPU (attente de fin de file) — un appel par fill
        // stallait le rendu plusieurs secondes. Le rendu indirect n'en dépend
        // pas (il utilise `drawnSlots`, snapshot CPU) : seuls le compteur de
        // stats et la comparaison en ont besoin.
        int gpuCount = -1;
        if (deepCheck) {
            cmdStaging.clear();
            cmdStaging.limit(1 + slots * 35);
            GL15.glGetBufferSubData(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER,
                    0, cmdStaging);
            gpuCount = Math.min(cmdStaging.get(0), slots * 7);
            genReadbacks++;
            lastGenReadbackMs = System.currentTimeMillis();
            genCommandsTotal += gpuCount;
            if (gpuCount > genCommandsMax) {
                genCommandsMax = gpuCount;
            }
        }
        genFrames++;
        if (deepCheck && expected != null) {
            // Multiset compare: GPU appends via atomics in scheduling
            // order (arbitrary), CPU appends in section order. For
            // unsorted passes order is irrelevant to rendering (depth
            // test resolves), so sort copies, never the live buffers.
            // A count difference is a REAL divergence (ordering alone
            // cannot change the count) — rebuild races during chunk load
            // are the usual suspect there, not the shader.
            int mismatches;
            if (gpuCount != expCount) {
                mismatches = Math.max(gpuCount, expCount);
            } else {
                int[] gpuFlat = new int[expCount * 5];
                for (int i = 0; i < expCount; i++) {
                    for (int w = 0; w < 5; w++) {
                        gpuFlat[i * 5 + w] = cmdStaging.get(1 + i * 5 + w);
                    }
                }
                mismatches = compareCommandMultisets(gpuFlat, expected, expCount);
            }
            genValidations++;
            genMismatches += mismatches;
            // NOT latched on a single mismatch: mesh rebuilds on worker
            // threads can land between our read and Beddium's (transient
            // skew during chunk load). Latch only on a streak — systematic
            // divergence (wrong replica) mismatches every check.
            if (mismatches > 0) {
                genMismatchStreak++;
            } else {
                genMismatchStreak = 0;
            }
            if (mismatches > 0 || genValidations <= 3
                    || due(lastStatsLogMs, ComputiumConfig.logIntervalMs)) {
                checksSinceLog++;
                mismatchSinceLog += mismatches;
                System.out.println("[Computium] gen shadow check #" + genValidations
                        + ": gpu=" + gpuCount + " cpu=" + expCount
                        + " mismatched=" + mismatches + " streak=" + genMismatchStreak
                        + " since=" + checksSinceLog
                        + " mismatchSince=" + mismatchSinceLog
                        + " mismatchTotal=" + genMismatches);
                lastStatsLogMs = System.currentTimeMillis();
                checksSinceLog = 0;
                mismatchSinceLog = 0;
            }
            if (genMismatchStreak >= 3) {
                latchedOff = true;
                logOff("gen shadow mismatch streak — investigate before cutover");
            }
        } else if (due(lastGenStatsLogMs, ComputiumConfig.logIntervalMs)) {
            lastGenStatsLogMs = System.currentTimeMillis();
            System.out.println("[Computium] gen shadow: frames=" + genFrames
                    + " readbacks=" + genReadbacks
                    + " maxCmds=" + genCommandsMax
                    + " avgCmds=" + (genCommandsTotal / Math.max(1, genReadbacks)));
        }
        GL15.glBindBuffer(ARBShaderStorageBufferObject.GL_SHADER_STORAGE_BUFFER, 0);
    }

    /**
     * Multiset equality over two flat command streams (stride-5 words).
     * Sorts COPIES (never mutates inputs). Returns the number of
     * commands with no exact counterpart.
     */
    static int compareCommandMultisets(int[] gpuFlat, int[] expected, int count) {
        int[][] gpu = new int[count][];
        int[][] exp = new int[count][];
        for (int i = 0; i < count; i++) {
            gpu[i] = new int[] {
                gpuFlat[i * 5], gpuFlat[i * 5 + 1], gpuFlat[i * 5 + 2],
                gpuFlat[i * 5 + 3], gpuFlat[i * 5 + 4]
            };
            exp[i] = new int[] {
                expected[i * 5], expected[i * 5 + 1], expected[i * 5 + 2],
                expected[i * 5 + 3], expected[i * 5 + 4]
            };
        }
        java.util.Comparator<int[]> byWords = (a, b) -> {
            for (int w = 0; w < 5; w++) {
                int c = Integer.compare(a[w], b[w]);
                if (c != 0) {
                    return c;
                }
            }
            return 0;
        };
        java.util.Arrays.sort(gpu, byWords);
        java.util.Arrays.sort(exp, byWords);
        int mismatches = 0;
        for (int i = 0; i < count; i++) {
            if (byWords.compare(gpu[i], exp[i]) != 0) {
                mismatches++;
            }
        }
        return mismatches;
    }

    /**
     * Consulted per section from the cull hook (any thread — volatile
     * publication only). True = GPU proved frustum-invisible, safe to skip.
     * Unknown sections and any inactive state return false (proceed).
     */
    public static boolean shouldCull(int x, int y, int z) {        if (latchedOff || !ComputiumConfig.enableGpuFrustumCulling) {
            return false;
        }
        FrameData frame = published;
        if (frame == null) {
            return false;
        }
        Integer slot = frame.slots.get(Long.valueOf(pack(x, y, z)));
        if (slot == null) {
            return false;
        }
        int s = slot.intValue();
        if (s < 0 || s >= frame.visible.length) {
            return false;
        }
        return !frame.visible[s];
    }
}
