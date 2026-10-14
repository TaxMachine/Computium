# Computium — GPU chunk pipeline for 1.7.10

Computium generates terrain draw commands on the GPU and draws them with
indirect multidraw calls, instead of Beddium's CPU fill + direct draws.
It is a companion to Beddium (1.2.3-j21) on the modernized 1.7.10 stack
(lwjgl3ify + UniMixins + Java 21), running everything on the existing
GL 4.6 context — no ROCm/CUDA, no extra SDKs.

By TaxMachine.

## Status

Phase 2 is cut over and validated live, every session:

- **Generate**: per fill (unsorted passes only), section descriptors are
  packed on CPU and a compute shader emits `DrawElementsIndirectCommand`
  structs into a GPU command buffer. Fully async — no readbacks in the
  steady state.
- **Draw**: a redirect on Beddium's draw call issues
  `glMultiDrawElementsIndirect` from that buffer. Sorted/translucent
  passes stay on Beddium's CPU path untouched.
- **Validate**: a zero-GL shadow continuously compares an independent
  CPU mirror against Beddium's own batch (churn-gated, race-proof), and
  a periodic heartbeat compares the GPU buffer against the mirror.
- **Fallback**: any systematic divergence disables generation with a
  loud `CUTOVER FALLBACK` log line — Beddium's own emitter stays warm,
  so rendering continues on the stock path seamlessly.

Phase 1 (GPU frustum culling) is implemented but parked OFF: Beddium
already frustum-culls on CPU for microseconds, so a GPU round-trip buys
nothing there. Honest accounting: the cutover is roughly
performance-neutral today (the CPU descriptor walk remains); it is the
correctness gate for GPU-resident submission later.

## Requirements

- Minecraft 1.7.10 + Forge 10.13.4.1614, Beddium 1.2.3-j21 present.
- Java 21 runtime (tested on GraalVM CE 21); GL 4.3+ context
  (4.6 compat confirmed on Mesa/RADV).
- Below GL 4.3, or without Beddium, every phase degrades to stock
  rendering silently.

## Config (`config/computium.cfg`, all default OFF)

- `enableGpuCommandValidation` — zero-GL shadow validation (logs only).
- `enableGpuCommandGeneration` — the cutover: GPU gen + indirect draws.
- `enableGpuFrustumCulling` — Phase 1 (parked; leave OFF).

## Build

```sh
export JAVA_HOME=<jdk8u504+>   # Gradle/ForgeGradle run on Java 8
./gradlew build                # → build/libs/computium.jar
```

Compilation forks a Java 21 `javac` (Beddium's sources need it; emitted
bytecode stays 1.8). It is auto-detected from `../../toolchain`
(`jdk-21*`), or point `$COMPUTIUM_JAVAC21` at a `javac` binary.
`mods/` holds the compile-only deps (UniMixins API, Beddium).

Mixin registration needs all three on the `@Mod` class (documented so
nobody re-breaks it): `@LateMixin` + GTNHMixins `ILateMixinLoader` +
MixinBooterLegacy `ILateMixinLoader`, and NO coremod.

## Layout

- `src/main/java/dev/taxmachine/computium/` — `Computium` (mod entry),
  `ComputiumCompute` (gen/draw/validate/fallback), `ComputiumConfig`,
  `ComputiumFrustum` + `ComputiumGl` (Phase-1 support, parked).
- `.../mixin/` — fill/draw hooks, occlusion + section-manager hooks.
- `src/main/resources/assets/computium/shaders/` — `gen.comp`
  (command generation), `cull.comp` (Phase-1 frustum, parked).

## Portability notes (learned the hard way)

- Beddium 1.2.3-j21 draws terrain DIRECT from client-memory triples,
  not indirect — the shadow reads those triples; the cutover reroutes
  to indirect draws separately.
- lwjgl3ify's runtime LWJGL3 is partial: only call entry points
  Beddium itself uses or that are verified present (`GL43`, never
  `GL43C`; no `glClearBufferSubData(IntBuffer)` overload).
- Never do synchronous GL (readbacks, stalls) inside Beddium's fill
  path, and always save/restore the bound program around foreign GL —
  Beddium binds programs per pass and assumes persistence.
