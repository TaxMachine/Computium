package dev.taxmachine.computium;

import com.gtnewhorizon.gtnhmixins.ILateMixinLoader;
import cpw.mods.fml.common.Mod;
import cpw.mods.fml.common.event.FMLPreInitializationEvent;
import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.apache.logging.log4j.Logger;

@Mod(modid = Computium.MODID, version = "@VERSION@", acceptableRemoteVersions = "*")
@io.github.tox1cozz.mixinbooterlegacy.LateMixin
public class Computium
        implements
            ILateMixinLoader,
            io.github.tox1cozz.mixinbooterlegacy.ILateMixinLoader {
    public static final String MODID = "computium";

    private static Logger logger;
    private static boolean configLoaded = false;

    // GTNHMixins late path.
    @Override
    public String getMixinConfig() {
        return "computium.mixin.json";
    }

    @Override
    public List<String> getMixins(Set<String> loadedMods) {
        // Mixins are listed statically in the config; nothing conditional.
        return Collections.emptyList();
    }

    // MixinBooterLegacy late path (the one that logs "Instantiating all
    // ILateMixinLoader implemented classes" — this is what actually queues
    // the config in this stack).
    @Override
    public List<String> getMixinConfigs() {
        return Collections.singletonList("computium.mixin.json");
    }

    @Override
    public boolean shouldMixinConfigQueue(String config) {
        return true;
    }

    @Override
    public void onMixinConfigQueued(String config) {
    }

    @Mod.EventHandler
    public void preInit(FMLPreInitializationEvent event) {
        logger = event.getModLog();
        ComputiumConfig.init(new File(event.getModConfigurationDirectory(), "computium.cfg"));
        configLoaded = true;
        logger.info("[Computium] config loaded (frustum={}, commands={}, validate={})",
                ComputiumConfig.enableGpuFrustumCulling, ComputiumConfig.enableGpuCommandGeneration,
                ComputiumConfig.enableGpuCommandValidation);
    }

    public static Logger logger() {
        return logger;
    }

    public static boolean isConfigLoaded() {
        return configLoaded;
    }
}
