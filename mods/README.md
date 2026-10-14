# mods/ — compile-time references go here (NEVER committed, NEVER shipped)
#
# Drop these jars here so `javac` resolves the closed/reference classes:
#
#   unimixins-all.jar  (compileOnly Mixin API — provided by the launcher
#                       at runtime, never embedded; same rule as the shim
#                       template)
#   beddium-*.jar      (ONLY for reference checks; the mixin uses
#                       `targets = "..."` strings + `remap = false`, so the
#                       build does NOT need it — but a full reference check
#                       against the real Beddium is warmly recommended)
#
# Ratio rule: if the annotation processor cannot resolve a `targets =`
# entry, the build still succeeds (targets are strings). A target typo
# surfaces at GAME boot in the log — watch for it on the matrix.
