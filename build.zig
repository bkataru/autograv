const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build the autograv core library as a shared library
    const lib = b.addSharedLibrary(.{
        .name = "autograv_core",
        .root_source_file = b.path("src/zig/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Export C ABI for FFI compatibility
    lib.linkLibC();

    // Install the library
    b.installArtifact(lib);

    // Create a test executable
    const tests = b.addTest(.{
        .root_source_file = b.path("src/zig/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
