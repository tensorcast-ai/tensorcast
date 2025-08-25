platform(
    name = "docker_image_platform",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
        # "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "OSFamily": "Linux",
        "container-image": "docker://yuchuluo/stepcast-cuda:12.4.0-devel-ubuntu22.04",
        "dockerNetwork": "off",
    },
)
