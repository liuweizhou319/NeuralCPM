add_rules("mode.debug", "mode.release")
set_languages("cxx17")
set_runtimes("MD")

local libigl = os.getenv("LIBIGL_ROOT") or "D:/DATA/codes/DiriNormal/deps/libigl"
local external = path.join(libigl, "external")
local eigen = path.join(external, "eigen")
local gmm = path.join(external, "CoMISo", "ext", "gmm-4.2", "include")
local vendor = path.join(os.projectdir(), "vendor")
local xmake_packages = path.join(os.getenv("LOCALAPPDATA") or "C:/Users/omen/AppData/Local", ".xmake", "packages")
local polyscope = path.join(xmake_packages, "p", "polyscope", "v2.5.0", "b8aae154d8cd4b889064923059b590e2")
local glad = path.join(xmake_packages, "g", "glad", "v0.1.36", "728361337c5c4347a2bbe35cbbe1c092")
local glfw = path.join(xmake_packages, "g", "glfw", "3.4", "d59f6dd71e7647cc95d2c1d35e51a39a")
local glm = path.join(xmake_packages, "g", "glm", "1.0.1", "18615f0387cd4e2ab2e9ea9965afc784")
local imgui = path.join(xmake_packages, "i", "imgui", "v1.91.0-docking", "bca072a9c3f8430db3b8da3959a39d22")
local implot = path.join(xmake_packages, "i", "implot", "v0.17", "3da744c0e9a14cc099f23a91bc309732")
local json = path.join(xmake_packages, "n", "nlohmann_json", "v3.12.0", "ab8414047c424750bdc9d7a0ff7afdde")
local stb = path.join(xmake_packages, "s", "stb", "2025.03.14", "5add428b704648039c039d4d74bed865")
local common_defines = {
    "NOMINMAX",
    "_USE_MATH_DEFINES",
    "INCLUDE_TEMPLATES",
    "_SCL_SECURE_NO_DEPRECATE",
    "_CRT_SECURE_NO_WARNINGS"
}

target("comiso")
    set_kind("static")
    add_defines(table.unpack(common_defines))
    add_includedirs(vendor, eigen, external, gmm)
    add_files(
        "vendor/CoMISo/Solver/EigenLDLTSolver.cc",
        "vendor/CoMISo/Solver/MISolver.cc",
        "vendor/CoMISo/Solver/ConstrainedSolver.cc",
        "vendor/CoMISo/Solver/IterativeSolverT.cc",
        "vendor/CoMISo/Solver/GMM_Tools.cc"
    )

target("surfaceframefield_igl")
    set_kind("binary")
    add_deps("comiso")
    add_defines(table.unpack(common_defines))
    add_defines("GLM_ENABLE_EXPERIMENTAL", "GLFW_INCLUDE_NONE")
    add_includedirs(
        vendor,
        eigen,
        path.join(libigl, "include"),
        external,
        gmm,
        path.join(polyscope, "include"),
        path.join(glad, "include"),
        path.join(glfw, "include"),
        path.join(glm, "include"),
        path.join(imgui, "include"),
        path.join(imgui, "include", "backends"),
        path.join(imgui, "include", "misc", "cpp"),
        path.join(implot, "include"),
        path.join(json, "include"),
        path.join(stb, "include"),
        path.join(stb, "include", "stb")
    )
    add_linkdirs(
        path.join(polyscope, "lib"),
        path.join(glad, "lib"),
        path.join(glfw, "lib"),
        path.join(imgui, "lib"),
        path.join(implot, "lib")
    )
    add_links("polyscope", "glad", "glfw3", "imgui", "implot")
    add_syslinks("opengl32", "imm32", "user32", "shell32", "gdi32")
    add_files("src/main.cpp")
