{
  cmake,
  crocoddyl,
  lib,
  pkg-config,
  python3Packages,
  stdenv,
  yaml-cpp,
}:

stdenv.mkDerivation rec {
  pname = "sobec";
  version = "1.4.0-unstable-2024-10-02";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./benchmark
      ./CMakeLists.txt
      ./examples
      ./include
      ./package.xml
      ./python
      ./src
      ./tests
    ];
  };

  cmakeFlags = [
    (lib.cmakeBool "BUILD_BENCHMARK" false)
    (lib.cmakeBool "BUILD_DOCUMENTATION" false)
    (lib.cmakeBool "BUILD_EXAMPLES" false)
    (lib.cmakeBool "BUILD_TESTING" false)
    (lib.cmakeBool "GENERATE_PYTHON_STUBS" false)
  ];

  nativeBuildInputs = [
    cmake
    pkg-config
    python3Packages.pythonImportsCheckHook
  ];
  buildInputs = [ yaml-cpp ];
  propagatedBuildInputs = [
    crocoddyl
    python3Packages.ndcurves
    python3Packages.tqdm
    python3Packages.pyyaml
  ];

  doCheck = true;
  pythonImportsCheck = [ "sobec" ];

  meta = {
    description = "Sandbox for optimal control explicitly for bipeds";
    homepage = "https://github.com/MeMory-of-MOtion/sobec/";
    license = lib.licenses.bsd2;
    maintainers = with lib.maintainers; [ nim65s ];
    platforms = lib.platforms.unix;
  };
}
