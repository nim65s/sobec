{
  description = "optimal control library for robot control under contact sequence.";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    # ndcurves is not yet upstream
    nixpkgs.url = "github:gepetto/nixpkgs/master";
    crocoddyl = {
      url = "github:LudovicDeMatteis/crocoddyl/topic/contact-6D-closed-loop";
      inputs = {
        flake-parts.follows = "flake-parts";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', system, ... }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.sobec;
            sobec = pkgs.python3Packages.toPythonModule (pkgs.callPackage ./package.nix { 
              inherit (inputs.crocoddyl.packages.${system}) crocoddyl;
            });
          };
        };
    };
}
