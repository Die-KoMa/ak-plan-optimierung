{ pkgs ? import <nixpkgs> { } }:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "akplan-opt-fhsnev";

    targetPkgs = _: [
      pkgs.micromamba
      pkgs.python311
    ];

    profile = ''
      set -e
      eval "$(micromamba shell hook --shell=posix)"
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
      set +e
    '';
  };
in
fhs.env


