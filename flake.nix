{
  inputs = {
    mysystem.url = "git:/home/lorenzo/nix-config";
  };
  outputs = { self, mysystem, ... }@inputs:
    let
      pkgs = import mysystem.inputs.nixpkgs { system = "x86_64-linux"; };
      fhs = pkgs.buildFHSUserEnv {
        name = "my-fhs-environment";

        targetPkgs = pkgs: [
          pkgs.micromamba
        ];

        # profile = ''
        #   set -e
        #   eval "$(micromamba shell hook --shell=posix)"
        #   export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
        #   set +e
        # '';
      };
    in
    {
      devShells."x86_64-linux".default = fhs.env;
    };
}

