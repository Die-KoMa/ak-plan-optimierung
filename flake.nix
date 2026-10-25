{
  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixos-25.11/nixexprs.tar.xz";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" ] (
          system:
          f (
            import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            }
          )
        );
      python' = p: p.python3;
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          python = python' pkgs;
        in
        rec {
          ak-plan-optimierung =
            let
              src = pkgs.fetchFromGitHub {
                owner = "die-koma";
                repo = "ak-plan-optimierung";
                rev = "e5453e225369820bab3e2d294f0f226bffcabc58";
                hash = "sha256-a2thR6ntXU5CAuCbg37BIHj8QhpHjkb/0js9w2sxKPQ=";
              };
            in
            python.pkgs.buildPythonApplication {
              name = "ak-plan-optimierung";
              version = "0.0";
              pyproject = true;
              build-system = [ python.pkgs.setuptools ];
              nativeBuildInputs = [ python.pkgs.setuptools-scm ];

              inherit src;

              dependencies = pkgs.lib.attrValues {
                inherit (python.pkgs)
                  dacite
                  tqdm
                  numpy
                  pandas
                  xarray
                  gurobipy
                  highspy
                  pytest
                  pytest-timeout
                  ;
                inherit (self.packages.${pkgs.stdenv.hostPlatform.system})
                  linopy
                  ;
              };

              meta = {
                license = pkgs.lib.licenses.mit;
                mainProgram = "akplan-solve";
              };
            };

          linopy = python.pkgs.buildPythonPackage rec {
            pname = "linopy";
            version = "0.5.8";
            pyproject = true;

            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-pN1mEFRKJ50KL3YguQChi1FzBhzZAtXCBiHElXbFwnE=";
            };

            build-system = [ python.pkgs.setuptools ];
            nativeBuildInputs = [
              python.pkgs.setuptools-scm
              pkgs.which
            ];

            dependencies = pkgs.lib.attrValues {
              inherit (python.pkgs)
                numpy
                scipy
                bottleneck
                toolz
                numexpr
                xarray
                dask
                polars
                tqdm
                deprecation
                packaging
                gurobipy
                requests
                google-cloud-storage
                ;
            };

            pythonImportsCheck = [ "linopy" ];
          };

          default = ak-plan-optimierung;
        }
      );

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = pkgs.lib.attrValues {
            inherit ((python' pkgs).pkgs)
              black
              ruff
              pytest
              mypy
              coverage
              ;
            inherit (pkgs) gurobi;
          };
          inputsFrom = [ self.packages.${pkgs.stdenv.hostPlatform.system}.ak-plan-optimierung ];
        };
      });
    };
}
