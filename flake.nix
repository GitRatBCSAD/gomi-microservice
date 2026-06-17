{
  description = "Python environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311 
            uv
            ruff
            basedpyright
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib # Required for C++ bindings (numpy, pandas)
              pkgs.zlib             # Required by many ML tools
              pkgs.glib             # Common UI/core library
            ]}"
            
            # Automatically create the virtual env if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Initializing uv virtual environment..."
              uv venv
            fi
            
            # Automatically activate it
            source .venv/bin/activate
          '';
        };
      }
    );
}
